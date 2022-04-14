import mlflow
import torch
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer


class NewsInference:
    """Mlflow에 올라와 있는 transformer자연어 모델을 기반으로 새로운 문장과 데이터프레임의 관계를 예측합니다.

    Parameters
    ---------
    server_uri: str
            모델이 저장되어 있는 Mlflow server_uri
    model_name: str
            model runs에 들어갈 model 이름. mlflow server에서 사용자가 지정한 model_runs의 이름
    model: torch.nn
        사용할 모델로 instance내에 저장된다.  
    current_state: str
            가져오고 있는 모델의 상태. ex) Production      
    PRE_TRAINED_MODEL_NAME: str
        tokenizer가 사용할 PRE_TRAINED_MODEL_NAME의 이름. 사용할 모델과 PRE_TRAINED_MODEL_NAME의 정보가 맞아야 한다.
        참고: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
    """

    def __init__(
        self,
        server_uri,
        model_name,
        current_state="Production",
        pretrained_model_name="google/mobilebert-uncased",
    ):
        self.pretrained_model_name = pretrained_model_name
        self.server_uri = server_uri
        self.model_name = model_name
        self.model = self._load_model(model_name, current_state)

    def _load_model(self, model_name, current_state):
        """mlflow 저장된 모델에서 되어 있는 모델을 불러오는 함수

        Parameters
        ---------
        model_name: str
            model runs에 들어갈 model 이름. mlflow server에서 사용자가 지정한 model_runs의 이름
        current_state: str
            가져오고 있는 모델의 상태. ex) Production

        Return
        ---------
        model: torch.nn
            사전에 학습된 pytorch model
        """
        loaded_model = None

        try:
            tracking_server_uri = "{}".format(self.server_uri)
            mlflow.set_tracking_uri(tracking_server_uri)
            client = MlflowClient()
            filter_string = "name = '{}'".format(model_name)
            print(filter_string)
            result = client.search_model_versions(filter_string)
            print(result)
            for res in result:
                if res.current_stage == "{}".format(current_state):
                    deploy_version = res.version
            model_uri = client.get_model_version_download_uri(
                self.model_name, deploy_version
            )
            loaded_model = mlflow.pytorch.load_model(model_uri)

        except Exception as e:
            print("NewsInference cannot load model since {}".format(e))

        return loaded_model

    def _embedding(self, input_text):
        """input text가 들어오면 모델에 inference할 text를 torch model이 사용할 수 있게, input text를 embedding하는 함수.

        Parameters
        ---------
        input_text: str
            사용자가 넣을 문장 정보.

        Return
        ---------
        input_ids: tensor
            encoding된 단어들이 숫자로 표현된 결과
        attention_mask: tensor
            단어가 있는지의 여부를 표시하는 결과, 문장이 있으면 1, 문자가 없고 padding이 되어 있으면 0으로 표시된다.
        """
        device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, return_dict=False
        )

        encoded_review = tokenizer.encode_plus(
            input_text,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_review["input_ids"].to(device)
        attention_mask = encoded_review["attention_mask"].to(device)

        return input_ids, attention_mask

    def _inference(self, input_ids, attention_mask):
        """
        pytorch 모델과 embedding된 문장을 사용해서 문장이 긍정적인지, 부정적인지 분류한다.

        Parameters
        ---------
        input_ids: tensor
            encoding된 단어들이 숫자로 표현된 결과
        attention_mask: tensor
            단어가 있는지의 여부를 표시하는 결과, 문장이 있으면 1, 문자가 없고 padding이 되어 있으면 0으로 표시된다.

        Returns
            softmax_prob: tensor
                문장이 긍정적인지 부정적인지 확률로 나타낸 결과
            prediction: tensor
                문장이 긍정적인지 부정적인지 0 혹은 1로 나타낸 결과
        ---------
        """
        logits = self.model(input_ids, attention_mask)
        softmax_prob = torch.nn.functional.softmax(logits, dim=1)
        _, prediction = torch.max(softmax_prob, dim=1)

        return softmax_prob, prediction

    def inference_sentence(self, input_text: str):
        """inference 함수를 사용해서, 문장 단위로 문장이 긍정적인지, 부정적인지 보여준다.

        Parameters
        ---------
        input_text: str
            모델에 사용하고자 하는 문장

        Returns
        ---------
        softmax_prob: tensor
            문장이 긍정적인지 부정적인지 확률로 나타낸 결과
        prediction: tensor
            문장이 긍정적인지 부정적인지 0 혹은 1로 나타낸 결과
        ---------
        """
        input_ids, attention_mask = self._embedding(
            input_text)

        if self.model == None:
            self.model = self._load_model(self.model_name, self.current_state)
        class_prob, pred = self._inference(input_ids, attention_mask)

        return (
            class_prob.detach().cpu().numpy()[0],
            pred.detach().cpu().numpy()[0],
        )

    def inference_df(self, preprocessed_data):
        """
        inference 함수를 사용해서, dataframe 단위로 문장이 긍정적인지, 부정적인지 보여준다.

        Parameters
        ---------
        preprocessed_data: pandas.dataframe
            모델에 사용하고자 하는 데이터 프레임

        Return:
        ---------
        preprocessed_data: pandas.dataframe
            기존 데이터프레임에서 class_prob와 pred labedl column이 추가된 데이터프레임이 나온다.
        """
        preprocessed_data["class_prob"] = ""
        preprocessed_data["pred"] = ""
        for i in range(len(preprocessed_data)):
            (
                preprocessed_data["class_prob"][i],
                preprocessed_data["pred"][i],
            ) = self.inference_sentence(
                preprocessed_data["input_text"][i]
            )
        return preprocessed_data
