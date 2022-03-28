import pandas as pd
from config import PathConfig
from dataloader import DataIOSteam
from inference import inference_df
from preprocess import NewspieacePreprocess
from trainer import NewspieceModeling
from utils import model_dic


class NewspieaceMain(PathConfig):
    def __init__(self):
        """NewspieaceMain class는 2가지 기능을 한다.
        1. 크롤러를 통해서 수집된 데이터를 기존에 학습시킨 모델을 통해서 inference한다.
        2. 기존에 labeling된 데이터를 통해서 model을 train한다.
        """
        PathConfig.__init__(self)

        # 전체 모듈들을 생성자에서 main을 만들때 내무 instance로 객체를

    def run_modelinference(
        self, PRE_TRAINED_MODEL_NAME, model_name, tracking_ip, current_state
    ):
        """json이나 csv로 수집된 데이터를 전처리하고, 전처리된 데이터를 특정한 모델을 선택하여 inference한다.

        # Parameter
        ---------
        PRE_TRAINED_MODEL_NAME: str
            tokenizer가 사용할 PRE_TRAINED_MODEL_NAME의 이름. 사용할 모델과 PRE_TRAINED_MODEL_NAME의 정보가 맞아야 한다.
            참고: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
        model_name: str
            model runs에 들어갈 model 이름. mlflow server에서 사용자가 지정한 model_runs의 이름
        tracking_ip: str
            mlflow sever가 저장되어 있는 ip주소
        current_state: str
            가져오고 있는 모델의 상태. ex) Production .
        -------------
        # Return

        inferenced_data:
        : inference에 사용할 데이터
        """
        if self.news_path.endswith(".json"):
            input_data = DataIOSteam.get_jsondata(self.news_path)
        else:
            input_data = pd.read_csv(self.news_path)

        preprocessed_data = NewspieacePreprocess.run_preprocessing(input_data)

        inferenced_data = inference_df(
            preprocessed_data,
            PRE_TRAINED_MODEL_NAME,
            model_name,
            tracking_ip,
            current_state,
        )
        # 이 부분 주의. output값이 어디로 가는가에 대한 고려 필요.
        inferenced_data.to_csv(
            self.output_path, +"/new_inference.csv", index=False
        )
        return inferenced_data

    def run_modeltrain(
        self,
        pretrained_model_name,
        batch_size=2,
        epoch=1,
        random_seed=42,
        is_quantization=True,
    ):
        """
        # Description: 기존에 있는 labeled data를 통해서 모델을 학습시킨다.
        -------------
        # Parameter
        -------------
        # Return
        : torch scrip 모델 파일이 지정된 경로로 save된다.
        """
        print(pretrained_model_name)
        model, metric = NewspieceModeling.run_bert(
            pretrained_model_name,
            batch_size,
            epoch,
            random_seed,
            self.model_path,
            self.labeled_path,
            is_quantization,
        )
        return model, metric


if __name__ == "__main__":
    NewspieaceMain = NewspieaceMain()
    data = NewspieaceMain.run_modelinference(
        model_dic["mobilebert"], "mobilebert_tmp"
    )
    print(data)
