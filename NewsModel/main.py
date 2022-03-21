import pandas as pd
from config import PathConfig
from dataloader import DataIOSteam
from inference import inference_sentence
from preprocess import NewspieacePreprocess
from trainer import NewspieceModeling
from utils import model_dic


class NewspieaceMain(PathConfig):
    def __init__(self):
        """
        # Description: 아래의 class 객체들에서 정의한 여러 함수들을 가져옵니다.
                       통해 전처리한 data를 반환합니다.
        """
        PathConfig.__init__(self)

        # 전체 모듈들을 생성자에서 main을 만들때 내무 instance로 객체를

    def run_modelinference(self, PRE_TRAINED_MODEL_NAME, model_name):
        """
        # Description: json으로 불러온 데이터를 전처리하고, 전처리된 데이터를 특정한 모델을 선택하여 inference한다.
        -------------
        # Parameter
            PRE_TRAINED_MODEL_NAME: str
                torch에서 사용하는 pretrain data set을 다운 받는 위치를 써야 한다.
        -------------
        # Return
        : inference에 사용할 데이터
        """
        json_data = DataIOSteam._get_jsondata(self.news_path)

        preprocessed_data = NewspieacePreprocess.run_preprocessing(json_data)

        for i in range(len(preprocessed_data)):
            (
                preprocessed_data["class_prob"][i],
                preprocessed_data["pred"][i],
            ) = inference_sentence(
                preprocessed_data["input_text"][i],
                PRE_TRAINED_MODEL_NAME,
                model_name,
            )
        # 이 부분 주의. output값이 어디로 가는가에 대한 고려 필요.
        preprocessed_data.to_csv(
            self.output_path, +"/new_inference.csv", index=False
        )
        return preprocessed_data

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
