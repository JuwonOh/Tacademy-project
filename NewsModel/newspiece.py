import pandas as pd
from config import PathConfig
from dataloader.dataio import DataIOSteam
from inference.inference import inference_class
from preprocess.preprocess import NewspieacePreprocess
from trainer.nlp_model import NewspieceModeling
from utils import model_dic


class NewspieaceMain(
    NewspieacePreprocess, NewspieceModeling, DataIOSteam, inference_class
):
    def __init__(self):
        """
        # Description: 아래의 class 객체들에서 정의한 여러 함수들을 가져옵니다.
                       통해 전처리한 data를 반환합니다.
        """
        NewspieacePreprocess.__init__(self)
        PathConfig.__init__(self)
        NewspieceModeling.__init__(self)
        DataIOSteam.__init__(self)
        inference_class.__init__(self)

    def run_modelinference(
        self, PRE_TRAINED_MODEL_NAME="google/mobilebert-uncased"
    ):
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
        json_data = self._get_jsondata(self.news_path)

        preprocessed_data = self.run_preprocessing(json_data)
        preprocessed_data["class_prob"] = ""
        preprocessed_data["pred"] = ""

        for i in range(len(preprocessed_data)):
            (
                preprocessed_data["class_prob"][i],
                preprocessed_data["pred"][i],
            ) = self.inference_sentence(
                preprocessed_data["input_text"][i], PRE_TRAINED_MODEL_NAME
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
        self.run_bert(
            pretrained_model_name,
            batch_size,
            epoch,
            random_seed,
            self.model_path,
            self.labeled_path,
            is_quantization,
        )

        # mlflow를 여기에 넣어놓으면 응집도(쓸데업싱 의존성이 생김)가 생김.


# 시험용
if __name__ == "__main__":
    NewspieaceMain = NewspieaceMain()
    data = NewspieaceMain.run_modeltrain(
        model_dic["mobilebert"],
        batch_size=2,
        epoch=1,
        random_seed=42,
        is_quantization=True,
    )
    print(data)  # 이 부분에서 mlflow 모델을 적용해서 mlflow로 정보를 갈 수 있게 해야
    ## newspiece, mlflow를
    # mlflow class를 별도로 사용할 수 있게 해야함. 결과를 어떻게 가져오던 상관 없이 결과를 가져오게 하는것.
    # 완전 독립적인 객체가 되게- 일반화가 되어 일반적으로 범용적으로 사용할 수 있게 할 수 있어야 한다.
    # 이상적인 객체지향적 프로그래밍, 모듈화의 구현.

if __name__ == "__main__":
    NewspieaceMain = NewspieaceMain()
    data = NewspieaceMain.run_modelinference(model_dic["mobilebert"])
    print(data)
