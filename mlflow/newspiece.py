import pandas as pd
from config import PathConfig  # config.py라는 파일에서 "PathConfig" 라는 class 가져옴

# config 파일에는 Path, Env 에 대한 class가 들어있음
from dataio import DataIOSteam  # dataio.py 라는 파일에서 "DataIOSteam" 이라는 class 가져옴
from inference import Predict

# model.py 라는 파일에서 "TitanicModeling" 이라는 class 가져옴
from nlp_model import NewspieceModeling
from preprocess import (  # preprocess.py라는 파일에서 "TitanicPreprocess" 라는 class 가져옴
    NewspieacePreprocess,
)


class NewspieaceMain(
    NewspieacePreprocess, NewspieceModeling, DataIOSteam, Predict
):
    def __init__(self):
        """
        # Description: 아래의 class 객체들에서 정의한 여러 함수들을 가져옵니다.
                       통해 전처리한 data를 반환합니다.
        -------------
        # Parameter
        -------------
        # Return
        """
        NewspieacePreprocess.__init__(self)  # pass
        PathConfig.__init__(self)  # 현재 작업 디렉토리 / 데이터 위치 경로 받아옴
        NewspieceModeling.__init__(self)  # pass
        DataIOSteam.__init__(self)
        Predict.__init__(self)
        # "DataIOSteam" 라는 class에 "__init__" 자체가 없을 때: class를 선언하는 것과 같음 => 선언한 class의 함수들 사용

    def run_jsoninference(
        self,
    ):  # run함수는 is_keras=0, n_estimator=100 이 default
        """
        # Description:
        -------------
        # Parameter
        -------------
        # Return
        :
        """
        json_data = self._get_jsondata(
            self.news_path
        )  # self가 모두 같기 때문에, 위에서 불러온 class의 함수들을 "self.~"을 통해 사용할 수 있다.
        # config.py에서 PathConfig라는 class를 호출하고, 그 안에 self.titanic_path를 호출한다.

        preprocessed_data = self.run_preprocessing(json_data)
        print(preprocessed_data)
        for i in range(len(preprocessed_data)):
            (
                preprocessed_data["class_prob"][i],
                preprocessed_data["pred"][i],
            ) = self.inference_sentence(preprocessed_data["input_text"][i])
        # preprocess.py 에서 TitanicPreprocess라는 class의 함수인 run_preprocessing() 를 사용하여 전처리
        preprocessed_data.to_excel("temp.xlsx", index=False)
        return preprocessed_data

    def run_modeltrain(self):
        NewspieaceMain = NewspieaceMain()
        NewspieaceMain.run_mobilebert(
            batch_size=2,
            epoch=1,
            random_seed=42,
            model_directory="./model",
            data_directory="./data",
        )


# 시험용
if __name__ == "__main__":
    monilebert = NewspieaceMain()
    data = monilebert.run_jsoninference()
    print(data)
