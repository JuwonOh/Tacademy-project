from pydantic import BaseModel
from typing import List, Dict

class Data(BaseModel):
    """
    집어넣을 데이터의 형식 클래스
    """

    Sex: List[int]
    Age_band: List[int]
    Pclass: List[int]

class ModelInput():

    def __init__(self, data: Data):
        """
        클래스 객체 생성시 집어넣은 데이터의 값을 저장
        :param data: 인풋할 데이터 클래스
        :return(return type):
        """
        self.sex = data.Sex
        self.age_band = data.Age_band
        self.pclass = data.Pclass

    def input(self):
        """
        집어넣을 데이터의 구조(dict)로 만들어주는 함수
        :param data:
        :return(return type): 인풋 데이터(Dict)
        """
        return {"Sex": self.sex, "Age_band": self.age_band, "Pclass": self.pclass}

class Fib(BaseModel):
    fibNumber: int
# if __name__ == "__main__":
#     # a = Data({"Sex":[1], "Age_band":[1], "Pclass":[1]})
#     a = Data(Sex=[1],Age_band=[1],Pclass=[1])
#     # b = ModelInput(a).input()
#     b= ModelInput(a)
#     b.input
#     print(b.input())