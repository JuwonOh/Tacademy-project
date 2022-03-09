from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from pydantic import BaseModel
from typing import List

from google.cloud import storage

from schema import *
from utils import *
from predict_module import *
import os

router = APIRouter(prefix='/local')

@router.get('/')
def activate():
    """
    :params:
    :return(return type): 현재의 로컬 위치
    """
    f = os.getcwd()
    return {"path":f}


@router.get('/model/info')
def info():
    """
    :params:
    :return(return type):현재 저장된 모든 로컬 모델의 meta정보(Json)
    """
    a = show_params()
    return JSONResponse(content=a)


@router.get('/model/run')
def model_run():
    """
    학습을 시킨 후 현재 로컬에 저장된 모델 리스트 반환
    :params:
    :return(return type): 현재 로컬에 저장된 모델 리스트(List)
    """
    os.system('python ./MLflow/main.py --is_keras 0')
    b = show_model()
    return b



@router.get('/model/predict')
def data_predict(data: Data, model: str):
    """
    예측할 데이터를 집어넣고 결과를 보여주는 페이지
    :params data: 예측할 데이터
    :params model: 결과 예측시 사용할 모델의 uuid
    :return(return type): 예측 결과(List)
    """
    # print(**data.dict())
    # print(data)
    # print(data.dict())
    # print(pd.DataFrame(data))
    # models= "330ded0fb7ba462a881357ab456591f5"
    # results = {"Sex": [1,2,3,1], "Age_band":[1,2,3,4], "Pclass":[1,2,3,1]}
    # print(type(results))
    # print(results, type(results))
    # print(model)
    input = ModelInput(data).input()
    result = Predict(model).loaded_model(input).tolist()
    return result


@router.get("/model/{model}/predict")
def each_model_predict(model: str):

    result = model_load_from_local(model).predict([[1,1,1]]).tolist()

    return result