from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from pydantic import BaseModel
from typing import List
import numpy as np

from google.cloud import storage

from schema import *
from utils import *
from predict_module import *
import os

router = APIRouter(prefix='/gcp')

@router.get('/')
def activate():
    """
    :params:
    :return(return type): 현재 연결된 gcp의 bucket(dict) 
    """
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    return {'bucket':buckets}

@router.get("/model")
def model_in_gcp():
    """
    :params:
    :return(return type): gcp에 저장된 모델들 리스트를 가져오기(List)
    """

    model_list = get_model_list_from_gcp()

    return model_list

@router.post("/model/{model}/predict")
def run_with_gcp_model(data: Data, model: str):
    """
    gcp에 모델 피클 파일들을 가져와서 결과 예측하기
    :params:
    :return(return type): 예측결과(List)
    """

    input = np.array(data.Sex + data.Age_band + data.Pclass).reshape(1,-1)

    result = get_pkl_from_gcp(model).predict(input).tolist()

    return result