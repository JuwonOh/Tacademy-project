
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os, re
from utils import *
from schema import *
from typing import List, Dict
from predict_module import *
from pydantic import BaseModel
import pickle, numpy as np
import pandas as pd
from google.cloud import storage
# from router import local, gcp
from router import gcp, local, fib
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()


# @app.get('/')
# def activate():
#     """
#     :params:
#     :return(return type): 현재의 로컬 위치와 연동된 gcp의 bucket(dict) 
#     """
#     f = os.getcwd()
#     storage_client = storage.Client()
#     buckets = list(storage_client.list_buckets())
#     return {"path":f, 'bucket':buckets}


# @app.get('/model/info/')
# def info():
#     """
#     :params:
#     :return(return type):현재 저장된 모든 로컬 모델의 meta정보(Json)
#     """
#     a = show_params()
#     return JSONResponse(content=a)


# @app.get('/model/run/')
# def model_run():
#     """
#     학습을 시킨 후 현재 로컬에 저장된 모델 리스트 반환
#     :params:
#     :return(return type): 현재 로컬에 저장된 모델 리스트(List)
#     """
#     os.system('python ./MLflow/main.py --is_keras 0')
#     b = show_model()
#     return b



# @app.get('/model/predict')
# def data_predict(data: Data, model: str):
#     """
#     예측할 데이터를 집어넣고 결과를 보여주는 페이지
#     :params data: 예측할 데이터
#     :params model: 결과 예측시 사용할 모델의 uuid
#     :return(return type): 예측 결과(List)
#     """
#     # print(**data.dict())
#     # print(data)
#     # print(data.dict())
#     # print(pd.DataFrame(data))
#     # models= "330ded0fb7ba462a881357ab456591f5"
#     # results = {"Sex": [1,2,3,1], "Age_band":[1,2,3,4], "Pclass":[1,2,3,1]}
#     # print(type(results))
#     # print(results, type(results))
#     # print(model)
#     input = ModelInput(data).input()
#     result = Predict(model).loaded_model(input).tolist()
#     return result


# @app.get("/model/{model}/predict")
# def each_model_predict(model: str):

#     result = model_load_from_local(model).predict([[1,1,1]]).tolist()

#     return result


# @app.get("/gcp/model")
# def model_in_gcp():
#     """
#     :params:
#     :return(return type): gcp에 저장된 모델들 리스트를 가져오기(List)
#     """

#     model_list = get_model_list_from_gcp()

#     return model_list

# @app.post("/gcp/model/{model}/predict")
# def run_with_gcp_model(data: Data, model: str):
#     """
#     gcp에 모델 피클 파일들을 가져와서 결과 예측하기
#     :params:
#     :return(return type): 예측결과(List)
#     """

#     input = np.array(data.Sex + data.Age_band + data.Pclass).reshape(1,-1)

#     result = get_pkl_from_gcp(model).predict(input).tolist()

#     return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(gcp.router)
app.include_router(local.router)
app.include_router(fib.router)

@app.get('/')
async def root():
    return {'message':'Hello!'}

@app.get("/example")
async def show_example(input_text: str):
    a = NLPpredict()
    class_prob, pred = a.loaded_model(input_text)
    # print(class_prob, pred)
    return class_prob, pred

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8003)



 