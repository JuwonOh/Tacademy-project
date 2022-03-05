from importlib.resources import contents
import os
import yaml
import re
from typing import List

# os.system('conda activate cs')

# with open("./MLflow/conda.yaml") as f:
#     content = yaml.load(f)

# print(content['dependencies'][2]['pip'])

# # pip_list = content['dependencies'][2]['pip']

# # for i in pip_list:
# #     os.system('pip install '+i)

# # os.system('pip list')
# # os.system("conda activate cs")
# # os.system('pip list')

# a = os.listdir('./mlruns/0')
# print(a,type(a))

# # for i in list(a):
# #     if i 
# #     print(i)
# model_list=[]

# for i in a:
#     if not re.search('yaml$',i):
#         print(i)
#         model_list.append(i)

# print(model_list) # 모델 폴더 이름 리스트 ['9602e4000eef40d4847ce8f6d3c41eda', 'ba74438be59d4ffdbb89e7326e95a780', 'bd1519e7d3614d76b849ee5512a2a5f7']

# # print(os.listdir("./mlruns/0/"+model_list[0]+"/metrics/"))

# # with open('./mlruns/0/'+model_list[0]+'/metrics/model_score') as f:
# #     print(f.readlines())

# model_params = {}

# for i in model_list:
    
#     # 폴더 params 밑에 있는 값들 집어 넣기
#     path_params = './mlruns/0/'+i+'/params/'
#     content = {}
#     for j in os.listdir(path_params):
#         with open(path_params+j) as f:
#             content[j] = f.readlines()[0]

#     # 폴더 metrics 밑에 있는 값들 집어 넣기
#     path_metrics = './mlruns/0/'+i+'/metrics/'
#     for k in os.listdir(path_metrics):
#         with open(path_metrics+k) as g:
#             content[k] = g.readlines()[0].split()[1]

#     model_params[i] = content

# print(model_params)

# from predict_module import Predict

# model = '08ebe951121040549f5556946e4f23df'
# data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}
# a = Predict(model).loaded_model(data)

# import pandas as pd

# print(pd.DataFrame(data).to_json())

# what = {
#     "0": {
#       "0": "Sex",
#       "1": "Age_band",
#       "2": "Pclass"
#     },
#     "1": {
#       "0": [
#         0
#       ],
#       "1": [
#         0
#       ],
#       "2": [
#         0
#       ]
#     }
#   }
# print(pd.DataFrame(what))

# from pydantic import BaseModel


# class Data(BaseModel):

#     Sex: List[int]
#     Age_band: List[int]
#     Pclass: List[int]

# def what_dict(data: Data):
#     return data

# print(what_dict({
#   "Sex": [
#     1,2,3
#   ],
#   "Age_band": [
#     1,2,3
#   ],
#   "Pclass": [
#     1,2,3
#   ]
# }))

# print(pd.DataFrame(what_dict({
#   "Sex": [
#     1,2,3
#   ],
#   "Age_band": [
#     1,2,3
#   ],
#   "Pclass": [
#     1,2,3
#   ]
# })))

# from predict_module import Predict

# print(Predict('4ad2bc16d37c492ca00cd7f50c38f0cf').loaded_model(what_dict({
#   "Sex": [
#     1,2,3
#   ],
#   "Age_band": [
#     1,2,3
#   ],
#   "Pclass": [
#     1,2,3
#   ]
# })))

from predict_module import Predict
import pickle, numpy as np

# Predict('4ad2bc16d37c492ca00cd7f50c38f0cf').loaded_model({'Sex': [1, 1, 1, 1], 'Age_band': [1, 1, 1, 1], 'Pclass': [1, 1, 1, 1]}).tolist()

# from predict_module import Predict


# data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}
# a = Predict("330ded0fb7ba462a881357ab456591f5").loaded_model(data)
# print(a)
# model = '08ebe951121040549f5556946e4f23df'

# path = f"./mlruns/0/{model}/artifacts/ml_model/model.pkl"

# with open(path,'rb') as f:
#     ai_model = pickle.load(f)

# a = ai_model.predict([[1,1,1]]).tolist()

# print('답은 :',a)

# x = [1]
# y = [2]
# z = [3]
# print(x+y)

import psycopg2

# db = psycopg2.connect(host = '34.64.203.39',
# dbname = 'titanic_db', user='postgres', password='mlflow', port=5432)
# cursor=db.cursor()

# class Databases():
#     def __init__(self):
#         self.db = psycopg2.connect(host = '34.64.203.39', dbname = 'titanic_db', user='postgres', password='mlflow', port=5432)
#         self.cursor = self.db.cursor()

#     def __del__(self):
#         self.db.close()
#         self.cursor.close()

#     def readDB(self,schema,table,colum):
#         sql = " SELECT {colum} from {schema}.{table}".format(colum=colum,schema=schema,table=table)
#         try:
#             self.cursor.execute(sql)
#             result = self.cursor.fetchall()
#         except Exception as e :
#             result = (" read DB err",e)
        
#         return result

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLARCHEMY_DATABASE_URL = "postgresql://postgres:mlflow@34.64.203.39/titanic_db"

engine = create_engine( SQLARCHEMY_DATABASE_URL )

SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)

# Base = declarative_base()

# result = engine.execute("select * from titanic_db")
    
# print(result)

import pandas as pd

pd.read_sql("select * from titanic_db", engine)

# if __name__ == "__main__":
#     a = Databases()
#     b = a.readDB()
