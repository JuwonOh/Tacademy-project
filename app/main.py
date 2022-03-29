import uvicorn
from fastapi import FastAPI
import os, re
from utils import *
from schema import *
from typing import List, Dict
from pydantic import BaseModel
import pickle, numpy as np
import pandas as pd
from google.cloud import storage
from router import celery
from fastapi.middleware.cors import CORSMiddleware


description ="""
국제 뉴스기사 데이터와 자연어처리 모델을 활용하여 국가 간 관계 분석을 할 수 있는 API
## nlp_working
국가 간 관계를 **분석**해주는 API 입니다.
"""

app = FastAPI(
        description = description,
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(celery.router)


@app.get('/')
def root():
    """
    API 접속을 알리는 문구
        Parameters
        ----------
        Returns
        -------
        message: str
        Hello!란 메세지를 출력한다.
    """
    return {'message':'Hello!'}



 