from fastapi import APIRouter
from celery import Celery
from worker.predict import NLPpredict
import worker.celery_predict
from schema import *

router = APIRouter(prefix='/api_with_celery')

@router.get('/')
def test():
    return 'API is running'

@router.get('/predict')
def predict(input_text: NLPText):
    result = celery_predict.nlp_working.delay(input_text)
    value = result.get()
    return value