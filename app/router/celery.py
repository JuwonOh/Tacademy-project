from fastapi import APIRouter
from worker.worker import nlp_working
from schema import *

router = APIRouter(prefix='/api_with_celery')

@router.get('/')
def test():
    return 'API is running'

@router.post("/predict")
def predict(information: NLPText):

    print(f"Request body: ongoing")
    result = nlp_working.delay(information.dict())
    print("result complete")
    value = result.get()
    print("value complete")

    print(f"Result: {value}")
    return value