from fastapi import APIRouter
from schema import *
from worker.worker import nlp_working, prepared_nlp_working

router = APIRouter(prefix="/api_with_celery")


@router.get("/")
def test():
    return "API is running"


@router.post("/predict")
def predict(information: NLPText):

    print(f"Request body: ongoing")
    result = nlp_working.delay(information.dict())
    print("result complete")
    value = result.get()
    print("value complete")

    print(f"Result: {value}")
    return value

@router.post("/prepared_predict")
def prepared_predict(information: NLPText):

    print(f"Request body: ongoing")
    result = prepared_nlp_working.delay(information.dict())
    print("result complete")
    value = result.get()
    print("value complete")

    print(f"Result: {value}")
    return value

