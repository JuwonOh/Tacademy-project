from fastapi import APIRouter
from schema import *
from worker.worker import nlp_working

router = APIRouter(prefix="/api_with_celery")


@router.get("/")
def test():
    return "API is running"


@router.post("/predict")
def predict(input_text):

    print(f"Request body: {input_text}")
    result = nlp_working.delay(input_text.input_text)
    value = result.get()

    print(f"Result: {value}")
    return value
