from fastapi import APIRouter
<<<<<<< HEAD
from worker.worker import nlp_working
=======
>>>>>>> 369262eeb96a9a253064e32c3d9683be45111acb
from schema import *
from worker.worker import nlp_working

router = APIRouter(prefix="/api_with_celery")


@router.get("/")
def test():
    return "API is running"

<<<<<<< HEAD
@router.post("/predict")
def predict(information: NLPText):

    print(f"Request body: ongoing")
    result = nlp_working.delay(information.dict())
    print("result complete")
    value = result.get()
    print("value complete")

    print(f"Result: {value}")
    return value
=======

@router.post("/predict")
def predict(input_text):

    print(f"Request body: {input_text}")
    result = nlp_working.delay(input_text.input_text)
    value = result.get()

    print(f"Result: {value}")
    return value
>>>>>>> 369262eeb96a9a253064e32c3d9683be45111acb
