from fastapi import APIRouter
from celery import Celery
from predict_module import NLPpredict
import cel_predict

router = APIRouter(prefix='/api_with_celery')

app = Celery('my_tasks',
              broker='amqp://guest:guest@localhost:5672//',
              backend="rpc://")

@router.get('/')
def test():
    return 'API is running'

@router.get('/predict')
def predict():
    result = cel_predict.nlp_working.delay("President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender.")
    value = result.get()
    return value