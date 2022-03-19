import random
import time

from celery import Celery
from worker.predict import predicting
from schema import NLPText
from typing import Dict

app = Celery(
    "my_tasks",
     broker="amqp://guest:guest@localhost:5672//",
     backend="rpc://"
)

@app.task
def nlp_working(text: Dict):
    time.sleep(1)
    # a = NLPpredict()
    # class_prob, pred = a.loaded_model(text)
    # print(f"각 확률은 {class_prob}이고, 따라서 결과는 {pred}이다")
    # print(type(text.input_text))
    # print(type(text.pretrained_model_name))
    # print(type(text.model_name))
    result = predicting( 
            # "President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender.",
            # "google/mobilebert-uncased",
            # "mobilebert_ver1"
            # text.input_text,
            # text.pretrained_model_name,
            # text.model_name
            text['input_text'],
            text['pretrained_model_name'],
            text['model_name']
    
    )
    return result


@app.task
def callback(result):
    return result
