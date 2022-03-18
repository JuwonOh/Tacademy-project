import random
import time

from celery import app
from worker.predict import NLPpredict


@app.task
def nlp_working(text):
    time.sleep(1)
    a = NLPpredict()
    class_prob, pred = a.loaded_model(text)
    print(f"각 확률은 {class_prob}이고, 따라서 결과는 {pred}이다")
    return class_prob, pred


@app.task
def callback(result):
    return result
