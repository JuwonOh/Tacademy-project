import time
from celery_practice import app
# from router.celery import app
from predict_module import NLPpredict

@app.task
def nlp_working(text: str):

    time.sleep(1)

    a = NLPpredict()
    class_prob, pred = a.loaded_model(text)
    
    print(f'각 확률은 {class_prob}이고, 따라서 결과는 {pred}이다')

    return class_prob, pred

@app.task
def callback(result):
    return result