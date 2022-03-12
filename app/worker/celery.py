import time
import random

from celery import Celery
from API.predict_module import NLPpredict
# import cel_predict

app = Celery('my_tasks',
              broker='amqp://guest:guest@localhost:5672//',
              backend="rpc://")

# @app.task
# def working(id=1):

#     # 1~5초 사이의 랜덤한 Delay를 발생.
#     time.sleep(random.randint(1,5))

#     return '{}번째, 일을 끝냈다.'.format(id)

# @app.task
# def nlp_working(text: str):

#     time.sleep(1)

#     a = NLPpredict()
#     class_prob, pred = a.loaded_model(text)

#     return f'각 확률은 {class_prob}이고, 따라서 결과는 {pred}이다'
