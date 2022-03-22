import random
import time

from celery import Celery, Task
from worker.predict import embedding, predicting
from schema import NLPText
from typing import Dict

import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from worker.preprocessing import morethan_two_countries
from transformers import AutoTokenizer

app = Celery(
    "my_tasks",
     broker="amqp://guest:guest@localhost:5672//",
     backend="rpc://"
)

class SimplePredict(Task):

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self): #, *args):
        if  not self.model:
            model_name = "mobilebert_ver1"
            tracking_server_uri = "http://34.64.184.112:5000/"
            mlflow.set_tracking_uri(tracking_server_uri)
            client = MlflowClient()
            filter_string = "name = '{}'".format( #self.model_name[0])
                    model_name)
            result = client.search_model_versions(filter_string)

            for res in result:
                if res.current_stage == "Production":
                    deploy_version = res.version
            model_uri = client.get_model_version_download_uri(
                           # self.model_name[0], deploy_version
                           model_name, deploy_version
                        )
            self.model = mlflow.pytorch.load_model(model_uri)

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

@app.task(
            ignore_result = False,
            bind = True,
            base = SimplePredict,
            # model_name = ("mobilebert_ver1")
        )
def prepared_nlp_working(text: Dict):
    input_ids, attention_mask = embedding(text['input_text'], text['pretrained_model_name'])
    logits = self.model(input_ids, attention_mask)
    
    # softmax_prob = torch.nn.functional.softmax(logits, dim=1)
    
    # class_prob = torch.nn.functional.softmax(logits, dim=1)
    # class_prob = class_prob.detach().cpu().numpy()[0]
    
    class_prob = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    # _, prediction = torch.max(softmax_prob, dim=1)
    
    _, pred = torch.max(softmax_prob, dim=1)
    pred = pred.detach().cpu().numpy()[0] 

    valid, related_nation = morethan_two_countries(text['input_text'])
    if valid:
        
        relation_dict = {"0": "나쁘", "1": "좋음"}
        relation = relation_dict[str(pred)]
        answer = (
            "이 문장은 {}사이의 관계에 대한 문장입니다. 이 문장에서는 {}의 관계가 {}다고 예측합니다.".format(
                related_nation, related_nation, relation
            )
        )
        print(answer)

    else:
        answer = (
            "이 문장은 국가간 관계를 살펴보기에 맞는 문장이 아닙니다. 국가가 2개 언급된 다른 문장을 넣어주세요."
        )
        print(answer)
        class_prob, pred = None, None
    return (class_prob, pred, answer)



@app.task
def callback(result):
    return result
