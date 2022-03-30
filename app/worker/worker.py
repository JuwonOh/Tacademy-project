from typing import Dict
import numpy as np
import pandas as pd
import torch
from schema import NLPText
from celery import Celery, Task
import newsmodel
from newsmodel.inference import embedding, load_model, inference, inference_sentence
from newsmodel.preprocess import NewspieacePreprocess, morethan_two_countries
from worker.predict import predicting

app = Celery(
    "my_tasks",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://"
)


class SimplePredict(Task):
    def __init__(self):
        super().__init__()
        self.model = None
    def __call__(self, *args, **kwargs):
        if not self.model:
            self.model = load_model(self.model_name, self.ip_params)
            return self.run(*args, **kwargs)


@app.task
def nlp_working(text: Dict):
    """
    값(문장, 모델) 입력 시 분석해주는 Task.
        Parameters
        ----------
        text : Dict
            API router로부터 NLPText 클래스에 해당하는 입력값을 dictionary 형태로 받아온다.
        Returns
        -------
        result : Dict
            값은 class_prob, pred, result 값을 가지고 있는 dictionary 형태로 반환한다.
    """
    result = predicting(
        text['input_text'],
        text['pretrained_model_name'],
        text['model_name'],
        text['ip_param']
    )
    return result

    
@app.task(
    ignore_result=False,
    bind=True,
    base=SimplePredict,
    model_name = "mobilebert",
    ip_params = "http://34.64.73.79"
)
def prepared_nlp_working(self, text: Dict):
    
    """
    값(문장, 모델) 입력 시 미리 메모리에 올린 모델을 이용하여 입력값을 분석해주는 Task.
        Parameters
        ----------
        text : Dict
            API router로부터 NLPText 클래스에 해당하는 입력값을 dictionary 형태로 받아온다.
        Returns
        -------
        result : Dict
            값은 class_prob, pred, result 값을 가지고 있는 dictionary 형태로 반환한다.
    """
    valid, related_nation = morethan_two_countries(text['input_text'])
    if valid:
        input_ids, attention_mask = embedding(text['input_text'], text['pretrained_model_name'])
        class_prob, pred = inference.inference(self.model, input_ids, attention_mask)
        class_prob = class_prob.detach().cpu().numpy()[0]
        pred = pred.detach().cpu().numpy()[0]
        relation_dict = {"0": "나쁘", "1": "좋"}
        relation = relation_dict[str(pred)]
        answer = (
            "이 문장은 {}사이의 관계에 대한 문장입니다. 이 문장에서는 {}의 관계가 {}다고 예측합니다.".format(
                related_nation, related_nation, relation
            )
        )
        result = {'class_prob':class_prob.tolist(), 'pred':int(pred), 'answer':answer}
    else:
        answer = "이 문장은 국가간 관계를 살펴보기에 맞는 문장이 아닙니다. 국가가 2개 언급된 다른 문장을 넣어주세요
."
        print(answer)
        class_prob, pred = None, None
        result = {'class_prob':'nothing', 'pred':'nothing', 'answer':answer}

    return result
