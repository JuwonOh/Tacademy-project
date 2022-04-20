from typing import Dict
import numpy as np
import pandas as pd
from schema import NLPText
from celery import Celery, Task
from newsmodel.inference.inference import NewsInference
from newsmodel.preprocess.countryset import morethan_two_countries

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
            self.inferencer = NewsInference(server_uri="your_mlflow_server_uri", model_name="mobile_bert")
            self.model = self.inferencer._load_model("mobile_bert","Production")
            return self.run(*args, **kwargs)

@app.task(
    ignore_result=False,
    bind=True,
    base=SimplePredict,
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
        class_prob, pred = self.inferencer.inference(text['input_text'])
        relation_dict = {"0": "나쁘", "1": "좋"}
        relation = relation_dict[str(pred)]
        answer = (
            "이 문장은 {}사이의 관계에 대한 문장입니다. 이 문장에서는 {}의 관계가 {}다고 예측합니다.".format(
                related_nation, related_nation, relation
            )
        )
        result = {'class_prob':class_prob.tolist(), 'pred':int(pred), 'answer':answer}
    else:
        answer = "이 문장은 국가간 관계를 살펴보기에 맞는 문장이 아닙니다. 국가가 2개 언급된 다른 문장을 넣어주세요."
        class_prob, pred = None, None
        result = {'answer':answer}

    return result
