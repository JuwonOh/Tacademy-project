import mlflow
import pandas as pd
import os, re
import numpy as np

import torch
from torch import nn
from transformers import MobileBertModel, MobileBertTokenizer

class Predict:
    
    def __init__(self, model: str):
        """
        클래스 객체 생성시 모델의 주소 반환
        :params model: 실행시킬 모델의 uuid
        :return: 
        """
        self.logged_model = 'runs:/'+ model +'/ml_model'

    def loaded_model(self, data):
        """
        모델을 통하여 값 예측
        :params data: 예측할 데이터
        :return(return type): 예측값(리스트)
        """
        return mlflow.pyfunc.load_model(
            self.logged_model
        ).predict(pd.DataFrame(data))


class SentimentClassifier(nn.Module):
    """
    모델 준비코드 (토치 스크립트 파일을 받아들이기 전에 해야하는 전초작업)
    """
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = MobileBertModel.from_pretrained(
            "google/mobilebert-uncased", return_dict=False
        )
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


class NLPpredict:
    """
    인풋데이터를 분류해주는 코드
    """
    def inference(self, input_text, model, PRE_TRAINED_MODEL_NAME):
        """
        Description: 특정 문장이 들어오면 input text를 embedding하고, inference 해주는 모듈
            api serving에서 사용.
        ---------
        Arguments
        ---------
        input_text: str
            사용자가 넣을 문장 정보.
        model: model
            사용자가 지정한 모델 정보.
        ---------
        Return: 0과 1 사이의 결과 값
        ---------
        """
        device = "cpu"

        tokenizer = MobileBertTokenizer.from_pretrained(
            PRE_TRAINED_MODEL_NAME, return_dict=False
        )

        encoded_review = tokenizer.encode_plus(
            input_text,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_review["input_ids"].to(device)
        attention_mask = encoded_review["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        softmax_prob = torch.nn.functional.softmax(logits, dim=1)
        _, prediction = torch.max(softmax_prob, dim=1)

        return softmax_prob, prediction

    # def __init__(self, model: str):
    #    self.logged_model = "runs:/" + model + "/ml_model"

    def loaded_model(self, input_text: str):
        """
        준비한 토치 스크립트와 SentimentClassfier 클래스를 통하여 준비된 레이어를 가지고 분류하는 함수
        : params input_text: 예측하려고 하는 인풋 데이터
        : return(return type): prob, pred(List, int)
        """
        PRE_TRAINED_MODEL_NAME = "google/mobilebert-uncased"
        model = SentimentClassifier(2)
        model.load_state_dict(
            torch.load("mobilebert.pt", map_location="cpu"),
            strict=False,
        )
        model = model.to("cpu")
        class_prob, pred = self.inference(input_text, model, PRE_TRAINED_MODEL_NAME)
        return class_prob.detach().cpu().numpy().tolist()[0], pred.detach().cpu().numpy().tolist()[0] 

if __name__ == "__main__":
    # a = Predict("330ded0fb7ba462a881357ab456591f5")
    # data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}
    # print(a.loaded_model(data))

    def predicting(input_text: str):
        a = NLPpredict()
        # input_text = "President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender."
        class_prob, pred = a.loaded_model(input_text)
        return (class_prob, pred)
    
    from line_profiler import LineProfiler

    line_profiler = LineProfiler()
    # line_profiler.add_function(NLPpredict().inference)
    # line_profiler.add_function(NLPpredict().loaded_model)
    print(predicting("President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender."))
    a = ()

    lp_wrapper = line_profiler(predicting("President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender."))
    lp_wrapper

    line_profiler.print_stats()
    