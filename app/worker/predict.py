import os
import re

import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from torch import nn
from transformers import MobileBertModel, MobileBertTokenizer

import mlflow


class Predict:
    def __init__(self, model: str):
        """
        클래스 객체 생성시 모델의 주소 반환
        :params model: 실행시킬 모델의 uuid
        :return:
        """
        self.logged_model = "runs:/" + model + "/ml_model"

    def loaded_model(self, data):
        """
        모델을 통하여 값 예측
        :params data: 예측할 데이터
        :return(return type): 예측값(리스트)
        """
        return mlflow.pyfunc.load_model(self.logged_model).predict(
            pd.DataFrame(data)
        )


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

    def inference_sentence(
        self, input_text: str, PRE_TRAINED_MODEL_NAME, experiment_name
    ):
        tracking_server_uri = "http://34.64.184.112:5000/"
        mlflow.set_tracking_uri(tracking_server_uri)
        client = MlflowClient()
        mlflow.set_experiment(experiment_name)

        model_uri = client.search_runs(
            experiment_ids=[
                client.get_experiment_by_name(
                    PRE_TRAINED_MODEL_NAME
                ).experiment_id
            ],
            order_by=["metrics.val_acc DESC"],
        )
        # 이 부분에서 수정 필요. -> gcp instance에 있는 모델을 어떻게 불러올 수 있는가에 대한 경로를 알아야 한다.
        model = mlflow.pytorch.load_state_dict(
            "runs:{}".format(model_uri[0].info.artifact_uri)
        )

        model = model.to("cpu")
        class_prob, pred = self.inference(
            input_text, model, PRE_TRAINED_MODEL_NAME
        )
        return (
            class_prob.detach().cpu().numpy().tolist()[0],
            pred.detach().cpu().numpy().tolist()[0],
        )


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
    print(
        predicting(
            "President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender."
        )
    )
    a = ()

    lp_wrapper = line_profiler(
        predicting(
            "President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender."
        )
    )
    lp_wrapper

    line_profiler.print_stats()
