import argparse
import os
import re

import pandas as pd
import torch
from config import PathConfig
from mlflow.tracking import MlflowClient
from model.model import SentimentClassifier
from torch import nn
from transformers import AutoModel, AutoTokenizer

import mlflow


def inference(input_text, model, PRE_TRAINED_MODEL_NAME):
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

    tokenizer = AutoTokenizer.from_pretrained(
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


class inference_class(PathConfig):
    def __init__(self):
        PathConfig.__init__(self)

    # parameter로 어떤 모델을 사용할지 지정할 수 있게 해야한다.

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
        class_prob, pred = inference(input_text, model, PRE_TRAINED_MODEL_NAME)
        return (
            class_prob.detach().cpu().numpy()[0],
            pred.detach().cpu().numpy()[0],
        )
