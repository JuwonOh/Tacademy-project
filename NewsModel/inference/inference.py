import argparse
import os
import re

import pandas as pd
import torch
from config import PathConfig
from model.model import SentimentClassifier
from torch import nn
from transformers import AutoModel, AutoTokenizer


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
        self, input_text: str, PRE_TRAINED_MODEL_NAME, model_path
    ):

        model = SentimentClassifier(PRE_TRAINED_MODEL_NAME, 2)
        model.load_state_dict(
            # 모델 위치 변경 필요.
            torch.load(model_path, map_location="cpu"),  # model_server
            strict=False,
        )
        model = model.to("cpu")
        class_prob, pred = inference(input_text, model, PRE_TRAINED_MODEL_NAME)
        return (
            class_prob.detach().cpu().numpy()[0],
            pred.detach().cpu().numpy()[0],
        )
