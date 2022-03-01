import argparse
import os
import re

import pandas as pd
import torch
from torch import nn
from transformers import MobileBertModel, MobileBertTokenizer


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = MobileBertModel.from_pretrained(
            "google/mobilebert-uncased"
        )
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_text",
        type=str,
        default="President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender.",
        help="inference 하고자 하는 txt를 입력하십쇼.",
    )

    args = parser.parse_args()
    input_text = args.input_text

    PRE_TRAINED_MODEL_NAME = "google/mobilebert-uncased"
    model = SentimentClassifier(2)
    model.load_state_dict(
        torch.load("saved/mobilebert.pt", map_location="cpu"),
        strict=False,
    )
    model = model.to("cpu")

    class_prob, pred = inference(input_text, model, PRE_TRAINED_MODEL_NAME)
    print(class_prob.detach().cpu().numpy()[0])
    print(pred.detach().cpu().numpy()[0])
