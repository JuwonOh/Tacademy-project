import argparse
import json
import os
import warnings
from ast import Yield
from collections import defaultdict

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import transformers
from config import PathConfig
from dataloader.dataloader import SentimentDataset, load_dataset
from model.model import SentimentClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


class NewspieceModeling(PathConfig, SentimentClassifier):
    def __init__(self):
        PathConfig.__init__(self)

    def run_bert(
        self,
        pretrained_model_name,
        batch_size,
        epoch,
        random_seed,
        model_directory,
        data_directory,
        is_quantization,
    ):
        """
        # Description: sklearn API를 사용하여 모델을 학습하고, 예측에 사용할 모델과 기록할 지표들을 반환합니다.
        -------------
        # Parameter
        - X: train data (feature)
        - y: train data (label)
        - n_estimator: The number of trees in the forest (hyper parameter)
        -------------
        # Return
        : model object, model information(metric, parameter)
        """
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # random seed 지정
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name  # , return_dict=False  ## 이 부분이 모델에 따라 달라짐.
        )
        # batch size
        BATCH_SIZE = batch_size
        # 원 자료에서 label이 있는 column
        tag = "sentiment"

        # data split
        train_df, valid_df = load_dataset(tag, data_directory)
        print(train_df)
        print(valid_df)
        df_train, df_test = train_test_split(
            train_df, test_size=0.2, random_state=random_seed
        )

        # load dataset
        train_dataset = SentimentDataset(
            df_train.sentence.values,
            df_train.sentiment.values,
            tokenizer,
            max_len=512,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, num_workers=0
        )

        test_dataset = SentimentDataset(
            df_test.sentence.values,
            df_test.sentiment.values,
            tokenizer,
            max_len=512,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, num_workers=0
        )
        # torch dataloader 지정.
        data = next(iter(train_dataloader))

        bert_model = AutoModel.from_pretrained(pretrained_model_name)

        # 지정한 classifier에 label의 수와 모델이름을 넣는다.
        model = SentimentClassifier(pretrained_model_name, 2)
        model = model.to(device)

        # gpu cpu와 메모리를 비움. 실 사용에서 주의
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        EPOCHS = epoch  # 바꿔야할 파라미터
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(device)
        # mlflow.pytorch.autolog() autolog를 넣어야 하나
        history = defaultdict(list)

        # 실제 학습 코드
        best_accuracy = 0
        for epoch in range(EPOCHS):
            print("start {}th train".format(epoch))

            train_acc, train_loss = train_epoch(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train),
            )

            val_acc, val_loss = eval_model(
                model, test_dataloader, loss_fn, device, len(df_test)
            )

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] Train loss: {train_loss} acc: {train_acc} | Val loss: {val_loss} acc: {val_acc}"
            )

            print()
            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(
                    model.state_dict(),
                    self.model_path
                    + "/{}.pt".format(pretrained_model_name.split("/")[-1]),
                )  # model path
                best_accuracy = val_acc
                yield best_accuracy, epoch  # return metric change

                if is_quantization:
                    quantized_model = torch.quantization.quantize_dynamic(
                        model.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8
                    )
                    torch.save(
                        quantized_model.state_dict(),
                        self.model_path
                        + "/quantized_{}.pt".format(
                            pretrained_model_name.split("/")[-1]
                        ),
                    )
                else:
                    pass
        return model, history


def train_epoch(
    model, data_loader, loss_fn, optimizer, device, scheduler, n_examples
):
    """
    Description: train을 해주는 모듈
    ---------
    Arguments

    model: nn.module
        정의한 모델
    loss_fn :
        손실함수
    tokenizer:
        앞에서 지정한 tokenizer
    max_len: int
        지정한 문장의 최대길이
    optimizer:
        사용하고자 하는 optimizer
    device: device(type='cuda')
        gpu사용 혹은 cpu사용
    scheduler: scheduler
        사용할 scheduler
    n_examples : int
        전체 분류기에 사용할 자료의 수
    ---------
    Return: train_accuracy, train_loss
    ---------
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    Description: train된 모델을 evaluation을 해주는 모듈
    ---------
    Arguments
    ---------
    model: nn.module
        정의한 모델
    loss_fn : CrossEntropyLoss()
        손실함수
    device: device(type='cuda')
        gpu사용 혹은 cpu사용
    n_examples : int
        전체 분류기에 사용할 자료의 수
    ---------
    Return: eval_accuracy, eval_loss
    ---------
    """
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)
