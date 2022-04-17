import argparse
import json
import os
import warnings
from ast import Yield
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
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


class NewsTrain:
    """기존에 laebling된 데이터를 통해서 모델을 학습하고, 지정된 mlflow server에 저장하는 클래스

    Parameters
    ---------
    server_uri: str
        모델이 저장되어 있는 Mlflow server_uri
    experiment_name: str
        사용자가 지정하고자 하는 expermiment의 이름
    pretrained_model_name: str
        사용할 Bert model의 이름.
    random_seed:int
        사용할 랜덤시드
    model_directory: str
        학습시킨 모델이 저장되는 위치
    data_directory: str
        학습 데이터가 있는 위치
    quantization: boolean
        model 양자화 여부를 선택
    version: str
        지정하고자 하는 version의 정보
    """

    def __init__(
        self,
        server_uri,
        experiment_name,
        device="cuda",
        pretrained_model_name="google/mobilebert-uncased",
        random_seed=42,
        model_directory="./saved_model",
        data_directory="./input_data/labeled_data",
        quantization=True,
        version="1.0",
    ):
        self.pretrained_model_name = pretrained_model_name
        self.random_seed = random_seed
        self.model_directory = model_directory
        self.data_directory = data_directory
        self.quantization = quantization
        self.server_uri = server_uri
        self.experiment_name = experiment_name
        self.version = version
        self.device = device

    def train_model(self, batch_size, epoch):
        """기존에 라벨링된 데이터를 불러와서 모델을 학습하는 함수입니다.

        Parameters
        -------------
        batch_size: int
            모델에서 사용할 배치 사이즈
        epoch: int
            모델이 학습할 횟수

        Return
        -------------
        model : pytorch.nn
            전체 epoch에서 가장 정확도가 높은 torch model을 반환합니다.
        quantized_model : pytorch.nn
            train한 모델을 양자화 모델
        best_accuracy: numpy
            train epoch에서 가장 높은 정확도. mlflow의 model metric으로 사용된다.
        """
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

        # random seed 지정
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        tokenizer = AutoTokenizer.from_pretrained(
            # , return_dict=False  ## 이 부분이 모델에 따라 달라짐.
            self.pretrained_model_name
        )
        # batch size
        # 원 자료에서 label이 있는 column
        tag = "sentiment"

        # data split
        train_df, valid_df = load_dataset(tag, self.data_directory)
        df_train, df_test = train_test_split(
            train_df, test_size=0.2, random_state=self.random_seed
        )

        # load dataset
        train_dataset = SentimentDataset(
            df_train.sentence.values,
            df_train.sentiment.values,
            tokenizer,
            max_len=512,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=0
        )

        test_dataset = SentimentDataset(
            df_test.sentence.values,
            df_test.sentiment.values,
            tokenizer,
            max_len=512,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=0
        )
        # torch dataloader 지정.
        # data = next(iter(train_dataloader))

        # 지정한 classifier에 label의 수와 모델이름을 넣는다.
        model = SentimentClassifier(self.pretrained_model_name, 2)
        model = model.to(self.device)

        # gpu cpu와 메모리를 비움. 실 사용에서 주의

        EPOCHS = epoch  # 바꿔야할 파라미터
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        # mlflow.pytorch.autolog() autolog를 넣어야 하나
        history = defaultdict(list)

        # 실제 학습 코드
        best_accuracy = 0
        for epoch in range(EPOCHS):
            print("start {}th train".format(epoch))

            train_acc, train_loss = self._train_epoch(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                scheduler,
                len(df_train),
            )

            val_acc, val_loss = self._eval_model(
                model, test_dataloader, loss_fn, len(df_test)
            )

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] Train loss: {train_loss} acc: {train_acc} | Val loss: {val_loss} acc: {val_acc}"
            )

            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            if val_acc > best_accuracy:
                if self.model_directory is None:
                    pass
                else:
                    torch.save(
                        model.state_dict(),
                        self.model_directory
                        + "/{}.pt".format(
                            self.pretrained_model_name.split("/")[-1]
                        ),
                    )
                    best_accuracy = val_acc

                    if self.quantization is True:
                        quantized_model = torch.quantization.quantize_dynamic(
                            model.to("cpu"),
                            {torch.nn.Linear},
                            dtype=torch.qint8,
                        )
                        torch.save(
                            quantized_model.state_dict(),
                            self.model_directory
                            + "/quantized_{}.pt".format(
                                self.pretrained_model_name.split("/")[-1]
                            ),
                        )
                    else:
                        quantized_model = None
        return model, quantized_model, best_accuracy.item()

    def mlflow_save(self, run_name, model, best_accuracy):
        """train_model에서 학습한 모델을 mlflow server에 저장한다.

        Parameters
        -------------
        run_name: str
            사용자가 지정하고자 하는 run의 이름. run은 expermiment의 하위개념이며 expermiment내의 개별 model run을 구분하기 위해 사용한다.
        model : pytorch.nn
            전체 epoch에서 가장 정확도가 높은 torch model을 반환합니다.
        best_accuracy: numpy
            train epoch에서 가장 높은 정확도. mlflow의 model metric으로 사용된다.

        """

        mlflow.set_tracking_uri(self.server_uri)
        print("save uri is {}".format(mlflow.get_tracking_uri()))
        mlflow.set_experiment(self.experiment_name)

        tags = {"model_name": run_name, "release.version": self.version}

        with mlflow.start_run(nested=True) as run:

            print("save uri is {}".format(mlflow.get_artifact_uri()))
            mlflow.pytorch.log_model(model, run_name)
            mlflow.log_params(NewsTrain.__dict__)
            mlflow.log_metric("val_acc", best_accuracy)
            mlflow.set_tags(tags)

        mlflow.end_run()

    def _train_epoch(
        self, model, data_loader, loss_fn, optimizer, scheduler, n_examples
    ):
        """
        Description: train을 해주는 모듈
        ---------
        Arguments

        model: nn.module
            정의한 모델
        data_loader
            torch data_loader
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
        print(self.device)
        model = model.train().to(self.device)
        losses = []
        correct_predictions = 0
        for d in data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["labels"].to(self.device)
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

    def _eval_model(self, model, data_loader, loss_fn, n_examples):
        """
        Description: train된 모델을 evaluation을 해주는 모듈
        ---------
        Arguments
        ---------
        model: nn.module
            정의한 모델
        data_loader: 
            torch data_loader
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
        model = model.eval().to(self.device)
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["labels"].to(self.device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())
        return correct_predictions.double() / n_examples, np.mean(losses)
