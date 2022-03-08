import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import transformers
from config import PathConfig
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    MobileBertModel,
    MobileBertTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


class NewspieceModeling(PathConfig):
    def __init__(self):
        PathConfig.__init__(self)

    def run_mobilebert(
        self, batch_size, epoch, random_seed, model_directory, data_directory 
    ):## path 미수정.
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

        # model 지정. MOBILE BERT이외의 변수를 선택 할 수 있도록 차후 변경 예정.
        PRE_TRAINED_MODEL_NAME = "google/mobilebert-uncased"
        tokenizer = MobileBertTokenizer.from_pretrained(
            PRE_TRAINED_MODEL_NAME  # , return_dict=False  ## 이 부분이 모델에 따라 달라짐.
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

        bert_model = MobileBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        # 지정한 classifier에 label의 수와 모델이름을 넣는다.
        model = SentimentClassifier(2)
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
                torch.save(model.state_dict(), self.model_path)  # model path
                best_accuracy = val_acc


def load_dataset(tag, data_directory):
    """## path 미수정.
    Description: 본 데이터 셋에서는 응답자가 의미없다는 답변을 할 수 있다.
    의미 없다는 답변을 받은 데이터는 'irrelevant', 의미있는 답변이 담긴 데이터 프레임을 받은 데이터 프레임은 'sentiment'로 선택한다.
    ---------
    Arguments

    tag : str
        'irrelevant', 'sentiment'로 원하는 데이터를 뽑아낸다.
    ---------
    Return: pandas.Dataframe
    ---------

    """
    if tag == "irrelevant":
        train_df = pd.read_csv(
            "{}/irrelevant_train.tsv".format(data_directory), sep="\t"
        )
        valid_df = pd.read_csv(
            "{}/irrelevant_test.tsv".format(data_directory), sep="\t"
        )
    elif tag == "sentiment":
        train_df = pd.read_csv(
            "{}/sentiment_train.tsv".format(data_directory), sep="\t"
        )
        valid_df = pd.read_csv(
            "{}/sentiment_test.tsv".format(data_directory), sep="\t"
        )
    else:
        train_df = None
        valid_df = None
        print("data_directory doesn`t exist")

    return train_df, valid_df


class SentimentDataset(Dataset):
    """
    Description: 불러온 데이터프레임에서 필요한 정보를 뽑아내고, 임베딩한다.
    ---------
    Arguments

    sentences: str
        문장에 대한 정보를 저장
    labels : int
        들어온 문장의 id를 가지고 있는다.
    tokenizer: str
        어떤 tokenizer를 사용할지 지정한다.
    max_len: int
        문장의 최대 길이를 지정하여 지나치게 긴 문장을 잘라낸다. max_len이 길어지면, padding을 해야하기에 연산이 느려진다.
    ---------
    Return: dict. dict안의 value와 item은 다음과 같다.

    sentence: str
        input에 들어갈 문장정보
    input_ids: tensor
        tokenizer가 임베딩한 문장의 정보
    attention_mask: tensor
        최대 길이로 지정한 512에서 단어가 차지하는 길이에 대한 정보
    labels: int
        분류 모델이 분류할 타겟 변수
    """

    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "sentence": sentence,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SentimentClassifier(nn.Module):
    """
    Description: torch의 nn.Module을 사용해서 분류기 클래스를 만든다.
    ---------
    """

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = MobileBertModel.from_pretrained(
            "google/mobilebert-uncased", return_dict=False
        )
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


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
