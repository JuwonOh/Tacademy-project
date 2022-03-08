import pandas as pd


def load_dataset(tag, data_directory):
    """
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
