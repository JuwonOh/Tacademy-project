import torch.nn as nn
from transformers import AutoModel


class SentimentClassifier(nn.Module):
    """torch의 nn.Module을 사용해서 분류기 클래스의 기본적인 구조가 들어있는 모듈
    ---------
    """
    # caution: 코드가 중복되지 않게 pretrained_model_name만을 사용해서 할 수는 없을까?

    def __init__(self, pretrained_model_name, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name, return_dict=False
        )
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
