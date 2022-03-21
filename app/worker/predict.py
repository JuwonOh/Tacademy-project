import os
import re
import sys

from countryset import morethan_two_countries
from inference import inference_sentence

sys.path.insert(
    0, "/home/sktechxtacademy/Tacademy-project/NewsModel/preprocess"
)
sys.path.insert(
    0, "/home/sktechxtacademy/Tacademy-project/NewsModel/inference"
)

def predicting(input_text: str, PRE_TRAINED_MODEL_NAME, model_name):
    # input_text = "President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender."
    valid, related_nation = morethan_two_countries(input_text)
    if valid:
        class_prob, pred = inference_sentence(
            input_text, PRE_TRAINED_MODEL_NAME, model_name
        )
        relation_dict = {"0": "나쁘", "1": "좋음"}
        relation = relation_dict[str(pred)]
        answer = (
            "이 문장은 {}사이의 관계에 대한 문장입니다. 이 문장에서는 {}의 관계가 {}다고 예측합니다.".format(
                related_nation, related_nation, relation
            )
        )
        print(answer)
        answer = "이 문장은 국가간 관계를 살펴보기에 맞는 문장이 아닙니다. 국가가 2개 언급된 다른 문장을 넣어주세요."
        print(answer)
        class_prob, pred = None, None
    return (class_prob, pred, answer)



if __name__ == "__main__":
    print(
        predicting(
            "President Joe Biden must take expeditious and decisive action immediately against the Russian Federation. The President must order all Russian and civilians to lay down their arms and surrender.",
            "google/mobilebert-uncased",
            "mobilebert_tmp",
        )
    )
