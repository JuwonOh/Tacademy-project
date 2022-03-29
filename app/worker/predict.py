import os
import re
import sys
import newsmodel
from newsmodel.inference import inference_sentence
from newsmodel.preprocess import NewspieacePreprocess, morethan_two_countries


def predicting(input_text: str, PRE_TRAINED_MODEL_NAME, model_name, tracking_ip):

    """
    newsmodel 모듈을 이용하여 입력값을 분석해주는 함수
        Parameters
        ----------
        input_text : str
            분석할 텍스트
        PRE_TRAINED_MODEL_NAME : str
            입력값 예측에 앞서 입력값 임베딩에 필요한 모데의 이름
        model_name : strs
            학습시킨 모델의 이름
        tracking_ip : str
            예측할 서버의 주소
        
        Returns
        -------
        result : Dict
            class_prob, pred, answer를 dictionary 형태로 반환함
        
    """

    valid, related_nation = morethan_two_countries(input_text)

    if valid:
        class_prob, pred = inference_sentence(
            input_text, PRE_TRAINED_MODEL_NAME, model_name, tracking_ip
        )
        relation_dict = {"0": "나쁘", "1": "좋"}
        relation = relation_dict[str(pred)]
        answer = (
            "이 문장은 {}사이의 관계에 대한 문장입니다. 이 문장에서는 {}의 관계가 {}다고 예측합니다.".format(
                related_nation, related_nation, relation
            )
        print(answer)
        result = {'class_prob':class_prob.tolist(), 'pred':int(pred), 'answer':answer}

    else:
        answer = "이 문장은 국가간 관계를 살펴보기에 맞는 문장이 아닙니다. 국가가 2개 언급된 다른 문장을 넣어주세요."
        print(answer)
        class_prob, pred = None, None
        result = {'class_prob': 'nothing', 'pred':'nothing', 'answer':answer}
    return result