from pydantic import BaseModel
from typing import List, Dict


class NLPText(BaseModel):
    """
    입력값 검증 클래스
    Parameters
    ----------
    Attributes
    ----------
    input_text : str
        분석하려는 문장
        
    pretrained_model_name : str
        Default value = 'google/mobilebert-uncased'
        모델 입력 전 임베딩 단계에서 필요한 모델의 이름

    """
    input_text: str
    pretrained_model_name: str = "google/mobilebert-uncased"