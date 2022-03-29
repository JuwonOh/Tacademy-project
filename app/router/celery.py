from fastapi import APIRouter
from schema import *
from worker.worker import prepared_nlp_working


router = APIRouter(prefix="/api_with_celery")


@router.get("/")
def test():
    """
    API router 접속을 알림
        Parameters
        ----------
        Returns
        -------
        message: str
        "API is running"이란 메세지를 출력한다.
    
    """
    return "API is running"


@router.post("/predict")
def predict(information: NLPText):
    """
    입력된 값을 토대로 예측해주는 API주소
        Parameters
        ----------
        information : NLPText
            NLPText 클래스에 해당하는 데이터를 입력받음
        Returns
        -------
        value : Dict
            class_prob, pred, answer를  반환하며
            class_prob은 리스트의 형태로 각 class의 확률 나타내고
            pred는 예측 결과 클래스를 반환하며
            answer는 결과를 해석한 문장을 반환합니다.
    """
    print("Request body: ongoing")
    result = prepared_nlp_working.delay(information.dict())
    print("result complete")
    value = result.get()
    print("value complete")
    print(f"Result: {value}")
    return value