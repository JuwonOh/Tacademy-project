# API

## Features
- newsmodel을 사용하여 문장을 분석하고, 그 결과를 서빙하는 API입니다.

## Data

- Input_text : 분석하고자 하는 문장이 필요합니다.
- Pretrained_model : 준비된 모델에 입력하기 전 임베딩 단계에서 사용될 모델을 입력합니다. 저희 서비스의 경우 준비된 모델이 따로 있어 반드시 입력해야할 값은 아닙니다.
- Model_name : 저희가 준비한 모델 중 사용할 모델의 이름을 입력하시면 됩니다. 
- IP : 저희가 제공할 서버의 주소를 입력하는 곳으로서, 이 역시 준비된 Default 값이 있기 때문에 필수 입력 데이터는 아닙니다.
  
## Requirements

- Dependencies
  - uvicorn
  - celery
  - fastapi
  - newsmodel

## Usage

### 실행 순서

1. CLI 환경에서 docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.9-management 명령어를 입력합니다.
  
2. app폴더로 이동후 celery -A worker.worker worker -l info -c 10 명령어를 입력합니다. 

3. 브라우저 주소창에 IP주소:포트번호/docs를 실행한다.

## Folder Structure
  ```
  app/
  │
  ├── router
  │       └── celery.py : API router 정보를 담고 있는 파일
  ├── worker
  │       └── worker.py : Celery 실행 정보와 Celery worker에서 실행할 Task를 담고 있는 파일
  ├── main.py : Fast API 실행 파일
  │
  ├── schema.py : API에서 사용될 데이터를 검증하는 클래스를 담은 파일
  │
  └── requirements.txt : API 서버를 이용할때 필요한 모듈들을 나열한 파일
