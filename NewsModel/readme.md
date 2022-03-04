# newspiece mlflow 

- 현재 상황은 기존 model과 mlflow의 틀을 통합한 상황입니다.
- 현 상황에서 주요한 파일은 newspiece.py입니다. 
- newspiece.py의 NewspieaceMain 클래스의 run_jsoninference는 inference를 담당하며, run_modeltrain은 모델 학습을 당당합니다.

## Requirements
* Python >= 3.7
* PyTorch >= torch==1.8.2+cu111
* tqdm
* transformers 4.x

## Folder Structure
  ```
  Mlflow/
  │
  ├── main.py - 미완성: newspiece에 있는 함수를 사용해서 mlflow를 사용할 수 있게 해주는 모듈
  │
  ├── newspiece.py - preprocess 완료된 파일 파일을 만들고, 그걸 통해서 inference와 train을 담당하는 모듈
  ├── nlpmodel.py - 기존에 있는 라벨링된 데이터를 기반으로 model을 train하는 파일(실행 가능)
  ├── inference.py - 이전에 학습된 모델을 사용해서 json 파일을 predict해주는 모듈
  │
  ├── preprocess.py - dataio모듈을 통해서 만든 파일을 전처리 해주는 모듈
  ├── dataio.py - json 파일과 xlsx 파일을 불러오는 모듈
  ├── config.json - train의 설정에 필요한 컬럼과 파일 경로를 지정해준다.
  ├── utils.py - 전처리에 필요한 국가쌍 등의 전보가 들어가 있는 파일
  ├── textrank.py - textrank 알고리즘을 사용하기 위한 모듈이 있는 파일
  │
  ├── data/pgdata - parameter가 local에서 사용할 수 있는 경우에 local에서 사용할 데이터가 있는 폴더
  │       ├── newsjson: 크롤링된 json 파일들이 들어가는 폴더 
  │       └── labeled data: 라벨링된 파일이 들어가는 폴더
  ├── model - 학습된 모델이 저장되는 폴더, inference에 사용하는 모듈을 불러오는 폴더
 
  ```