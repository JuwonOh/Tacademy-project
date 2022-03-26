# nesmodel pakage

## Features
- 국가간 관계를 학습하고 예측하기 위해 만들어진 패키지입니다.
- 모델을 학습하고, 학습된 모델을 불러와서 새로운 데이터를 예측합니다.

## Data

- 미국, 중국, 러시아, 일본, 한국, 인도의 9개 언론사와 8개 국가기관의 기사, 문서
  
## Requirements

- Dependencies
  - Python >= 3.8)
  - mlflow >=1.23.1)
  - torch >=1.4.2+cu111) 
  - transformers>=4.16.2
  - nltk
- pypi link(https://pypi.org/project/newsmodel/)

## Usage

### configuration

- main.py를 돌리는 configuration을 만들까 생각중.
  
### train

- 장기적으로는 mlflow와 연동된 model_run.py와 연결하려고 함.

```
import newsmodel 
from newsmodel.trainer import NewspieceModeling
modeling = NewspieceModeling()
modeling.run_bert(pretrained_model_name,batch_size, epoch,random_seed,model_directory,data_directory,is_quantization)
```

### inference

- app/worker의 predicting을 가져올까 생각중(수현씨와 논의중)

```
import newsmodel
from newsmodel.inference import inference_sentence

inference_sentence(input_text: str, PRE_TRAINED_MODEL_NAME, model_name)

```



## Folder Structure
  ```
  newsmodel/
  │
  ├── main.py - preprocess 완료된 파일 파일을 만들고, 그걸 통해서 inference와 train을 담당하는 모듈(실행 가능)
  │
  ├── model_run.py: 모델 학습과 실험을 mlflow 내부에서 할때 사용하는 모듈
  │
  │
  ├── inference/ - mlflow에 production 상태에 있는 모델을 불러와서 inference하는 모듈
  │       ├──inference.py 
  │
  ├── trainer/ - 기존에 있는 라벨링된 데이터를 기반으로 model을 train하는 모듈
  │       ├── nlpmodel.py 
  │
  ├── model/ - 사용될 전체 모델들이 class로 들어가 있는 파일(개별 모델별 py파일을 만들어야 하나 고민중.
  │       ├──model.py
  │
  ├── preprocess 
  │       ├── preprocess.py - 전처리 모듈들을 사용해서 파일을 전처리하는 모듈
  │       ├── textrank.py - textrank 알고리즘을 사용하기 위한 모듈이 있는 파일
  │       ├── ner.py - ner model을 사용하기 위한 모듈이 있는 파일
  │       ├── countryset.py - 국가쌍을 사용하기 위한 파일
  │
  ├── dataloader
  │       ├── dataio.py - json 파일과 xlsx 파일을 불러오는 모듈
  │       ├── dataloader.py - 국가쌍을 사용하기 위한 파일
  │
  ├── config.json - train의 설정에 필요한 컬럼과 파일 경로를 지정해준다.
  ├── utils.py - 모델 사전이 들어있는 파일
  │
  ├── data/pgdata - parameter가 local에서 사용할 수 있는 경우에 local에서 사용할 데이터가 있는 폴더
  │       ├── newsjson: 크롤링된 json 파일들이 들어가는 폴더 
  │       └── labeled data: 라벨링된 파일이 들어가는 폴더
  ├── saved_model - 학습된 모델이 저장되는 폴더, inference에 사용하는 모듈을 불러오는 폴더
  │
  ├── setup.py - package를 설치하는 모듈
 
  ```