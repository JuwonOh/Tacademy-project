# Newsmodel pakage
- Newsmodel pakage는 Tacademy project의 일부입니다.
   국가간 관계를 학습하고 예측하기 위해 만들어진 패키지입니다.
- 모델을 학습하고, 학습된 모델을 불러와서 새로운 데이터를 예측합니다.

## Features

### Train feature
- labeling되어 있는 데이터를 기반으로 모델을 학습합니다.
- 현재는 3가지 pre-trained model을 지원합니다. 
  - 지원하고 있는 모델은 다음과 같습니다. "bert", "mobilebert", "Electra"
  - 모델 경량화를 위해 quantization 기능을 지원합니다.
- 현 데이터에서 모델의 성능은 다음과 같습니다. 
  - 다음의 표는 동일한 random state에서 epoch 100를 한 결과입니다.

| 모델 이름  | 성능(accuracy, F1 score) | 용량 | quantization 용량 | static quantization 정확도 |
| ---------- | ------------------------ | ---- | ----------------- | -------------------------- |
| BERT       | 61, 59                   | 420  | 100               | 55                         |
| Electra    | 64, 63                   | 120  | 42                | 60                         |
| MobileBert | 70, 67                   | 95   | 31                | 61                         |

- 현재 사용하고 있는 주요한 model은 MobileBert이며, defalt 모델로 사용하고 있습니다.

### Inference feature
- 15개의 news source에서 수집한 데이터와 기존에 학습한 모델을 사용합니다.
- Inference에서는 다음과 같은 문제를 고려했고, 문제를 해결하기 위해 전처리 방법을 사용했습니다. 
  1. 문장이 전체 기사에서 중요한 정보를 담고 있는지 -> text rank를 사용한 중요 문장 추출
  2. 문장이 특정 국가들 사이의 관계에 대한 정보인가 -> 국가를 나타내는 키워드를 기반으로 국가쌍을 추출했습니다.
  3. 지정한 키워드가 정말로 유의미한 키워드인가 -> NER를 사용해서 고유 명사에 해당하는 키워드가 존재하는 국가쌍만을 분석했습니다.
- 사용하고자 하는 model을 지정하기 위해서는 저장된 mlflow server에 있는 model_name과 current stage를 지정해줘야 합니다.
- 주로 사용하고 있는 모델은 MobileBert입니다.

## Data
- 미국, 중국, 러시아, 일본, 한국, 인도의 9개 언론사와 8개 국가기관의 기사, 문서

## Folder Structure
  ```
  newsmodel/
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
  ├── utils.py - 모델 사전이 들어있는 파일
  │
  ├── setup.py - package를 설치하는 모듈
  ├── data/pgdata - parameter가 local에서 사용할 수 있는 경우에 local에서 사용할 데이터가 있는 폴더
  │       ├── newsjson: 크롤링된 json 파일들이 들어가는 폴더 
  │       └── labeled data: 라벨링된 파일이 들어가는 폴더
  ├── saved_model - 학습된 모델이 저장되는 폴더, inference에 사용하는 모듈을 불러오는 폴더
  
 
  ```
  
## Requirements

- Dependencies
  - Python >= 3.8)
  - mlflow >=1.23.1)
  - torch >=1.4.2+cu111) 
  - transformers>=4.16.2
  - nltk
  - progressbar
- 패키지는 다음의 pypi link(https://pypi.org/project/newsmodel/) 에서도 볼 수 있습니다.

## Usage

### Train

- Newsmodel.trainer

```
## model import
import newsmodel
from newsmodel.trainer import NewsTrain

## instance setting
Trainer = NewsTrain(server_uri="your_mlflow_server_uri", experiment_name= "mobile_bert", device="cuda")

## model fitting
model, quantized_model, best_accuracy = Trainer.train_model(batch_size, epoch)

## save mlflow
Trainer.mlflow_save(run_name, model, best_accuracy)
```

### Inference

```
## model import
from newsmodel.inference import NewsInference

## instance setting
Inferencer = NewsInference(server_uri= "your_mlflow_server_uri", model_name = "mobile_bert")

## inference_sentence : 문장 단위로 문장을 분석하고 싶을 때 사용하세요.
inferenced_label = Inferencer.inference_sentence(input_text)

## inference_df : 데이터 프레임 단위로 문장을 분석하고 싶을 때 사용하세요.
inferenced_df = Inferencer.inference_df(pandas_df)
```

### Preprocessing

- preprocessing은 textrank, ner, 국가쌍 유무 확인을 지원합니다.
  - run_preprocessing: 데이터 프레임 단위에서 textrank, ner, 국가쌍 유무 확인을 제공하는 method입니다.
  - morethan_two_countries: 문장내에서 국가쌍의 유무를 알려주는 함수입니다.
```
## model import
import newsmodel
from newsmodel.preprocess import NewspieacePreprocess

## instance setting
Preprocesor = NewspieacePreprocess(body_column_name = "content")

## run_preprocessing: 전체 데이터 프레임을 전처리할때 사용하세요. 
preprocessed_df = Preprocesor.run_preprocessing

## morethan_two_countries: 국가쌍을 뽑을때 사용하세요. 
from newsmodel.preprocess import morethan_two_countries

morethan_two_countries(input_text)
```