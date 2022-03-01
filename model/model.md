# models

## Requirements
* Python >= 3.7
* PyTorch = 1.82
* tqdm 
* cudatoolkit = 11.1
* transformers  = 4.16.2

## Features

- 국가간 관계 데이터의 전처리와 모델링에 필요한 파일을 모은 폴더입니다.
- 시도한 apex, quantization, pruning에 대한 코드는 1-BERT Fine-tuning Classification.ipynb에 있습니다. 지속적으로 시도중이지만, 용량, 연산 속도에서의 이점보다 정확도가 떨어져 보류중입니다.

## Folder Structure
  ```
model/
  │
  ├── train.py - main 모델로 상정한 mobile bert를 사용하여 모델을 학습시키는 py파일입니다.
  ├── predict.py - model 파일에 들어가 있는 torch script 파일을 불러와서 새로운 데이터를 inference하는 pt파일입니다.(수정중)
  │
  ├── spare_model - 모델의 score가 너무 낮아 후순위로 넘겨진 모델들이 있는 폴더입니다.
  │
  ├── data/ - 학습에 필요한 데이터가 있는 폴더입니다.
  │  
  ├── saved/
  │   ├── models/ - trained models are saved here
  │ 
  ├── data_preprocessing/ - local에 있는 크롤러에서 데이터가 들어올라는 전제하에서 전처리 모듈이 있는 폴더입니다.(ipynb에서 py파일로 변환중)
  │  



## 모델 결과 비교

| 모델 이름 | 성능(accuracy, F1 score) | 용량 | quantization 용량 | quantization 정확도(dynamic, static) |
| --- | --- | --- | --- | --- |
| BERT | 61, 59 | 420 | 100 | 55 |
| Electra | 64, 63 | 120 | 42 | ? |
| MobileBert | 70, 67 | 92 | 31 | 60 |
| Robert | 57,59(3epoch, 너무 느림) | 진행중 | 진행중 | ? |

## cution
- pytorch-template(https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py)에서 큰 영향을 받았습니다. 
- 아직 개별 요소를 분해하지 않고, 함수만 비슷한 형태로 만들어두고, train.py에 올려두었습니다.
-
