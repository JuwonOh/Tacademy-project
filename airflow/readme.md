# Airflow
- Airflow 폴더는 Tacademy project의 일부입니다.
   국가간 관계를 분석하기 위한 자료들을 airflow를 사용해서 자동으로 수집합니다.

# Data

- 15개의 Newssource의 자료를 크롤링 할 수 있는 크롤러가 있습니다.
- 크롤러는 "Newssource_scraper" 형태로 저장되어 있습니다.
 <img src="../img/newsource.png" width="800" height="400"> 


# Folder structure

- 각 크롤러 폴더의 기본적인 구조는 다음과 같습니다. 
```
 airflow/
  │
  ├── crawler - airflow의 data pipeline이 사용하는 크롤러들이 있는 곳
  │
  ├── dags/
  │       ├── data_pipeline - 데이터 파이프라인
  │       ├── model_pipeline - 모델 파이프라인
  │
  ├── logs
  │
```

