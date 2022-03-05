import os, re
from typing import List
import pickle5 as pickle
from google.cloud import storage



def show_model(): 
    """
    mlruns 밑에 있는 결과물 폴더명 가져와서 model_list 안에 집어넣습니다.
    :param model:
    :return(return type): local에 저장된 모델의 목록(리스트)
    """
    model_list = []

    for i in os.listdir('./mlruns/0'):
        if not re.search('yaml$',i):
            # print(i)
            model_list.append(i)
    
    return model_list




def show_params():
    """
    결과물 폴더 안에 각 모델의 model_score와 params 폴더 밑에 있는 parameter 값들을
    가져와서 params 딕셔너리 안에 {모델명: {모델 파라미터1: 값1, ...}, ...} 형태로 집어넣습니다.
    :param model:
    :return(return type): 각 모델들의 파라미터 값(json)
    """
    
    model_params = {}
    model_list = show_model()

    for i in range(len(model_list)):

        # 폴더 params 밑에 있는 값들 집어 넣기
        path_params = './mlruns/0/'+model_list[i]+'/params/'
        content = {}
        for j in os.listdir(path_params):
            with open(path_params+j) as f:
                content[j] = f.readlines()[0]

        # 폴더 metrics 밑에 있는 값들 집어 넣기
        path_metrics = './mlruns/0/'+ model_list[i] +'/metrics/'
        for k in os.listdir(path_metrics):
            with open(path_metrics+k) as g:
                content[k] = g.readlines()[0].split()[1]
        
        content['model_name'] = model_list[i]
        model_params[i+1] = content
    
    return model_params



def model_load_from_local(model: str):
    """
    로컬 저장소에서 모델을 불러옵니다
    :param model: 가져올 모델의 uuid
    :return(return type): 학습된 모델(model)
    """

    path = f"./mlruns/0/{model}/artifacts/ml_model/model.pkl"
    with open(path,'rb') as f:
        ai_model = pickle.load(f)

    return ai_model



def get_model_list_from_gcp():
    """
    GCP에서 모델 목록을 가져옵니다.
    :param model:
    :return(return type): gcp 내의 학습된 모델 리스트(리스트)
    """
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/TFG5076XG/mlfapi_practice/app/compute-engine-342100-5747b11244f0.json"

    bucket_name = 'alzal_bucket'
    client = storage.Client()
    blobs = client.list_blobs(bucket_name)

    model_list=[]

    # for i in blobs:
    #     # print(i.name)
    #     try:
    #         if (len(i.name.split('/')[2])>=20)&(i.name.split('/')[2] not in model_list):
    #             model_list.append(i.name.split('/')[2])
    #     except: pass

    for i in blobs:
        try:
            if (len(i.name.split('/')) == 8) & (i.name.split('/')[4] not in model_list):
                print(i.name.split('/')[4])
                model_list.append(i.name.split('/')[4])
        except: pass

    return model_list




def get_pkl_from_gcp(model: str):
    """
    GCP에서 모델 피클 파일을 가져옵니다
    :param model: 가져올 모델의 uuid
    :return(return type): 학습된 모델(model)
    """

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/TFG5076XG/mlfapi_practice/app/compute-engine-342100-5747b11244f0.json"

    bucket_name = 'alzal_bucket'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    try:
        source_path = f'titanic/1/{model}/artifacts/ml_model/model.pkl'
        blobs = bucket.blob(source_path)
    except:
        source_path = f'titanic/2/{model}/artifacts/ml_model/model.pkl'
        blobs = bucket.blob(source_path)

    model_bytes = blobs.download_as_bytes(raw_download=True)

    ai_model = pickle.loads(model_bytes, encoding="bytes")

    return ai_model


# print(get_pkl_from_gcp('241b7d1d60854ff1b49149cbbf82e2fe').predict([[1,1,1]]).tolist())

if __name__ == "__main":
    # print(get_pkl_from_gcp('241b7d1d60854ff1b49149cbbf82e2fe').predict([[1,1,1]]).tolist())