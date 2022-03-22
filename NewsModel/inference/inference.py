import torch
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer

import mlflow


def embedding(input_text, PRE_TRAINED_MODEL_NAME):
    """
    Description: input text가 들어오면 모델에 inference할 text를 사용할 수 있게, input text를 embedding해준다.
    ---------
    Arguments:
    input_text: str
        사용자가 넣을 문장 정보.
    PRE_TRAINED_MODEL_NAME: model name
        tokenizer가 사용할 PRE_TRAINED_MODEL_NAME의 이름.
    ---------
    Return:
        input_ids
        attention_mask
    ---------
    """
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        PRE_TRAINED_MODEL_NAME, return_dict=False
    )

    encoded_review = tokenizer.encode_plus(
        input_text,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded_review["input_ids"].to(device)
    attention_mask = encoded_review["attention_mask"].to(device)

    return input_ids, attention_mask


def load_model(model_name):
    """
    Description: mlflow에 올라가 있는 모델과 embedding된 결과를 사용해서
    ---------
    Arguments:
        model_name: model에 들어갈 model 이름.
    ---------
    Return:
        model: mlflow에서 production 단계로 올라가 있는 model
    ---------
    """
    tracking_server_uri = "http://34.64.184.112:5000/"
    mlflow.set_tracking_uri(tracking_server_uri)
    client = MlflowClient()
    filter_string = "name = '{}'".format(model_name)
    result = client.search_model_versions(filter_string)

    for res in result:
        if res.current_stage == "Production":
            deploy_version = res.version
    model_uri = client.get_model_version_download_uri(
        model_name, deploy_version
    )
    model = mlflow.pytorch.load_model(model_uri)

    return model


def inference(model, input_ids, attention_mask):
    """
    Description: mlflow에 올라가 있는 모델과 embedding된 결과를 사용해서 모델을 inference한다.
    ---------
    Arguments:
    input_text: str
        사용자가 넣을 문장 정보.
    PRE_TRAINED_MODEL_NAME: model name
        tokenizer가 사용할 PRE_TRAINED_MODEL_NAME의 이름.
    ---------
    Return:
        input_ids
        attention_mask
    ---------
    """
    logits = model(input_ids, attention_mask)
    softmax_prob = torch.nn.functional.softmax(logits, dim=1)
    _, prediction = torch.max(softmax_prob, dim=1)

    return softmax_prob, prediction


def inference_sentence(input_text: str, PRE_TRAINED_MODEL_NAME, model_name):
    input_ids, attention_mask = embedding(input_text, PRE_TRAINED_MODEL_NAME)
    model = load_model(model_name)
    class_prob, pred = inference(model, input_ids, attention_mask)
    return (
        class_prob.detach().cpu().numpy()[0],
        pred.detach().cpu().numpy()[0],
    )


def inference_df(preprocessed_data, PRE_TRAINED_MODEL_NAME, model_name):
    preprocessed_data["class_prob"] = ""
    preprocessed_data["pred"] = ""
    for i in range(len(preprocessed_data)):
        (
            preprocessed_data["class_prob"][i],
            preprocessed_data["pred"][i],
        ) = inference_sentence(
            preprocessed_data["input_text"][i],
            PRE_TRAINED_MODEL_NAME,
            model_name,
        )
    return preprocessed_data
