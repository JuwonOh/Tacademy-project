import argparse
import sys

from config import PathConfig
from inference.inference import inference
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from model.model import SentimentClassifier
from newspiece import NewspieaceMain
from utils import model_dic

import mlflow

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment_name",
    )
    argument_parser.add_argument(
        "--model_name",
        type=str,
        default=model_dic["mobilebert"],
    )
    argument_parser.add_argument(
        "--input_path",
        type=str,
        default=PathConfig.news_path,
    )

    args = argument_parser.parse_args()
    model_name = args.model_name
    experiment_name = args.experiment_name
    input_path = args.input_path

    tracking_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_server_uri)

    client = MlflowClient()
    model_uri = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name(experiment_name).experiment_id
        ],
        order_by=["metrics.val_acc DESC"],
        filter_string="model_name == {}".format(model_name),
    )

    state_dict = mlflow.pytorch.load_state_dict(model_uri)

    model = SentimentClassifier(model_name, 2)

    model = model.to("cpu")
    class_prob, pred = inference(input_path, model, model_name)

    return (
        class_prob.detach().cpu().numpy()[0],
        pred.detach().cpu().numpy()[0],
    )
