import argparse
import sys

from newspiece import NewspieaceMain
from utils import model_dic

import mlflow
from mlflow import log_metric, log_params, set_tags

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--model_name",
        type=str,
        default=model_dic["mobilebert"],
        help="bert 모델의 이름을 넣으시오",
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    argument_parser.add_argument(
        "--epoch",
        type=int,
        default=2,
    )
    argument_parser.add_argument(
        "--is_quantization",
        type=int,
        default=True,
    )
    argument_parser.add_argument(
        "--experiment_name",
        type=int,
        default="test_experiment",
    )

    args = argument_parser.parse_args()
    PRE_TRAINED_MODEL_NAME = args.model_name
    batch_size = args.batch_size
    epoch = args.epoch
    is_quantization = args.is_quantization

    tracking_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_server_uri)

    print(mlflow.get_tracking_uri())
    mlflow.set_experiment(args.experiment_name)

    run_name = PRE_TRAINED_MODEL_NAME

    tags = {"model_name": run_name, "release.version": "1.0"}

    Newspieace = NewspieaceMain()
    model, metric = NewspieaceMain.run_modeltrain(
        Newspieace,
        PRE_TRAINED_MODEL_NAME,
        batch_size,
        epoch,
        42,
        is_quantization,
    )
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_params(vars(args))
        mlflow.log_metric("val_acc", metric)
        mlflow.set_tags(tags)

    mlflow.end_run()

    print("run_id: {}".format(run.info.run_id))
    for artifact_path in ["model/data", "scripted_model/data"]:
        artifacts = [
            f.path
            for f in MlflowClient().list_artifacts(
                run.info.run_id, artifact_path
            )
        ]
        print("artifacts: {}".format(artifacts))
