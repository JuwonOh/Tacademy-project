import argparse
import sys

from mlflow.models.signature import infer_signature
from newspiece import NewspieaceMain

from mlflow import (
    log_artifact,
    log_artifacts,
    log_metric,
    log_metrics,
    log_param,
    log_params,
)

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--model_name",
        type=str,
        help="bert 모델의 이름을 넣으시오",
    )
    argument_parser.add_argument(
        "--batch size",
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

    args = argument_parser.parse_args()
    PRE_TRAINED_MODEL_NAME = args.model_name
    batch_size = args.batch_size
    epoch = args.epoch
    is_quantization = args.is_quantization

    tracking_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_server_uri)

    print(mlflow.get_tracking_uri())
    experiment_name = PRE_TRAINED_MODEL_NAME
    mlflow.set_experiment(experiment_name)

    run_name = "tmp_model"

    with mlflow.start_run(
        run_name=run_name
    ):  # mlflow.start_run(): 새 MLflow run을 시작하여 metrics와 parameters가 기록될 active run으로 설정

        Newspieace = NewspieaceMain()
        model, history = NewspieaceMain.run_modeltrain(
            PRE_TRAINED_MODEL_NAME, batch_size, epoch, is_quantization
        )

        # log metric]
        log_metric(history["val_acc"])
        # log model
        ml_tf.log_model(model, "torch_model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
