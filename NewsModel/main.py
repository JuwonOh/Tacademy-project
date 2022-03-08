# Library Imort
# argparse는 cmd에서 main.py를 실행하여 train할 때, "사용하고자 하는 모델 선택, hyperparameter 선택"을 위해 사용된다.
import argparse
import sys

# 아래와 같이 mlflow는 Python API로 ML/DL Library or Framework를 지원함(sklearn, tensorflow or pytorch etc.)
import mlflow

# experiment, model 등을 기록하는 Library
from mlflow import (  # MLflow에서 artifacts를 기록; MLflow에서 지원하는 metric을 저장하는 library -> -s가 붙으면 딕셔너리(key-value) 형태로 여러 metrics을 저장; MLflow에서 지원하는 parameter를 저장하는 library -> -s가 붙으면 딕셔너리(key-value) 형태로 여러 parameters를 저장
    log_artifact,
    log_artifacts,
    log_metric,
    log_metrics,
    log_param,
    log_params,
)
from mlflow import sklearn as ml_sklearn  # MLflow의 Python API
from mlflow import tensorflow as ml_tf  # MLflow의 ML Framework
from mlflow.models.signature import (  # predict 할 때 사용되는 input data와 output data의 feature name과 dtype을 알 수 있게 해주는 library
    infer_signature,
)

# py 파일로부터 생성된 class를 아래와 같이 import 할 수 있음
# titanic.py 파일에서 정의된 'TitanicMain()' class를 import
from titanic import TitanicMain

# ex) $ python main.py --is_keras 0 --n_estimator 120


def _str2bool(v):
    """
    # Description: 입력되는 매개변수 v를 bool dtype 으로 반환해 줍니다.

        JW: 입력단과 동일.
    -------------
    # Parameter
        - v: bool dtype으로 변경할 string
    -------------
    # Return
        : True or False
    """
    if isinstance(v, bool):  # isinstance 함수: 입력되는 매개변수 v가 bool형인지 확인하는 함수
        return v
    if v.lower() in (
        "yes",
        "true",
        "t",
        "y",
        "1",
    ):  # 입력한 매개변수 v를 소문자로 바꿨을 때, (yes, true, ...)안에 들어 있다면 -> True 반환
        return True
    elif v.lower() in (
        "no",
        "false",
        "f",
        "n",
        "0",
    ):  # 입력한 매개변수 v를 소문자로 바꿨을 때, (no, false ...)안에 들어 있다면 -> False 반환
        return False
    else:  # 우리가 의도하지 않는 방향으로 돌아가는 것을 방지하기 위해서 일부러 에러를 발생시켜야 할 때가 있는데 이 때, raise를 씀
        raise argparse.ArgumentTypeError(
            "Boolean value expected."
        )  # 위의 if, if, elif 조건에서 해당하지 않으면 "Boolean value expected."를 print 해라.


# 모듈을 실행할 수 있는 방법은 1)interpreter에 직접 실행하거나, 2)import 해서 사용하거나 ==> 이해를 돕기 위해 아래와 같은 파일(excuteThisModule.py)이 있다고 하자.
"""
//excuteThisModule.py
def func():
    print("function working")

if __name__ == "__main__":
    print("직접 실행")
    print(__name__)
else:
    print("임포트되어 사용됨")
    print(__name__)
"""
# 1)interpreter에 직접실행        : $ python excuteThisModule.py
# ``          한 결과 : 직접 실행                        (아래까지 결과임)
# __main__                        ==> interpreter에 직접 실행하면, __name__ 변수에 "__main__"이 담겨서 print 됨


# 2)모듈에서 import해서 실행         : import excuteThisModule
# ``            한 결과 : 임포트되어 사용됨               (아래까지 결과임)
# executeThisModule             ==> 모듈에서 import해서 실행하면, __name__ 변수에 "excuteThisModule"이 담겨서 print 됨


# 우리는 interpreter에서 실행할 것이다. 위의 설명에 따라 현재 파일인 main.py를 interpreter에서 실행하면 if 아래에 있는 code들을 실행할 것이다.
if (
    __name__ == "__main__"
):  # 파이썬에서 가장 일반적으로 사용하는 구문으로, 해당 어플리케이션이 모듈로 동작할 수 있도록 해준다.

    # argparse (interpreter에서 모듈 실행시 지정할 Arguments)
    argument_parser = (
        argparse.ArgumentParser()
    )  # argparse.ArgumentParser(): 인자값(argument)을 받을 수 있는 parser(인스턴스) 생성

    argument_parser.add_argument(  # add_argument() method: 원하는 만큼 인자 종류를 계속 추가할 수 있다.
        # main.py(해당파일)을 모듈로 실행하여 train 시킬 때, "--is_keras 0"을 Arguments로 입력함으로써, sklearn의 Random Forest를 사용
        "--is_keras",
        type=str,  # type 지정: 위에서 정의한 _str2bool 함수를 사용하기 위해서 str로 지정
        help="please input 1 or 0",
    )  # help: Argument에 대한 description

    argument_parser.add_argument(  # add_argument() method: 원하는 만큼 인자 종류를 계속 추가할 수 있음
        # main.py(해당파일)을 모듈로 실행하여 train 시킬 때, "--n_estimator 110"을 hyperparameter인 n_estimator 에 원하는 값을 입력할 수 있다.(default=100)
        "--n_estimator",
        # type 지정: sklearn의 Random Forest를 사용하여 train 시킬 때, hyperparameter는 int로 입력해준다.
        type=int,
        default=100,
    )  # default: 기본값을 지정할 수 있다.

    args = (
        argument_parser.parse_args()
    )  # parse_args(): 위에서 입력받은 인자값들을 args에 저장

    # 예외처리
    try:  # try 아래 code를 실행중에 오류가 발생하면 ( is_keras에 0 or 1 이외의 값이 할당되면 )
        is_keras = _str2bool(args.is_keras)
    # except 아래 code를 실행하라             ( print() 안의 문장을 실행시키고 )
    except argparse.ArgumentTypeError as E:
        print("ERROR!! please input is_keras 0 or 1")
        sys.exit()  # exit()과 sys.exit()이 있는데
        # exit()은 cmd(shell)에서 쓰고, sys.exit()은 python(.py) 파일 안에서 사용한다.
        # 역할: 프로그램 종료

    # tracking setting
    tracking_server_uri = "http://127.0.0.1:5000"  # URI setting
    mlflow.set_tracking_uri(
        tracking_server_uri
    )  # mlflow.set_tracking_uri: 내가 tracking할 URI를 setting하는 함수
    print(
        mlflow.get_tracking_uri()
    )  # mlflow.get_tracking_uri(): 현재 tracking하고 있는 URI를 반환하는 함수

    # experiment setting
    experiment_name = "titanic"  # model을 저장하고 싶은 experiment name 설정
    mlflow.set_experiment(
        experiment_name
    )  # mlflow.set_experiment(experiment_name): model을 저장하고 싶은 experiment를 지정

    # log model, metrics, params
    run_name = "2nd_model"
    with mlflow.start_run(
        run_name=run_name
    ):  # mlflow.start_run(): 새 MLflow run을 시작하여 metrics와 parameters가 기록될 active run으로 설정

        titanic = (
            TitanicMain()
        )  # titanic.py에서 정의된 'TitanicMain()' class를 'titanic' 이라는 변수에 객체생성

        # model selection
        """
        titanic.py에서 정의된 'TitanicMain()' class의
        'run'이라는 함수의 매개변수 중 하나인 'is_keras'의 default 값 = 0 이다.
        이는 기본적으로 train시, sklearn을 사용함을 의미한다.
        """
        if is_keras:  # is_keras=1을 입력하면 아래(keras)가 실행됨
            # model(tf_model), metric(eval), train_data(X_train) 값들을 변수에 초기화
            tf_model, eval, X_train = titanic.run(
                is_keras
            )  # run 함수 역할: data split, modeling(is_keras argument에 따라서 --> keras or sklearn)

            # log metric
            log_metric(
                "tf keras score", eval
            )  # log_metric("Metric name", value)             : Metric name을 지정하고, metric value를 기록할 수 있다.

            # schema 저장
            signature = infer_signature(
                X_train, tf_model.predict(X_train)
            )  # infer_signature(train_data, prediction_value): train data와 prediction값을 가지고 schema를 정해줄 수 있다. (mlflow ui에서 확인 가능)

            # log model
            ml_tf.log_model(
                tf_model, "tf2_model", signature=signature
            )  # log_model(model, 'Model name', signature)    : MLflow에서 Python API로 제공하는 tensorflow.keras의 model을 log_model로 기록할 수 있다.

            # 확인
            print(
                "Model saved in run %s" % mlflow.active_run().info.run_uuid
            )  # mlflow.active_run().info.run_uuid            : 현재 활성화된 run_uuid 가져오기

            # Model Registry에 저장
            # ml_tf.save_model( path = 'titanic_deep', tf_model = tf_model)        ## ml_tf.save_model( path = 'titanic_deep',
            # tf_model = tf_model)       : path에 지정한 경로에 'titanic_deep'이라는 이름의 폴더를 자동으로 생성하여 그 폴더에 artifacts에 들어가 있는 모든 파일들을 저장한다.

        else:  # is_keras=0(default)을 입력하면 아래(sklearn)가 실행됨
            # model(model), metric & params(model_info), train_data(X_train) 값들을 변수에 초기화
            model, model_info, X_train = titanic.run(
                is_keras, args.n_estimator
            )  # titanic.py에서 정의된 'TitanicMain()' class인 'titanic' 객체의 run 함수 역할: modeling

            # log metric
            # ex) log_metric("rf_score", score_info['rf_model_score'])             ## log_metric(metric_name(key), value): Metric name을 지정하고, metric value를 기록할 수 있다.
            ##     log_metric("lgbm_score", score_info['lgbm_model_score'])

            # log metrics
            """
            log_metrics
            : param metrics: Dict[str, float]
            : param step: int | None = None
            """
            log_metrics(
                model_info["score"]
            )  # model.py 안에, TitanicModeling 이라는 class의 함수인 run_sklearn_modeling 에
            # model_info라는 딕셔너리가 있고, 그 딕셔너리 안에 'score'라는 key가 있는데, 이 key의 value로 딕셔너리가 있다.
            # 그래서 이 딕셔너리(key:value)를 가져온다.

            # log parameters
            """
            log_params
            : param params: Dict[str, Any]
            """
            log_params(
                model_info["params"]
            )  # model.py 안에, TitanicModeling 이라는 class의 함수인 run_sklearn_modeling 에
            # model_info라는 딕셔너리가 있고, 그 딕셔너리 안에 'params'라는 key가 있는데, 이 key의 value에 딕셔너리가 있다.
            # 그래서 이 딕셔너리(key:value)를 가져온다.

            # schema 저장
            signature = infer_signature(
                X_train, model.predict(X_train)
            )  # infer_signature(train_data, prediction_value): train data와 prediction값을 가지고 schema를 정해줄 수 있다. (mlflow ui에서 확인 가능)

            # log model
            ml_sklearn.log_model(
                model, "ml_model", signature=signature
            )  # log_model(model, 'Model name'): MLflow에서 Python API로 제공하는 sklearn의 model을 mlflow의 log_model을 사용해 기록

            # 확인
            print(
                "Model saved in run %s" % mlflow.active_run().info.run_uuid
            )  # mlflow.active_run().info.run_uuid            : 현재 활성화된 run_uuid 가져오기

            # Model Registry에 저장
            # ml_sklearn.save_model( path='titanic_rf', sk_model=model)            ## ml_sklearn.save_model( path = 'titanic_rf',
            # sk_model = model)       : path에 지정한 경로에 'titanic_rf'라는 이름의 폴더를 자동으로 생성하여 그 폴더에 artifacts에 들어가 있는 모든 파일들을 저장한다.
