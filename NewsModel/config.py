import os


class PathConfig:
    def __init__(self):
        """
        # Description: 현재 작업경로를 얻고, 그 작업경로에서 data가 있는 경로를 지정해 줍니다.
        - project_path: 현재 작업 경로
        - titanic_path: train 시킬 data가 있는 경로
        -------------
        # Parameter
        -------------
        # Return
        : True or False
        """
        self.project_path = os.getcwd()
        # json_server
        self.news_path = f"{self.project_path}/data/news_json"
        # labeled_server
        self.labeled_path = f"{self.project_path}/data/labeled_data"
        self.model_path = f"{self.project_path}/saved_model"
        self.output_path = f"{self.project_path}/output"


# train 환경을 구성하는 class
class EnvConfig:
    def get_column_list(self):
        """
        # Description: train에 사용할 columns list를 반환합니다.
        - columns_list: train에 사용할 columns을 담은 list
        -------------
        # Parameter
        -------------
        # Return
        : columns_list인 list 자체를 반환
        """
        columns_list = [
            "date",
            "title",
            "content",
            "source",
            "url",
            "category",
        ]  # train에 사용할 columns 지정 => predict 할 때도 3개의 값을 주고 predcition 해야함
        return columns_list
