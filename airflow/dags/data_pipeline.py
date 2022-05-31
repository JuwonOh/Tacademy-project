import datetime
import json
import os
from textwrap import dedent

import pandas as pd

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import (
    LocalFilesystemToGCSOperator,
)
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# setting params

default_args = {
    "owner": "admin",
    "depend_on_past": False,
    "email": "13a71032776@gmail.com",
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=15),
}

dag_args = dict(
    dag_id="Data_pipeline",
    default_args=default_args,
    description="First datapipeline in python",
    schedule_interval=datetime.timedelta(days=1),
    start_date=datetime.datetime(2022, 4, 1),
    tags=["Datapipeline"],
    catchup=True  # backfill을 위해서 catchup을 True로 설정함.
)

# using news category


def filtering_category(json_data):
    # 필요 없는 카테고리에 속하는 기사들을 제외한다. 이 카테고리에 대한 부분은 지속적으로 보완해야 한다.
    use_category = "goverment|biden|trump|iran|whitehouse|corona|kingdom|europe|africa|security|military|china|hongkong|asia|france|russia|politics|national|security|europe|health|middle East|world|asia|opinion|opinion|africa|Military|politics|us|opinions|health|world|asia|economy|europe|uk|middleeast|africa|australia|india|china|briefing|us|opinion|health|world"
    notuse_category = "society|video|garden|biology|photo|sport|music|art|gallery|filrm|fashion|feature|comics|books|theature|culture"

    json_data = json_data[
        json_data["category"].str.contains(use_category)
    ].reset_index(drop=True)
    json_data = json_data[
        ~json_data["category"].str.contains(notuse_category)
    ].reset_index(drop=True)
    return json_data


def get_jsondata(json_path, csv_path, use_filtering=True):
    """
    # Description: 주어진 path에 위치한 json data를 반환합니다.
    차후에 패키지 부분으로 넘길 예정

    -------------
    # argument
    - path: json data가 위치한 경로
    -------------
    # Return
    - json_path:  지정한 path에 위치한 json data
    - csv_path : 저장할 csv 파일
    """

    json_files = [
        pos_json
        for pos_json in os.listdir(json_path)
        if pos_json.endswith(".json")
    ]

    json_data = pd.DataFrame(
        columns=["date", "title", "content", "source", "url", "category"]
    )

    for index, js in enumerate(json_files):
        with open(os.path.join(json_path, js), encoding="UTF8") as json_file:
            json_text = json.load(json_file)

            # here you need to know the layout of your json and each json has to have
            # the same structure (obviously not the structure I have here)
            date = json_text["date"]
            title = json_text["title"]
            content = json_text["content"]
            source = json_text["source"]
            url = json_text["url"]
            category = json_text["category"].lower()

            # here I push a list of data into a pandas DataFrame at row given by 'index'
            json_data.loc[index] = [
                date,
                title,
                content,
                source,
                url,
                category,
            ]

    if use_filtering == True:
        json_data = filtering_category(json_data)

    json_data["title"] = json_data["title"].str.replace("\n", " ")
    json_data["content"] = json_data["content"].str.replace("\n", " ")

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    json_data.to_csv(
        "{}/{}_raw_data.csv".format(
            csv_path, datetime.date.today().strftime("%Y-%m-%d")
        ),
        index=False,
    )


# 날짜 부분은 여기에 만들어준 Jinja template를 사용해서 묶어주자.
yesterday = '{{ execution_date.in_timezone("Asia/Seoul").strftime("%Y-%m-%d") }}'
today = '{{ next_execution_date.in_timezone("Asia/Seoul").strftime("%Y-%m-%d") }}'

# setting path
csv_path = "/home/joh87411/output/data_warehouse"
json_path = "/home/joh87411/output/json_data/{}".format(today)

# setting config params using airflow
dod_command = "python3 /home/joh87411/Pipeline/crawler/DoD_scraper/scraping_latest_news.py --begin_date {} --end_date {} --directory {}".format(
    yesterday, today, json_path
)
whitehouse_command = "python3 /home/joh87411/Pipeline/crawler/whitehouse_scraper/scraping_latest_news.py --begin_date {} --directory {}".format(
    yesterday, json_path
)
dos_command = "python3 /home/joh87411/Pipeline/crawler/dos_scraper/scraping_latest_news.py --begin_date {} --directory {}".format(
    yesterday, json_path
)


with DAG(**dag_args) as dag:
    start = BashOperator(
        task_id="start",
        bash_command='echo "start crawling"',
    )
    with TaskGroup(group_id="crawler") as crawler:
        dod = BashOperator(
            task_id="dod_crawler",
            depends_on_past=False,
            bash_command=dod_command,
        )

        whitehouse = BashOperator(
            task_id="whitehouse_crawler",
            depends_on_past=False,
            bash_command=whitehouse_command,
        )

        dos = BashOperator(
            task_id="dos_crawler",
            depends_on_past=False,
            bash_command=dos_command,
        )

    complete_crawling = DummyOperator(task_id="complete_crawling")

    to_csv = PythonOperator(
        task_id="to_csv",
        depends_on_past=False,
        python_callable=get_jsondata,
        op_args=[json_path, csv_path],
    )

    to_bucket = LocalFilesystemToGCSOperator(
        task_id="to_bucket",
        src="{}/{}_raw_data.csv".format(csv_path, today),
        dst="Data_warehouse/{}_raw_data.csv".format(today),
        bucket="alzal_bucket",
    )

    complete = BashOperator(
        task_id="complete_bash",
        depends_on_past=False,
        bash_command='echo "complete crawling"',
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    start >> crawler >> complete_crawling >> to_csv >> to_bucket >> complete
