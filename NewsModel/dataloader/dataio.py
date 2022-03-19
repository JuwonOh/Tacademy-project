import json
import os

import pandas as pd
from utils import notuse_category, use_category


class DataIOSteam:
    def _get_jsondata(self, path):
        """
        # Description: 주어진 path에 위치한 json data를 반환합니다.

        -------------
        # argument
        - path: json data가 위치한 경로
        -------------
        # Return
        : 지정한 path에 위치한 train.csv data
        """
        json_files = [
            pos_json
            for pos_json in os.listdir(path)
            if pos_json.endswith(".json")
        ]

        json_data = pd.DataFrame(
            columns=["date", "title", "content", "source", "url", "category"]
        )

        # we need both the json and an index number so use enumerate()
        for index, js in enumerate(json_files):
            with open(os.path.join(path, js), encoding="UTF8") as json_file:
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
        # 필요 없는 카테고리에 속하는 기사들을 제외한다. 이 카테고리에 대한 부분은 지속적으로 보완해야 한다.
        json_data = json_data[
            json_data["category"].str.contains(use_category)
        ].reset_index(drop=True)
        json_data = json_data[
            ~json_data["category"].str.contains(notuse_category)
        ].reset_index(drop=True)

        json_data["title"] = json_data["title"].str.replace("\n", " ")
        json_data["content"] = json_data["content"].str.replace("\n", " ")

        return json_data

    def _get_xlsxdata(self, path):
        """
        # Description: 주어진 path에 위치한 train.csv data를 반환합니다.
        -------------
        # Parameter
        - path: xlsx data가 위치한 경로
        -------------
        # Return
        : 지정한 path에 위치한 train.csv data
        """

        xlsx_files = [
            pos_json
            for pos_json in os.listdir(path)
            if pos_json.endswith(".xlsx")
        ]

        output_list = []
        for js in xlsx_files:
            with open(os.path.join(path, js), encoding="UTF8") as xlsx_files:
                xlsx_text = pd.read_excel(xlsx_files)
                output_list.append(xlsx_text)

        output_df = pd.concat(output_list, axis=0)

        output_df = output_df[
            output_df["category"].str.contains(use_category)
        ].reset_index(drop=True)
        output_df = output_df[
            ~output_df["category"].str.contains(notuse_category)
        ].reset_index(drop=True)

        output_df["title"] = output_df["title"].str.replace("\n", " ")
        output_df["content"] = output_df["content"].str.replace("\n", " ")

        return output_df
