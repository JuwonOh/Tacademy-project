import re
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from preprocess.countryset import morethan_two_countries
from preprocess.ner import sentence_to_nerlist
from preprocess.textrank import sort_sentence_importance
from progressbar import ProgressBar
from tqdm import tqdm

# 전처리를 구성하는 class


class NewspieacePreprocess:
    def __init__(self):
        pass

    def run_preprocessing(self, data):  # data = news.json or news.csv
        """
        # Description: 주어진 data의 특정 컬럼의 이름을 전처리, 결측치 imputation, feature를 이용한 새로운 변수정의, labeling, 필요없는 컬럼삭제 등을
                       통해 전처리한 data를 반환합니다.
        -------------
        # Parameter
        - data: train에 사용할 raw data
        -------------
        # Return
        : 전처리 후의 데이터
        """
        print(data["content"])
        ner_df = quasiNER_extractor3(data, "content")
        ner_df["sententce_ner"] = ""
        for a in range(len(ner_df)):
            try:
                ner_df["sententce_ner"][a] = [
                    each
                    for each in model.predict(ner_df["input_text"][a])
                    if each["tag"] != "O"
                ]
            except:
                ner_df["sententce_ner"][a] = ""
        return ner_df


def corpus_to_nerlist(input_corpus):
    return list(
        set(chain.from_iterable(map(sentence_to_nerlist, input_corpus)))
    )


def quasiNER_extractor3(dataframe, nameof_articlebody_column):
    """
    Description: 기존에 앞에서 작성한 함수를 연결하여 실행하는 코드이다.

    ---------
    Arguments
    ---------
    dataframe: pandas dataframe
        국가간 관계가 들어가 있는 데이터 프레임을 넣는다.
    nameof_articlebody_column: str
        데이터 프레임의 칼럼 이름을 넣는다.
    ---------
    Return: pandas.dataframe
    ---------
    """
    dataframe["lowercase_" + nameof_articlebody_column] = dataframe[
        nameof_articlebody_column
    ]
    quasinerdf_output_list = []
    dataframe["doc_id"] = ""
    for doc_id in range(
        len(dataframe["lowercase_" + nameof_articlebody_column])
    ):
        dataframe["doc_id"][doc_id] = doc_id
        input_text = dataframe["lowercase_" + nameof_articlebody_column][
            doc_id
        ]
        try:
            sentence = " ".join(
                sort_sentence_importance(input_text, standard="mean", topn=3)
            )
        except:
            sentence = ""

        vectorized_morethan_two_countries = np.vectorize(
            morethan_two_countries, otypes=[list]
        )
        output_list = vectorized_morethan_two_countries(sentence)
        output_df = pd.DataFrame.from_records(
            [output_list], columns=["isvalid", "list_of_countries"]
        )
        output_df["input_text"] = sentence

        output_df["doc_id"] = doc_id
        quasinerdf_output_list.append(output_df)

    quasinerdf_output_df = pd.concat(quasinerdf_output_list)
    quasinerdf_output_df = quasinerdf_output_df[
        quasinerdf_output_df["isvalid"] == True
    ].reset_index(drop=True)
    del quasinerdf_output_df["isvalid"]
    all_df = pd.merge(dataframe, quasinerdf_output_df)
    return all_df
