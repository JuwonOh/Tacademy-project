import argparse
import csv
import glob
import json
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta
from itertools import chain
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import clear_output
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from progressbar import ProgressBar
from textrank import nltk_tagger, textrank_keyword, textrank_list_keywords
from tqdm import tqdm


# Country Keyword List
def add_lowercase_country_keywords(input_keyword_list):
    """
    Description: 국가쌍을 뽑아내는데 필요한 키워드가 담긴 dictionary.
    너무 길어서 따로 module화 해서 결과값만 받아오는 형식으로 바꾸려 함.
    ---------
    Return: 국가 키워드가 담긴 dictionary.
    ---------

    """
    output_keyword_list = list(
        set(input_keyword_list + list(map(str.lower, input_keyword_list)))
    )
    return sorted(output_keyword_list)


us_text = add_lowercase_country_keywords(
    [
        "US",
        "American",
        "United States",
        "Biden",
        "Whitehouse",
        "Pentagon",
        "Blinken",
    ]
)
china_text = add_lowercase_country_keywords(
    ["China", "Xi", "Xi Jinping", "Jinping", "Chinese"]
)
southkorea_text = add_lowercase_country_keywords(
    ["S.Korea", "South Korea", "Moon Jae-in", "Moon Jae in", "Jae-In", "Jaein"]
)
northkorea_text = add_lowercase_country_keywords(
    ["N.Korea", "DPRK", "North Korea", "Jongun", "Jong-un", "Kim Jong-Un"]
)
japan_text = add_lowercase_country_keywords(
    ["Japan", "Fumio Kishida", "Kishida", "Japanese"]
)
russia_text = add_lowercase_country_keywords(
    ["Russia", "Putin", "kremlin", "Russian"]
)
india_text = add_lowercase_country_keywords(
    [
        "India",
        "Narendra Modi",
        "Modi",
        "Republic of India",
        "Bhārat",
        "Bharat",
        "Indian",
    ]
)
uk_text = add_lowercase_country_keywords(
    [
        "United Kingdom",
        "Britain",
        "UK",
        "Boris Johnson",
        "Westminster",
        "Downing street",
    ]
)
indonesia_text = add_lowercase_country_keywords(
    [
        "Indonesian",
        "Indonesia",
        "Joko Widodo",
        "Joko",
        "Widodo",
        "Republic of Indonesia",
    ]
)
taiwan_text = add_lowercase_country_keywords(
    ["Taiwan", "Tsai Ing-wen", "Tsai", "Taipei"]
)
germany_text = add_lowercase_country_keywords(
    [
        "Germany",
        "Federal Republic of Germany",
        "Berlin",
        "Angela Merkel",
        "Merkel",
        "Frank Walter Steinmeier",
        "Steinmeier",
    ]
)
mexico_text = add_lowercase_country_keywords(
    [
        "Mexico",
        "United Mexican States",
        "Andres Manuel Lopez Obrador",
        "Obrador",
    ]
)
france_text = add_lowercase_country_keywords(
    ["Frence", "French Republic", "Paris", "Macron", "Emmanuel Macron"]
)
australia_text = add_lowercase_country_keywords(
    [
        "Australia",
        "Commonwealth of Australia",
        "Canberra",
        "Scott John Morrison",
        "Scott Morrison",
    ]
)
singapore_text = add_lowercase_country_keywords(
    [
        "Republic of Singapore",
        "Singapore",
        "Singapura",
        "Lee Hsien Loong",
        "Lee Hsien-Loong",
    ]
)
saudi_text = add_lowercase_country_keywords(
    ["Saudi Arabia", "Riyadh", "Saudi", "Mohammed bin Salman", "King Salman"]
)
rsa_text = add_lowercase_country_keywords(
    [
        "Republic of South Africa",
        "South Africa",
        "Pretoria",
        "Cyril Ramaphosa",
        "Ramaphosa",
    ]
)
turkey_text = add_lowercase_country_keywords(
    [
        "Republic of Turkey",
        "Turkey",
        "Ankara",
        "Recep Tayyip Erdoğan",
        "Erdoğan",
        "Erdogan",
    ]
)
italy_text = add_lowercase_country_keywords(
    [
        "Italy",
        "Italian Republic",
        "Rome",
        "Giuseppe Conte",
        "Conte",
        "Sergio Mattarella",
    ]
)

countrykeywords_dictionary = dict(
    zip(
        [
            "USA",
            "China",
            "S.Korea",
            "N.Korea",
            "Japan",
            "Russia",
            "India",
            "UK",
            "Indonesia",
            "Taiwan",
            "Germany",
            "Mexico",
            "France",
            "Australia",
            "Singapore",
            "South Africa",
            "Saudi Arabia",
            "Turkey",
            "Italy",
        ],
        [
            us_text,
            china_text,
            southkorea_text,
            northkorea_text,
            japan_text,
            russia_text,
            india_text,
            uk_text,
            indonesia_text,
            taiwan_text,
            germany_text,
            mexico_text,
            france_text,
            australia_text,
            singapore_text,
            rsa_text,
            saudi_text,
            turkey_text,
            italy_text,
        ],
    )
)
us_text.pop(-2)
china_text.pop(-2)
northkorea_text.pop(-7)
uk_text.pop(-3)
japan_text.pop(-5)


def splitby_threesentences(input_article):
    """
    Description: 세 문장씩 문장을 잘라내는 함수.

    Artuments
    ---------
    input_article : str
        full article

    return
    ---------
    joined3_article : str
    """
    input_article = " ".join(input_article.split())
    spit_article = tokenize.sent_tokenize(input_article)
    iterated_split_article = iter(spit_article)
    joined3_article = [
        i + next(iterated_split_article, "") + next(iterated_split_article, "")
        for i in iterated_split_article
    ]
    return joined3_article


def morethan_two_countries(input_text):
    """
    Description: 국가쌍에 해당되는 문장을 뽑아주는 함수

    Artuments
    ---------
    input_article : str
        full article

    return
    ---------
    Boolean: True, False
    """
    list_of_countries = []
    counter = 0
    for each_country_keywords in countrykeywords_dictionary.keys():
        if (
            len(
                re.findall(
                    "|".join(
                        countrykeywords_dictionary[each_country_keywords]
                    ),
                    input_text,
                )
            )
            > 0
        ):
            counter += 1
            list_of_countries.append(each_country_keywords)
    list_of_countries = " / ".join(list(set(list_of_countries)))

    if counter >= 3:
        exist_keyword = False
    elif counter > 1:
        exist_keyword = True
    else:
        exist_keyword = False

    vectorized_morethan_two_countries = np.vectorize(
        exist_keyword, list_of_countries, otypes=[list]
    )
    return vectorized_morethan_two_countries


def sentence_to_nerlist(input_sentence):
    """
    Description: textrank 알고리즘을 기반으로 3문장 단위로 들어오는

    Artuments
    ---------
    input_article : str
        full article

    return
    ---------
    Boolean: True, False
    """
    try:
        ner_output_list = [
            each
            for each in model.predict(input_sentence)
            if each["tag"] != "O"
        ]

        if len(ner_output_list) != 0:
            ner_output_df = pd.DataFrame(ner_output_list)
            ner_output_df["first_tag"] = list(
                map(itemgetter(0), ner_output_df["tag"].str.split("-"))
            )
            ner_output_df["second_tag"] = list(
                map(itemgetter(1), ner_output_df["tag"].str.split("-"))
            )
            entity_list = []
            for idx in range(len(ner_output_df)):
                temp_entity_list = []
                if idx == len(ner_output_df):
                    temp_entity_list.append(ner_output_df["word"][idx])
                elif ner_output_df["first_tag"][idx] == "I":
                    pass
                else:
                    temp_idx = idx
                    temp_entity_list.append(ner_output_df["word"][temp_idx])
                    try:
                        while ner_output_df["first_tag"][temp_idx + 1] == "I":
                            temp_idx += 1
                            temp_entity_list.append(
                                ner_output_df["word"][temp_idx]
                            )
                    except KeyError:
                        pass
                if len(temp_entity_list) != 0:
                    entity_list.append(" ".join(temp_entity_list))
        else:
            entity_list = []
    except:
        entity_list = []
    return entity_list


def corpus_to_nerlist(input_corpus):
    return list(
        set(chain.from_iterable(map(sentence_to_nerlist, input_corpus)))
    )


def sort_sentence_importance(
    input_text, standard="mean", topn=3, countryfilter=False
):
    """
    Description: textrank 알고리즘을 기반으로 들어온 문장의 중요도를 뽑는 함수

    Artuments
    ---------
    input_article : str
        3문장 이내의 입력 문장.
    standard: str
        중요성을 판단하는 기준
    topn: int
        문장을 몇개까지 뽑을것 인지에 대해 선택하는 agrs.
    countryfilter: boolean
        2개 이상의 문장이 들어간 국가쌍을 뽑으면 True, 아니면 False

    return: pandas.series
    ---------

    """
    textrank_dict = dict(textrank_keyword(input_text, topk=30))
    output_list = []
    for idx, each_sentence in enumerate(sent_tokenize(input_text)):
        sentence_value = []
        pattern = re.compile(
            "photo|Photo|Related article|RELATED ARTICLES|Xinhua"
        )
        if not pattern.search(each_sentence):
            for each_token in word_tokenize(each_sentence):
                try:
                    sentence_value.append(textrank_dict[each_token])
                except:
                    sentence_value.append(0)
                    pass
                binary_sentence_value = list(
                    np.vectorize(lambda x: 1 if x > 0 else 0)(sentence_value)
                )

            importance_mean = np.mean(sentence_value)
            importance_sum = np.sum(sentence_value)
            importance_ratio = np.mean(binary_sentence_value)
            output_list.append(
                (
                    idx,
                    each_sentence,
                    importance_mean,
                    importance_sum,
                    importance_ratio,
                )
            )

    output_df = pd.DataFrame(
        output_list, columns=["idx", "sentence", "mean", "sum", "ratio"]
    )
    if countryfilter:
        output_df["numcountryfilter"] = np.vectorize(
            lambda x: morethan_two_countries(x)[0]
        )(output_df["sentence"])
        output_df = output_df[output_df["numcountryfilter"] == True]
    output_df = output_df.sort_values(standard, ascending=False).reset_index(
        drop=True
    )
    return output_df["sentence"][:topn].tolist()


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
    for label in range(
        len(dataframe["lowercase_" + nameof_articlebody_column])
    ):
        doc_id = label
        try:
            input_text = dataframe["lowercase_" + nameof_articlebody_column][
                label
            ]
            sentence = " ".join(
                sort_sentence_importance(input_text, standard="mean", topn=3)
            )
            output_list = morethan_two_countries(sentence)
            output_df = pd.DataFrame.from_records(
                [output_list], columns=["isvalid", "list_of_countries"]
            )
            output_df["input_text"] = sentence
            output_df["doc_id"] = doc_id
            quasinerdf_output_list.append(output_df)
        except:
            continue
    quasinerdf_output_df = pd.concat(quasinerdf_output_list, axis=0)
    quasinerdf_output_df = quasinerdf_output_df[
        quasinerdf_output_df["isvalid"] == True
    ].reset_index(drop=True)
    del quasinerdf_output_df["isvalid"]
    return quasinerdf_output_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/mobilebert-uncased",
        help="사용할 모델을 선택하시오. 차후에 하위 파일로 모델을 분리할 예정.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size를 입력하시오.",
    )
    parser.add_argument(
        "--model_directory",
        type=str,
        default="./model",
        help="Output model directory",
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default="./data",
        help="input data directory",
    )
    parser.add_argument("--epoch", type=int, default=1, help="epoch를 선택하시오.")
    parser.add_argument(
        "--random_seed", type=int, default=42, help="random_seed를 선택하시오."
    )

    args = parser.parse_args()
    model = args.model
    batch_size = args.batch_size
    model_directory = args.model_directory
    data_directory = args.data_directory
    epoch = args.epoch
    random_seed = args.random_seed
