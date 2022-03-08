import re
from collections import Counter
from itertools import chain
from operator import itemgetter

import numpy as np
import pandas as pd
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from progressbar import ProgressBar
from textrank import nltk_tagger, textrank_keyword, textrank_list_keywords
from tqdm import tqdm
from utils import countrykeywords_dictionary

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
        return False, list_of_countries
    elif counter > 1:
        return True, list_of_countries
    else:
        return False, list_of_countries


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


# def NER_check(dataframe, nameof_articlebody_column):
