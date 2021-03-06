import re
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from progressbar import ProgressBar

from .countryset import morethan_two_countries
from ._ner import sentence_to_nerlist
from ._textrank import sort_sentence_importance


class NewspieacePreprocess:
    """새로이 크롤링된 뉴스 기사들을 전처리해주는 모듈

    Parameters
    ---------
    body_column: srt
        전처리 해야하는 데이터의 column이 어디인지 설정한다.
    """

    def __init__(self, body_column):
        self._body_column = body_column

    def run_preprocessing(self, data):  # data = news.json or news.csv
        """주어진 data의 특정 컬럼의 이름을 전처리, 결측치 imputation, feature를 이용한 새로운 변수정의, labeling, 필요없는 컬럼삭제 등을
                       통해 전처리한 data를 반환합니다.
        Parameters
        -------------
        data: train에 사용할 raw data

        Return
        -------------
        ner_df: pandas dataframe
            전처리 후의 데이터
        """
        print(data["content"])
        ner_df = self.quasiNER_extractor3(data, "content")
        ner_df["sententce_ner"] = ""
        for a in range(len(ner_df)):
            try:
                ner_df["sententce_ner"][a] = [
                    each
                    for each in model.predict(ner_df["input_text"][a])
                    if each["tag"] != self._tag_filter_value
                ]
            except:
                ner_df["sententce_ner"][a] = ""
        return ner_df

    def corpus_to_nerlist(input_corpus):
        return list(
            set(chain.from_iterable(map(sentence_to_nerlist, input_corpus)))
        )

    def quasiNER_extractor3(self, dataframe):
        """기존에 앞에서 작성한 함수를 연결하여 실행하는 코드이다.

        Parameters
        ---------
        dataframe: pandas dataframe
            국가간 관계가 들어가 있는 데이터 프레임을 넣는다.
        nameof_articlebody_column: str
            데이터 프레임의 칼럼 이름을 넣는다.
        ---------
        Return: 
            전처리가 전부 완료된 데이터 프레임
        """
        dataframe["lowercase_" + self.body_column] = dataframe[
            self.body_column
        ]
        quasinerdf_output_list = []
        dataframe["doc_id"] = ""
        except_sentence = []
        for doc_id in ProgressBar(
            range(len(dataframe["lowercase_" + self.body_column]))
        ):
            dataframe["doc_id"][doc_id] = doc_id
            input_text = dataframe["lowercase_" + self.body_column][
                doc_id
            ]
            try:
                sentence = " ".join(
                    sort_sentence_importance(
                        input_text, standard="mean", topn=3
                    )
                )
            except:
                except_sentence.append(doc_id)
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
