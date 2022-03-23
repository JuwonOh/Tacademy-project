from operator import itemgetter

import pandas as pd


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
