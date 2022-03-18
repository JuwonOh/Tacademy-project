import re


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
        ["USA", "China", "S.Korea", "N.Korea", "Japan", "Russia", "India"],
        [
            us_text,
            china_text,
            southkorea_text,
            northkorea_text,
            japan_text,
            russia_text,
            india_text,
        ],
    )
)
us_text.pop(-2)
china_text.pop(-2)
northkorea_text.pop(-7)
uk_text.pop(-3)
japan_text.pop(-5)


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
