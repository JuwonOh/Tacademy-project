import json
import time
import requests
import re
from dateutil.parser import parse
from .utils import get_soup

from .parser import parse_article

from dateutil.relativedelta import relativedelta
from datetime import datetime

## caution: global times limit page 100, you should use date 

def yield_latest_article(begin_date, sleep=0.1):
    """
    Artuments
    ---------
    begin_date : str
        eg. 2018-07-01
    end_date :str
        eg. 2019-03-31
    max_num : int
        Maximum number of news to be scraped
    sleep : float
        Sleep time. Default 1.0 sec

    It yields
    ---------
    news : json object
    """

    # prepare parameters
    d_begin = parse(begin_date)
    end_page = 100
    n_news = 0
    

     # get urls
    ymonth = d_begin.strftime("%Y%m")
    url = "https://japan.kantei.go.jp/101_kishida/actions/{}/index.html".format(ymonth)

    soup = get_soup(url)
    links = ["https://japan.kantei.go.jp" + url['href'] for url in soup.find('div', class_='main-left png').find_all('a')]

    for url in links:
        print(url)
        try:
            news_json = parse_article(url)
            yield news_json

            # check date of scraped news

        except:
            try:
                month_after = (datetime.strptime(begin_date, "%Y-%m-%d") + relativedelta(months = 1)).strftime("%Y%m")
                url = "https://japan.kantei.go.jp/101_kishida/actions/{}/index.html".format(ymonth)
                news_json = parse_article(url)
                yield news_json

            except:
                print("This url is not available")