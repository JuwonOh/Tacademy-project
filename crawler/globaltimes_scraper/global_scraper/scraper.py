import json
import time
import requests
import re
from dateutil.parser import parse
from .utils import get_soup
from .parser import parse_article
from dateutil.relativedelta import relativedelta
from datetime import datetime

# caution: global times limit page 100, you should use date


def yield_latest_article(begin_date, max_num=10, sleep=0.1):
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
    outdate = False

    for page in range(1255227, 1319925):

        # check number of scraped news
        if n_news >= max_num or outdate:
            break

        # get urls
        page = str(page)
        ymonth = d_begin.strftime("%Y%m")
        url = "https://www.globaltimes.cn/page/{}/{}.shtml".format(
            ymonth, page)

        print(url)
        try:
            news_json = parse_article(url)
            yield news_json
            # check date of scraped news

        except:
            try:
                ymonth = (
                    datetime.strptime(begin_date, "%Y-%m-%d") +
                    relativedelta(months=1)
                ).strftime("%Y%m")
                url = "https://www.globaltimes.cn/page/{}/{}.shtml".format(
                    ymonth, page)
                news_json = parse_article(url)
                yield news_json

            except:
                print("This url is not available")
