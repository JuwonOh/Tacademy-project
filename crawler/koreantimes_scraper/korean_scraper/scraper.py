import json
import time
import requests
import re
from dateutil.parser import parse
from .utils import get_soup
from .parser import parse_article

## caution: global times limit page 100, you should use date 


def yield_latest_article(begin_date,end_date,  max_num=10, sleep=1.0):
    """
    Artuments
    ---------
    begin_date : str
        eg. 20180701
    end_date :str
        eg. 20190331
    max_num : int
        Maximum number of news to be scraped
    sleep : float
        Sleep time. Default 1.0 sec

    It yields
    ---------
    news : json object
    """

    # prepare parameters

    for page in range(1, 10000):
        # get urls
        page = str(page)
        url = "https://www.koreatimes.co.kr/www2/common/search.asp?kwd=&pageNum={}&pageSize=10&category=total&sort=&startDate={}&endDate={}&date=0&srchFd=&range=&author=all&authorData=&mysrchFd=".format(page, begin_date, end_date)
        soup = get_soup(url)
        sub_links = soup.find('tbody', id = 'divSearchList').find_all('a')
        links = [a['href'] for a in sub_links]

        for a in links:
            try:
                print(a)
                news_json = parse_article(a)
                yield(news_json)
                time.sleep(sleep)
            except:
                print("this "+url+" is not working")
                continue