import re
import time
from dateutil.parser import parse
from .parser import parse_page
from .utils import get_soup


def yield_latest_allnews(query, begin_date, end_page=10, sleep=1.0):
    """
    Artuments
    ---------
    section : str
        eg. "world, asia, opinion"
    begin_date : str
        eg. 2018-01-01
    max_num : int
        Maximum number of news to be scraped
    sleep : float
        Sleep time. Default 1.0 sec
    It yields
    ---------
    news : json object
    """
    base_url = "https://search.news.cn/getNews?sortField=0&searchFields=1&keyword={}&curPage={}&lang=en"

    # prepare parameters
    d_begin = parse(begin_date)
    n_news = 0
    outdate = False

    for page in range(1, end_page):

        # check number of scraped news
        if outdate:
            break

        # get urls
        url = base_url.format(query, page)
        soup = get_soup(url)

        url_list = soup.text.split(",{")
        real_url = []
        try:
            for url in url_list:
                end = re.search("url", url).end() + 3
                real_url.append(url[end:-2])
            real_url[-1] = real_url[-1][:-38]
        except:
            pass

        print(real_url)

        for url in set(real_url):
            try:
                news_json = parse_page(url)
                print(url, "is complete")
                yield news_json
            except:
                print("this url is not working")
