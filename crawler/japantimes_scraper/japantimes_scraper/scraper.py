import json
import time
import requests
import re
from dateutil.parser import parse
from .utils import get_soup
from .parser import parse_article

## caution: global times limit page 100, you should use date 


def yield_latest_article(section, begin_date, end_date, sleep=1.0):
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
    patterns = [
    re.compile('https://www.japantimes.co.jp/news/2022/[\w]+')]
    def is_matched(url):
        for pattern in patterns:
            if pattern.match(url):
                return True
        return False

    # prepare parameters
    d_begin = parse(begin_date)
    d_end = parse(end_date)
    outdate = False

    for page in range(1, 1000):
        # get urls
        page = str(page)
        url = "https://www.japantimes.co.jp/news/{}/page/{}/".format(section, page)
        soup = get_soup(url)
        sub_links = soup.find('div', class_ = 'main_content').find('div', class_ = 'padding_block').find_all('a')
        links = [a['href'] for a in sub_links]
        links = list(set([url for url in links if is_matched(url)]))
        print(links)

        for a in links:
            try:
                json_obj = parse_article(a)
            except:
                continue
            d_news = parse(json_obj['date'])
            if d_news > d_end:
                print('The latest news has been created after {}'.format(d_end))
                continue

            if d_begin > d_news:
                print('The oldest news has been created after {}'.format(begin_date))
                break

            yield json_obj
            time.sleep(sleep)