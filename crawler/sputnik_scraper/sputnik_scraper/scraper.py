import re
import time
from dateutil.parser import parse
import pandas as pd
from .parser import parse_page
from .utils import get_soup


def yield_latest_article(begin_date, end_date, sleep, section, verbose=True):
    """
    Arguments
    ---------
    begin & end : datetime.datetime
    pagenum : int
    verbose : Boolean
        If True, print current status

    Returns
    -------
    links_all : list of str
        List of urls on the page pagenum
    """
    daily_datarange = pd.date_range(start=begin_date,
                                    end=end_date,
                                    freq='b')

    for date in daily_datarange:
        base_url = "https://sputniknews.com/services/{}/more.html?date={}&tags=1"
        url = base_url.format(section, date)
        soup = get_soup(url)
        old_url = set()

        sub_urls = soup.find_all('div', class_='list__content')
        daily_url = set(['https://sputniknews.com' +
                        a.find('a')['href'] for a in sub_urls])
        # remove duplicate url
        links = list(daily_url - old_url)
        # update new urls
        old_url = set()
        old_url.update(daily_url)

        for url in links:
            print(url)
            news_json = parse_page(url)
            yield news_json
            time.sleep(sleep)
