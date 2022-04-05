import re
import time
from dateutil.parser import parse
from .parser import parse_page
from .utils import get_soup


patterns = [
    re.compile('https://www.whitehouse.gov/briefing-room/[\w]+')]
url_base = 'https://www.whitehouse.gov/briefing-room/page/{}/'


def is_matched(url):
    for pattern in patterns:
        if pattern.match(url):
            return True
    return False

def yield_latest_allnews(begin_date, max_num=10, sleep=1.0):
    """
    Artuments
    ---------
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

    # prepare parameters
    d_begin = parse(begin_date).strftime("%Y-%m-%d")
    end_page = get_last_page_num()
    n_news = 0
    outdate = False

    for page in range(1, end_page+1):

        # check number of scraped news
        if n_news >= max_num or outdate:
            break

        # get urls
        url = url_base.format(page)
        soup = get_soup(url)
        links = soup.find_all('a', class_ = 'news-item__title')
        urls = [i['href'] for i in links]
        urls = [url for url in urls if is_matched(url)]

        # scrap
        for url in urls:
            print(url)

            news_json = parse_page(url)
            

            # check date

            
            d_news = news_json['date']

            if d_begin > d_news:
                outdate = True
                print('Stop scrapping. {} / {} news was scrapped'.format(n_news, max_num))
                print('The oldest news has been created after {}'.format(begin_date))
                break

            # yield
            yield news_json

            # check number of scraped news
            n_news += 1
            if n_news >= max_num:
                break
            time.sleep(sleep)

def get_allnews_urls(begin_page=1, end_page=3, verbose=True):
    """
    Arguments
    ---------
    begin_page : int
        Default is 1
    end_page : int
        Default is 3
    verbose : Boolean
        If True, print current status

    Returns
    -------
    urls_all : list of str
        List of urls
    """

    urls_all = []
    for page in range(begin_page, end_page+1):
        url = url_base.format(page)
        soup = get_soup(url)
        links = soup.select('a[href^=https://www.whitehouse.gov/]')
        urls = [link.attrs.get('href', '') for link in links]
        urls = [url for url in urls if is_matched(url)]
        urls_all += urls
        if verbose:
            print('get briefing statement urls {} / {}'.format(page, end_page))
    return urls_all

def get_last_page_num():
    """
    Returns
    -------
    page : int
        Last page number. 
        eg: 503 in 'https://www.whitehouse.gov/news/page/503'
    """

    def last_element(url):
        parts = [p for p in url.split('/') if p]
        return int(parts[-1])

    soup = get_soup('https://www.whitehouse.gov/news/')
    last_page = max(
        last_element(a.attrs['href'])
        for a in soup.select('a[class=page-numbers]')
    )

    return last_page