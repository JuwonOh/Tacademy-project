import re
import time
from dateutil.parser import parse
from .parser import parse_page
from .utils import get_soup


def is_matched(url):
    for pattern in patterns:
        if pattern.match(url):
            return True
    return False


patterns = [re.compile('https://www.state.gov/.+')]

press_url = 'https://www.state.gov/press-releases/page/{}/'


def yield_latest_press(begin_date, max_num=100, sleep=0.01):
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
    d_begin = parse(begin_date)
    end_page = 100
    n_news = 0
    outdate = False

    for page in range(8, end_page+1):

        # check number of scraped news
        if n_news >= max_num or outdate:
            break

        # get urls

        url = press_url.format(page)
        soup = get_soup(url)
        sub_links = soup.find_all('li', class_='collection-result')
        links = [i.find('a')['href'] for i in sub_links]
        links_all = [url for url in links if is_matched(url)]

        # scrap
        for url in links_all:
            news_json = parse_page(url)
            print(url)
            try:
                # check date
                d_news = parse(news_json['date'])
                if d_begin > d_news:
                    outdate = True
                    print(
                        'Stop scrapping. {} / {} blog was scrapped'.format(n_news, max_num))
                    print(
                        'The oldest article has been created after {}'.format(begin_date))
                    break

                # yield
                yield news_json
            except Exception as e:
                print(e)
                print('Parsing error from {}'.format(url))
                return None

            # check number of scraped news
            n_news += 1
            if n_news >= max_num:
                break
            time.sleep(sleep)
