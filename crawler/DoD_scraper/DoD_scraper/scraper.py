import time
from .parser import parse_page
from .utils import get_soup
from dateutil.parser import parse

def get_latest_allnews(section, begin_date, end_date, max_num=10, sleep=1.0):
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
    links_all : list of str
        List of urls
    """
    section = str(section)
    for pagenum in range(1, max_num):   

        url = "https://www.defense.gov/{}/News-Stories/StartDate/{}/EndDate/{}/?Page={}".format(section, begin_date, end_date, pagenum)
        soup = get_soup(url)
        sub_url = soup.find_all('listing-with-preview')
        urls = [a['article-url'] for a in sub_url]
        
        for each_url in urls:
            print(each_url)
            try:
                news_json = parse_page(each_url)
            except:
                pass
            
            # yield
            yield news_json
            time.sleep(sleep)
