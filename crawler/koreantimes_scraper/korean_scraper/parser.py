from .utils import get_soup
from .utils import now
from .utils import normalize_text
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse
import re

## this function need to url from urls. use for

def parse_article(url):
    def parse_title(soup):
        title = soup.find('div', class_='view_headline HD')
        if not title:
            return 'title error'
        return title.text
    def parse_date(soup):
        date = soup.find_all('div', class_ = 'view_date')[0].text[9:19] ## need to fix date part by regex 
        if not date:
            return 'date error'
        return parse(date).strftime("%Y-%m-%d")
    
    def parse_content(soup):
        content = soup.find('div', itemprop='articleBody')
        return normalize_text(content.text)

    soup = get_soup(url)

    fist_dic =  {
            'url': url,
            'title': parse_title(soup),
            'subtitle' : "",
            'date': parse_date(soup),
            'content' :parse_content(soup),
            'category' : url.split("/")[4],
            'source': 'koreatimes',
            'scrap_time': now()
    }
    return fist_dic