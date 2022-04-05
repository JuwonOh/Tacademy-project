from .utils import get_soup
from .utils import now
from .utils import normalize_text
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse
import re

## this function need to url from urls. use for


def parse_article(url):
    def parse_title(soup):
        title = soup.find('div', class_ = 'main-left png').find('h3').text        
        if not title:
            return 'title error'
        return title
    
    def parse_date(soup):
        date = soup.find('p', class_ = 'date').text  ## need to fix date part by regex 
        date = parse(date).strftime("%Y-%m-%d")
        if not date:
            return 'date error'
        return date
    
    def parse_content(soup):
        sub_content = soup.find('div', id ='format').text
        content = ' '.join(sent_tokenize(sub_content))
        if not content:
            return ' '
        return normalize_text(content).replace(u'\xa0', u' ').replace('\n', '')

    soup = get_soup(url)

    fist_dic =  {
            'url': url,
            'title': parse_title(soup), 
            'date': parse_date(soup),
            'content' :parse_content(soup),
            'category' : "JPNPM",
            'source': "JPNPM",
            'scraping_date': now()
    }
    return fist_dic