from .utils import get_soup
from .utils import now
from .utils import normalize_text
from dateutil.parser import parse
import re

## this function need to url from urls. use for

def parse_article(url):
    def parse_title(soup):
        title = soup.find('div', class_='padding_block single-title').find('h1').text
        if not title:
            return 'title error'
        
        return title
    def parse_date(soup):
        date = soup.find('time')['datetime'][:10] ## need to fix date part by regex 
        if not date:
            return 'date error'
        return parse(date).strftime("%Y-%m-%d")
    
    def parse_content(soup):
        content_temp = soup.find('div', class_='entry').find_all('p')
        content = ' '.join([a.text for a in content_temp])
        if not content:
            return ' '
        return normalize_text(content)

    soup = get_soup(url)
    category = soup.find('div', class_ = "padding_block single-title").find('a').text
    fist_dic =  {
            'url': url,
            'title': parse_title(soup),
            'subtitle': "",
            'date': parse_date(soup),
            'content' :parse_content(soup),
            'category' : category,
            'source': 'Japantimes',
            'scraping_date': now()
    }
    return fist_dic