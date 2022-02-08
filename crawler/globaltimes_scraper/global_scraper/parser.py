from .utils import get_soup
from .utils import now
from .utils import normalize_text
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse
import re

## this function need to url from urls. use for


def parse_article(url):
    def parse_title(soup):
        title = soup.find('div', class_ = 'article_title').text
        title = re.sub('â\x80\x99', '`' , title, 10)
        title = re.sub('â\x80\x98', '`' , title, 10)
        
        if not title:
            return 'title error'
        return title
    
    def parse_subtitle(soup):
        try:
            sub_title = soup.find('div', class_= 'article_subtitle').text
            sub_title = re.sub('â\x80\x99', '`' , sub_title, 10)
            sub_title = re.sub('â\x80\x98', '`' , sub_title, 10)
            return normalize_text(sub_title)
        except:
            return 'sub_title error'
        
    
    def parse_date(soup):
        date = soup.find('span', class_ = 'pub_time').text[11:-9]  ## need to fix date part by regex 
        date = re.sub('00', '20', parse(date).strftime("%Y-%m-%d"))
        if not date:
            return 'date error'
        return date
    
    def parse_content(soup):
        sub_content = soup.find('div', class_='article_content')
        content = ' '.join(sent_tokenize(sub_content.text))
        if not content:
            return ' '
        return normalize_text(content).replace(u'\xa0', u' ').replace('\n', '')

    def parse_category(soup):
        category = re.sub('\xa0', '' , soup.find('div', class_ = 'article_column').text)
        if not category:
            return ' '
        return category
    soup = get_soup(url)

    fist_dic =  {
            'url': url,
            'title': parse_title(soup),
            'subtitle' : parse_subtitle(soup), 
            'date': parse_date(soup),
            'content' :parse_content(soup),
            'category' : parse_category(soup).replace(" ", ""),
            'source': "Global Times(China)",
            'scraping_date': now()
    }
    return fist_dic