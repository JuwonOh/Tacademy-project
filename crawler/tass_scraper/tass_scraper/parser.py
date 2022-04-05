from .utils import get_soup
from .utils import now
from .utils import normalize_text
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse
import re

## this function need to url from urls. use for


def parse_article(url):
    def parse_title(soup):
        title = soup.find('div', class_ ='news-header').text.replace('\n\n', '')
        subtitle = soup.find('div', class_ ='news-header__lead').text.replace('\n\n', '')
        fulltitle = title + '. ' +subtitle
        
        if not title:
            return 'title error'
        return title        
    
    def parse_date(soup):
        start = re.search("article_publication_date", str(soup)).end() + 3
        end = start + 16
        time = parse(str(soup)[start:end]).strftime("%Y-%m-%d")
        if not time:
            return 'date error'
        return time
    
    def parse_content(soup):
        content = soup.find('div', class_ = 'text-block').text.replace('’s', '`s')
        if not content:
            return ' '
        return normalize_text(content).replace(u'\xa0', u' ')

    def parse_category(soup):
        category = url.split('/')[3]
        if not category:
            return ' '
        return category
    soup = get_soup(url)

    fist_dic =  {
            'url': url,
            'title': parse_title(soup), 
            'date': parse_date(soup),
            'content' :parse_content(soup),
            'category' : parse_category(soup).replace(" ", ""),
            'source': "Tass (Russaia)",
            'scraping_date': now()
    }
    return fist_dic