from .utils import get_soup
from .utils import now
import re
from dateutil.parser import parse

def parse_page(url):
    """
    Argument
    --------
    url : str
        Web page url

    Returns
    -------
    json_object : dict
        JSON format web page contents
        It consists with
            title : article title
            time : article written time
            content : text with line separator \\n
            url : web page url
            scrap_time : scrapped time
    """

    try:
        soup = get_soup(url)
        title = soup.find('h1', class_ = 'page-title topper__title news').text 
        title = re.sub('\n', '' , title, 100).replace(u"\u202f", " ")
        title = re.sub('\t', '' , title, 100).replace(u"\xa0", " ")
        date = soup.find('time', class_='posted-on entry-date published updated').text
        date = parse(date).strftime("%Y-%m-%d")
        phrases = soup.find('section', class_='body-content').find_all('p')
        content = '\n'.join([p.text.strip() for p in phrases])
        content = re.sub('\n', '' , content, 100)
        content = re.sub('\t', '' , content, 100)
        content = re.sub('\xa0', '' , content, 1000).replace(u"\u202f", " ")
        

        
        json_object = {
            'title' : title,
            'date' : date,
            'content' : content,
            'url' : url,
            'scrap_time' : now()
        }
        return json_object
    except Exception as e:
        print(e)
        print('Parsing error from {}'.format(url))
        return None