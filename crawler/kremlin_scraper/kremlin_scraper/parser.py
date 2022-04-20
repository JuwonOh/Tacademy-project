import re
from .utils import get_soup
from .utils import now
from dateutil.parser import parse
from bs4 import BeautifulSoup, NavigableString


def to_string(instance):
    final_string = ''
    if isinstance(instance, NavigableString):
        return instance
    for contents in instance.contents:
        final_string += to_string(contents)
    return final_string
    
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

    soup = get_soup(url)
    title = soup.find('h1', class_='entry-title p-name').text.replace('\xa0',' ').replace('\n','')
    date = parse(soup.find('time', class_='read__published')['datetime']).strftime('%Y-%m-%d')
    sub_content = soup.find('div', class_='read__content').find_all('p')[:-4]
    category = soup.find('li', class_='p-category').text
    content = ''

    for paragraph in sub_content:
        content = content + to_string(paragraph) + ' '
    content = content.replace(u'\xa0', u' ').replace(u'\n', u' ')

    return {
        'title' : title,
        'date' : date,
        'content' : content,
        'category' : category, 
        'source' : "kremlin",
        'url' : url,
        'scrap_time' : now()
    }

    for paragraph in content_list:
        content = content + to_string(paragraph) + ' '
    content = content.replace(u'\xa0', u' ').replace(u'\n', u' ')

    return {
        'title' : title,
        'date' : date,
        'content' : content,
        'category' : category, 
        'source' : "kremlin",
        'url' : url,
        'scrap_time' : now()
    }
