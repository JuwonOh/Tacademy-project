import re
from dateutil.parser import parse
from .utils import get_soup
from .utils import now


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
        It consists of
            date : publication date of article
            title : article title
            subtitle : none
            content : text 
            category : World
            source : Guardian
            url : web page url
            scrap_time : scrapped time
    """

    # try:
    soup = get_soup(url)
    try:
        main_content = soup.find('div', class_='article__announce-text').text
        content_list = soup.find_all('div', class_='article__text')
    except:
        return 'error'

    content = ''
    for paragraph in content_list:
        content = content + to_string(paragraph) + ' '
    content = main_content + content.replace(u'\xa0', u' ')

    title = soup.find('h1', class_='article__title')
    title = title.text.replace('\n', '')

    time = parse(
        soup.find('div', class_='article__info-date').text[:22]).strftime("%Y-%m-%d")

    category_list = soup.find_all('li', class_='tag')
    category_all = ''
    for category in category_list:
        category_all = category_all + category.find('a').text + ', '

    json_object = {
        'date': time,
        'title': title,
        'subtitle': '',
        'content': content,
        'category': category.text,
        'source': 'Sputnik',
        'url': url,
        'scrap_time': now()
    }
    return json_object
    """
    except Exception as e:
        return e
    """
