import re
from .utils import get_soup
from .utils import now
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

    soup = get_soup(url)
    title = soup.find('h1', class_= 'featured-content__headline stars-above').text.replace('\t','').replace('\n','').replace(u"\u202f", " ")
    

    date = parse(soup.find('p', class_= 'article-meta__publish-date').text).strftime('%Y-%m-%d')
    content = soup.find('div', class_ = 'entry-content').text.replace('\n','')

    return {
        'title' : title,
        'date' : date,
        'content' : content,
        'url' : url,
        'scrap_time' : now()
    }
