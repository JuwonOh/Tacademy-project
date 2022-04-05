import re
from .utils import get_soup
from .utils import now
from dateutil.parser import parse
from .utils import strf_to_datetime
from .utils import news_dateformat

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
    
    title = soup.find('h1', class_= 'maintitle').text
    title = re.sub('\n', '' , title, 100)
    title = re.sub('\r', '' , title, 100)
    phrases = soup.find('div', class_= 'content content-wrap').find_all('p')
    content = '\n'.join([p.text.strip() for p in phrases])
    content = re.sub('\n', '' , content, 10000)
    content = re.sub('\t', '' , content, 10000)
    content = re.sub('\r', '' , content, 10000)
    content = re.sub('\xa0', '' , content, 1000)
    time = soup.find('span', class_ = 'date')
    print(time.text[:-50].replace('|', ' '))
    date =  parse(time.text[:-50].replace('|', ' ')).strftime("%Y-%m-%d")

    json_object = {
        'title' : title,
        'date' : date,
        'content' : content,
        'url' : url,
        'source' : "US Department of Defense",
        'scrap_time': now()}

    return json_object