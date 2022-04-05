import re
from dateutil.parser import parse
from bs4 import BeautifulSoup, NavigableString
from .utils import get_soup
from .utils import now
from .utils import normalize_text


def to_string(instance):
    final_string = ""
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
    soup = get_soup(url)

    content_list = soup.find("div", class_="content").find_all("p")
    content = ""
    for paragraph in content_list:
        content = content + to_string(paragraph) + " "
        content = content.replace(u"\xa0", u" ").replace("\n", "").replace("'s", "`s")

    try:
        title = soup.find("h1", class_="Btitle").text
    except:
        title = soup.find("h1").text
    title = title.replace("\r\n", "")

    try:
        time = soup.find("i", class_="time").text
    except:
        time = soup.find("p", class_="time").text
    time = parse(time).strftime("%Y-%m-%d")
    json_object = {
        "title": title,
        "date": time,
        "subtitle": "",
        "content": content,
        "source": "Xinhua",
        "url": url,
        "scrap_time": now(),
    }
    return json_object
