# DoD_scraper

미국 국방부(https://www.cato.org/)의 자료들을 받아오기 위한 크롤러입니다.

## User guide

크롤러의 파이썬 파일은 util.py, scraper.py 그리고 parser.py 총 세가지로 구성되어 있습니다. 
util.py는 크롤링 한 파이썬의 beautifulsoup 패키지를 받아서 url내의 html정보를 정리합니다.
scraper는 util.py내의 사이트내의 url 링크들을 get_soup함수를 통해 모아줍니다.
parser는 이렇게 만들어진 url리스트를 통해서 각 분석들의 제목/일자/내용을 모아줍니다.


Using Python script with arguments

| Argument name | Default Value | Note |
| --- | --- | --- |
| begin_date | 2019-01-10 | datetime YYYY-mm-dd |
| directory | ./output/ | Output directory |
| max_num | 100 | Maximum number of news to be scraped |
| sleep | 1.0 | Sleep time for each news |
| verbose | False, store_true | If True use verbose mode |

```
python scraping_latest_news.py
```

```
[1 / 10] (January 23, 2019) Temporary Protected Status and Immigration to the United States
[2 / 10] (January 22, 2019) How ‘Market Failure’ Arguments Lead to Misguided Policy
[3 / 10] (January 16, 2019) Do 40-Year-Old Facts Still Matter?: Long-Run Effects of Federal Oversight under the Voting Rights Act
[4 / 10] (January 15, 2019) Do Immigrants Import Terrorism?
[5 / 10] (January 15, 2019) The Myth of the Cyber Offense: The Case for Restraint
[6 / 10] (January 9, 2019) More Legislation, More Violence? The Impact of Dodd-Frank in the Democratic Republic of the Congo
[7 / 10] (January 8, 2019) The Case for an Immigration Tariff: How to Create a Price-Based Visa Category
[8 / 10] (January 2, 2019) The Spread of Deposit Insurance and the Global Rise in Bank Asset Risk since the 1970s
[9 / 10] (December 19, 2018) How Legalizing Marijuana Is  Securing the Border: The Border Wall, Drug Smuggling, and Lessons for Immigration Policy
[10 / 10] (December 19, 2018) Militarization Fails to Enhance Police Safety or Reduce Crime but May Harm Police Reputation
```

Get news urls from specific pages

```python
from whitehouse_scraper import parse_page
from whitehouse_scraper import get_allnews_urls

urls = get_allnews_urls(begin_page=1, end_page=3, verbose=True)
for url in urls[:3]:
    json_object = parse_page(url)    
```

## 참고 코드

본 코드는 https://github.com/lovit/whitehouse_scraper를 참조하여 만들어졌습니다.
