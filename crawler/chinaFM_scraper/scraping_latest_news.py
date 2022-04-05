import argparse
import json
import os
import re
from time import sleep, gmtime, strftime
from chinaFM_scraper import get_latest_allnews
from chinaFM_scraper import strf_to_datetime
from chinaFM_scraper import news_dateformat


def save(json_obj, directory):
    date = json_obj.get('date', '')
    title = json_obj.get('title', '')
    filepath = '{}/{}_{}.json'.format(directory, date, re.sub('[^a-zA-Z ]+',"", title[10:50]))
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False, sort_keys=True, default=str)

def scraping(section, begin_date, end_date, max_num, sleep, directory, verbose):

    n_exceptions = 0
#transcript
    for i, json_obj in enumerate(get_latest_allnews(section, begin_date, end_date, max_num, sleep)):
        save(json_obj, directory)
        print(json_obj.get('url') + "is scrapped")
        
        if verbose:
            title = json_obj['title']
            date = json_obj['date']
            print('[{} / {}] ({}) {}'.format(i+1, max_num, date, title))

def main():
    parser = argparse.ArgumentParser()
    today = strftime("%Y-%m-%d", gmtime())
    parser.add_argument('--section', type=str, default= 'News', help='datetime YYYY-mm-dd')
    parser.add_argument('--directory', type=str, default='C:/Users/13a71/Documents/crawling output/chn_fm', help='Output directory')
    parser.add_argument('--max_num', type=int, default=2000, help='Maximum number of news to be scraped')
    parser.add_argument('--sleep', type=float, default=0.00001, help='Sleep time for each news')
    parser.add_argument('--verbose', dest='VERBOSE', action='store_true')

    args = parser.parse_args()
    section = args.section
    directory = args.directory
    max_num = args.max_num
    sleep = args.sleep
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in get_latest_allnews(section, max_num, sleep):
        try:
            save(article, directory)
            print('scraped {}'.format(article.get('url'), ''))
        except Exception as e:
            n_exceptions += 1
            print(e)
            continue
        if n_exceptions > 0:
            print('Exist %d article exceptions' % n_exceptions)

if __name__ == '__main__':
    main()
