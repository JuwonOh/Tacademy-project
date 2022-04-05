import argparse
import json
import os
import re
from time import sleep, gmtime, strftime
from DoD_scraper import get_latest_allnews
from DoD_scraper import strf_to_datetime
from DoD_scraper import news_dateformat


def save(json_obj, directory):
    date = json_obj.get('date', '')
    title = json_obj.get('title', '')
    filepath = '{}/{}_{}.json'.format(directory,
                                      date, re.sub('[^a-zA-Z ]+', "", title[10:50]))
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False,
                  sort_keys=True, default=str)


def scraping(section, begin_date, end_date, max_num, sleep, directory, verbose):

    n_exceptions = 0
# transcript
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
    parser.add_argument('--begin_date', type=str,
                        default='20210901', help='datetime YYYYmmdd')
    parser.add_argument('--end_date', type=str,
                        default='20220318', help='datetime YYYYmmdd')
    parser.add_argument('--section', type=str,
                        default='News', help='datetime YYYY-mm-dd')
    parser.add_argument('--directory', type=str,
                        default='C:/Users/13a71/Documents/crawling output/dod', help='Output directory')
    parser.add_argument('--max_num', type=int, default=2000,
                        help='Maximum number of news to be scraped')
    parser.add_argument('--sleep', type=float, default=0.00001,
                        help='Sleep time for each news')
    parser.add_argument('--verbose', dest='VERBOSE', action='store_true')

    args = parser.parse_args()
    begin_date = args.begin_date
    end_date = args.end_date
    section = args.section
    directory = args.directory
    max_num = args.max_num
    sleep = args.sleep
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    scraping(section, begin_date, end_date, max_num, sleep, directory, VERBOSE)


if __name__ == '__main__':
    main()
