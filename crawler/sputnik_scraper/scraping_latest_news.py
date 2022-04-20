import argparse
import json
import os
import re
import datetime
from dateutil.parser import parse
from sputnik_scraper import yield_latest_article
from sputnik_scraper import now


def save(json_obj, directory):
    date = json_obj.get('date', '')
    title = json_obj.get('title', '')
    filepath = '{}/{}_{}.json'.format(directory,
                                      date, re.sub('[^a-zA-Z ]+', "", title[:50]))
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as fp:
            json.dump(json_obj, fp, indent=2, ensure_ascii=False)
            print('scraped {}'.format(json_obj['title']))
    else:
        print('this {}'.format(json_obj['title']) + 'is already scraped')


def main():
    parser = argparse.ArgumentParser()
    today = now()[:10]
    parser.add_argument('--directory', type=str,
                        default='C:/Users/13a71/Documents/crawling output/sputnik_outcome', help='Output directory')
    parser.add_argument('--sleep', type=float, default=0.01,
                        help='Sleep time for each submission (post)')
    parser.add_argument('--section', type=str, default=1000,
                        help='Choose newspaper`s section')
    parser.add_argument('--end_date', type=str,
                        default="2021-10-01", help='Number of end documents')
    parser.add_argument('--begin_date', type=str,
                        default="2021-11-08", help='Number of start documents')
    parser.add_argument('--verbose', dest='VERBOSE', action='store_true')

    args = parser.parse_args()
    directory = args.directory
    sleep = args.sleep
    section = args.section
    end_date = args.end_date
    begin_date = args.begin_date
    VERBOSE = args.VERBOSE

    # check output directory
    #directory += '/%s' % section
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in yield_latest_article(begin_date, end_date, sleep, section):
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
