import argparse
import json
import os
import re
from JPNPM_scraper import yield_latest_article
from JPNPM_scraper import now


def save(json_obj, directory):
    date = json_obj.get('date', '')
    title = json_obj.get('title', '')
    filepath = '{}/{}_{}.json'.format(directory,
                                      date, re.sub('[^a-zA-Z ]+', "", title[:50]))
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False)
        print('scraped {}'.format(json_obj['title']))


def main():
    parser = argparse.ArgumentParser()
    today = now()[:10]
    parser.add_argument('--directory', type=str,
                        default='C:/Users/13a71/Documents/crawling output/JPNPM', help='Output directory')
    parser.add_argument('--sleep', type=float, default=0.0001,
                        help='Sleep time for each submission (post)')
    parser.add_argument('--begin_date', type=str,
                        default="2022-03-01", help='Number of start documents')
    parser.add_argument('--verbose', dest='VERBOSE', action='store_true')

    args = parser.parse_args()
    directory = args.directory
    sleep = args.sleep
    begin_date = args.begin_date
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in yield_latest_article(begin_date, sleep):
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
