import argparse
import datetime
import json
import os
import re

from numpy import source
from whitehouse_scraper import (
    news_dateformat,
    strf_to_datetime,
    yield_latest_allnews,
)


def save(json_obj, directory):
    date = json_obj.get("date", "")
    title = json_obj.get("title", "")
    source = json_obj.get("source", "")
    filepath = "{}/{}_{}.json".format(
        directory, source, date, re.sub("[\/:*?\<>|%]", "", title[:50])
    )
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False)


def scraping(begin_date, max_num, sleep, directory, verbose):

    n_exceptions = 0
    for i, json_obj in enumerate(
        yield_latest_allnews(begin_date, max_num, sleep)
    ):
        try:
            save(json_obj, directory)
        except Exception as e:
            n_exceptions += 1
            print(e)
            continue

        if verbose:
            url = json_obj["url"]
            time = json_obj["date"]
            print("[{} / {}] ({}) {}".format(i + 1, max_num, time, url))

    if n_exceptions > 0:
        print("Exist %d exceptions" % n_exceptions)


def main():
    parser = argparse.ArgumentParser()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    parser.add_argument(
        "--begin_date", type=str, default=yesterday, help="datetime YYYY-mm-dd"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="/home/joh87411/output",
        help="Output directory",
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=1500,
        help="Maximum number of news to be scraped",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Sleep time for each news"
    )
    parser.add_argument("--verbose", dest="VERBOSE", action="store_true")

    args = parser.parse_args()
    begin_date = args.begin_date
    directory = args.directory
    max_num = args.max_num
    sleep = args.sleep
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    scraping(begin_date, max_num, sleep, directory, VERBOSE)


if __name__ == "__main__":
    main()
