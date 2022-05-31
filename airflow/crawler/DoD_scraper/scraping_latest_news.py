import argparse
import datetime
import json
import os
import re
from time import gmtime, sleep, strftime

from DoD_scraper import get_latest_allnews, news_dateformat, strf_to_datetime


def save(json_obj, directory):
    date = json_obj.get("date", "")
    title = json_obj.get("title", "")
    source = json_obj.get("source", "")
    filepath = "{}/{}_{}_{}.json".format(
        directory, source, date, re.sub("[^a-zA-Z ]+", "", title[10:50])
    )
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(
            json_obj,
            fp,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


def scraping(
    section, begin_date, end_date, max_num, sleep, directory, verbose
):

    # transcript
    for i, json_obj in enumerate(
        get_latest_allnews(section, begin_date, end_date, max_num, sleep)
    ):
        save(json_obj, directory)
        title = json_obj["title"]
        date = json_obj["date"]
        print("[{} / {}] ({}) {}".format(i + 1, max_num, date, title))


def main():
    parser = argparse.ArgumentParser()
    today = datetime.date.today().strftime("%Y-%m-%d")
    yesterday = (datetime.date.today() - datetime.timedelta(days=3)).strftime(
        "%Y-%m-%d"
    )

    parser.add_argument(
        "--begin_date", type=str, default=yesterday, help="datetime YYYYmmdd"
    )
    parser.add_argument(
        "--end_date", type=str, default=today, help="datetime YYYYmmdd"
    )
    parser.add_argument(
        "--section",
        type=str,
        default="News-Stories",
        help="datetime YYYY-mm-dd",
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
        default=20,
        help="Maximum number of news to be scraped",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.00001, help="Sleep time for each news"
    )
    parser.add_argument("--verbose", dest="VERBOSE", action="store_true")

    args = parser.parse_args()
    begin_date = args.begin_date
    print(begin_date)
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


if __name__ == "__main__":
    main()
