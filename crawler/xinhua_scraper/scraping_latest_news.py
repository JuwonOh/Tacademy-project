import argparse
import json
import os
import re
import datetime
from dateutil.parser import parse
from xinhua_scraper import yield_latest_allnews
from xinhua_scraper import now


def save(json_obj, directory):
    date = json_obj.get("date", "")
    title = json_obj.get("title", "")
    filepath = "{}/{}_{}.json".format(
        directory, date, re.sub("[^a-zA-Z ]+", "", title[:50])
    )
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False)
        print("scraped {}".format(json_obj["title"][:10]))


def main():
    parser = argparse.ArgumentParser()
    today = now()[:10]
    parser.add_argument(
        "--directory",
        type=str,
        default="C:/Users/13a71/Documents/crawling output/xinhua",
        help="Output directory",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5, help="Sleep time for each submission (post)"
    )
    parser.add_argument(
        "--begin_date", type=str, default="2020-07-01", help="Number of start documents"
    )
    parser.add_argument("--query", type=str,
                        default="Biden", help="query keyword")
    parser.add_argument("--end_page", type=int, default="300", help="end_page")
    parser.add_argument("--VERBOSE", dest="VERBOSE", action="store_true")

    args = parser.parse_args()
    directory = args.directory
    sleep = args.sleep
    end_page = args.end_page
    query = args.query
    begin_date = args.begin_date
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in yield_latest_allnews(query, begin_date, end_page, sleep):
        save(article, directory)
        print("scraped {}".format(article.get("url"), ""))


if __name__ == "__main__":
    main()
