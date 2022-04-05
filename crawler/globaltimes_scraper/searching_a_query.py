import argparse
import json
import os
import re
from global_scraper import yield_latest_article
from global_scraper import now
import unicodedata


def save(json_obj, directory):
    date = json_obj.get("date", "")
    title = json_obj.get("title", "")
    filepath = "{}/{}_{}.json".format(
        directory, date, re.sub("[^a-zA-Z ]+", "", title[:50])
    )
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False)
        print("scraped {}".format(json_obj["title"]))


def main():
    parser = argparse.ArgumentParser()
    today = now()[:10]
    parser.add_argument(
        "--directory",
        type=str,
        default="C:/Users/13a71/Documents/crawling output/globaltimes_outcome",
        help="Output directory",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0000001,
        help="Sleep time for each submission (post)",
    )
    parser.add_argument(
        "--max_page", type=int, default=1255227, help="Number of scrapped articles page"
    )
    parser.add_argument(
        "--begin_date", type=str, default="2022-03-04", help="Number of start documents"
    )
    parser.add_argument("--verbose", dest="VERBOSE", action="store_true")

    args = parser.parse_args()
    directory = args.directory
    sleep = args.sleep
    max_num = args.max_page
    begin_date = args.begin_date
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in yield_latest_article(begin_date, max_num, sleep):
        try:
            save(article, directory)
            print("scraped {}".format(article.get("url"), ""))
        except Exception as e:
            n_exceptions += 1
            print(e)
            continue
        if n_exceptions > 0:
            print("Exist %d article exceptions" % n_exceptions)


if __name__ == "__main__":
    main()
