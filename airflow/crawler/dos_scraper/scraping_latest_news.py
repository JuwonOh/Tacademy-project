import argparse
import datetime
import json
import os
import re

from dateutil.parser import parse
from DoS_scraper import yield_latest_press


def save(json_obj, directory):
    title = json_obj["title"]
    dt = parse(json_obj["date"])
    source = json_obj["source"]
    name = "{}_{}-{}-{}_{}".format(
        source,
        dt.year,
        dt.month,
        dt.day,
        re.sub("[-\/:*" "?+\<%>|]", "", title[:50]),
    )
    filepath = "{}/{}.json".format(directory, name)
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(
            json_obj,
            fp,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


def scraping_press(begin_date, max_num, sleep, directory, verbose):
    n_exceptions = 0
    for i, json_obj in enumerate(
        yield_latest_press(begin_date, max_num, sleep)
    ):
        try:
            save(json_obj, directory)
        except Exception as e:
            n_exceptions += 1
            print(e)
            continue

        if verbose:
            title = json_obj["title"]
            date = json_obj["date"]
            print("[{} / {}] ({}) {}".format(i + 1, max_num, date, title))

    if n_exceptions > 0:
        print("Exist %d exceptions" % n_exceptions)


def main():
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime(
        "%Y%m%d"
    )
    parser = argparse.ArgumentParser()
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
        "--sleep", type=float, default=5.0, help="Sleep time for each news"
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

    scraping_press(begin_date, max_num, sleep, directory, VERBOSE)


if __name__ == "__main__":
    main()
