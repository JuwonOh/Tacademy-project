import argparse
import json
import os
import re
from korean_scraper import yield_latest_article
from korean_scraper import now


def save(json_obj, directory):
    date = json_obj.get("date", "")
    title = json_obj.get("title", "")

    filepath = "{}/{}_{}.json".format(
        directory, date, re.sub('[\/:*?\<>|%]"', "", title[:50])
    )
    print("scraped {}".format(json_obj["title"]))
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    today = now()[:10]
    parser.add_argument(
        "--directory",
        type=str,
        default="C:/Users/13a71/Documents/crawling output/koreantimes",
        help="Output directory",
    )
    parser.add_argument("--begin_date", type=str,
                        default="20220101", help="begin_date")
    # 주의 사항: begindate와 end date 사이의 간격이 너무 길어지면, 사이트 검색 시스템이 작동하지 않습니다.
    parser.add_argument("--end_date", type=str,
                        default="20220402", help="end_date")
    parser.add_argument(
        "--sleep", type=float, default=0.1, help="Sleep time for each submission (post)"
    )
    parser.add_argument(
        "--max_page",
        type=int,
        default=10000000,
        help="Number of scrapped articles page",
    )
    parser.add_argument("--verbose", dest="VERBOSE", action="store_true")

    args = parser.parse_args()
    directory = args.directory
    begin_date = args.begin_date
    end_date = args.end_date
    sleep = args.sleep
    max_num = args.max_page
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in yield_latest_article(begin_date, end_date, max_num, sleep):
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
