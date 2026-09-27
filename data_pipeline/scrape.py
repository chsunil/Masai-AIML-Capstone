"""Scrape book listings from books.toscrape.com into raw_books.csv.

Scope: the categories listed in CATEGORIES (5 categories, ~93 books), which
satisfies the brief's ">= 3 categories and >= 60 books" requirement. Pagination
is followed automatically, so categories larger than one page (e.g. Mystery,
32 books over 2 pages) are captured in full.

Fields captured exactly as listed on the site, with no type conversion --
cleaning is clean.py's job.
"""

import csv
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
OUTPUT_CSV = "raw_books.csv"
REQUEST_DELAY = 0.2  # be polite to a free practice site

CATEGORIES = ["Travel", "Mystery", "Historical Fiction", "Classics", "Philosophy"]


def get_soup(url):
    """Fetch a page and parse it.

    books.toscrape.com serves UTF-8 but sends no charset in its Content-Type
    header, so requests falls back to ISO-8859-1 and the pound sign in prices
    decodes as 'Â£'. Forcing the encoding fixes the price strings at the source.
    """
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "lxml")


def find_category_urls(wanted):
    """Map category names to their listing URLs, using the homepage sidebar."""
    soup = get_soup(BASE_URL)
    available = {
        link.get_text(strip=True): urljoin(BASE_URL, link["href"])
        for link in soup.select("div.side_categories ul li ul li a")
    }

    missing = [name for name in wanted if name not in available]
    if missing:
        raise SystemExit(f"Categories not found on the site: {missing}")

    return {name: available[name] for name in wanted}


def parse_book(pod, category):
    """Extract one book's raw fields from an <article class="product_pod">."""
    link = pod.select_one("h3 a")
    # The <a> text is truncated on listing pages ("A Light in the ..."), so the
    # full title comes from the title attribute instead.
    title = link["title"]

    # class is e.g. ["star-rating", "Three"] -- the word is the rating.
    star_rating = pod.select_one("p.star-rating")["class"][1]

    return {
        "title": title,
        "price": pod.select_one("p.price_color").get_text(strip=True),
        "star_rating": star_rating,
        "availability": pod.select_one("p.instock.availability").get_text(strip=True),
        "category": category,
    }


def scrape_category(name, url):
    """Scrape every book in one category, following pagination."""
    books = []
    page_url = url

    while page_url:
        soup = get_soup(page_url)
        books.extend(
            parse_book(pod, name) for pod in soup.select("article.product_pod")
        )

        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else None
        time.sleep(REQUEST_DELAY)

    print(f"  {name}: {len(books)} books")
    return books


def main():
    category_urls = find_category_urls(CATEGORIES)

    print(f"Scraping {len(category_urls)} categories from {BASE_URL}")
    all_books = []
    for name, url in category_urls.items():
        all_books.extend(scrape_category(name, url))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["title", "price", "star_rating", "availability", "category"]
        )
        writer.writeheader()
        writer.writerows(all_books)

    print(f"\nWrote {len(all_books)} rows across {len(category_urls)} categories -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
