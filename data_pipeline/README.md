# Module 1 — Data Pipeline

Scrape → clean → convert → store → query. Raw HTML from a public practice site
becomes a normalized two-table SQLite database that is then queried with both
SQL and pandas.

## Run it

From the repository root, with the virtual environment active:

```bash
cd data_pipeline
python scrape.py        # books.toscrape.com -> raw_books.csv       (needs network)
python clean.py         # raw_books.csv      -> books_clean.csv
python build_db.py      # books_clean.csv    -> books.db
python run_queries.py   # books.db           -> queries_output.md
python pandas_check.py  # books.db           -> pandas_check_output.md
```

The scripts are ordered and each one's output is committed, so any single stage
can be re-run on its own. `scrape.py` is the only stage needing network access.

## Scope of the scraped dataset

**99 books across 5 categories** — comfortably over the brief's minimum of 60
books across 3 categories.

| Category | Books |
|---|---|
| Mystery | 32 |
| Historical Fiction | 26 |
| Classics | 19 |
| Travel | 11 |
| Philosophy | 11 |

Pagination is followed automatically via the `li.next` link, so categories
spanning more than one listing page (Mystery, Historical Fiction) are captured
in full rather than truncated at the first 20 results.

## Currency conversion

```
price_inr = price_gbp × 105.50
```

**1 GBP = 105.50 INR.** This is the fixed, project-defined baseline constant
specified by the brief — an artificial rate for this assignment, not a live or
historical market rate. It therefore requires no API call, no network access
and no date reference. The optional live-API stretch was not attempted, so
`price_inr` is computed entirely from this constant.

## Design decisions

### Full titles come from the `title` attribute

On listing pages the anchor text is truncated (`"A Light in the ..."`), while
the `title` attribute carries the complete string. The scraper reads the
attribute, so no title is silently cut short.

### Forcing UTF-8 on the response

books.toscrape.com serves UTF-8 but sends no `charset` in its `Content-Type`
header. `requests` therefore falls back to ISO-8859-1 and every price decodes as
`Â£45.17` instead of `£45.17`. `scrape.py` sets `response.encoding = "utf-8"`
before parsing, which fixes the price strings at the source rather than patching
mojibake downstream.

### Messy-row handling — imputation vs. dropping

The three parser functions return `None` rather than raising, and the policy
then differs by field type:

| Field | Unparseable value | Why |
|---|---|---|
| `price_gbp` | **median imputation** | Numeric. The median is robust to the outliers a price column naturally has, and keeps the row's title/rating/category information rather than discarding it over one bad cell. |
| `rating` | **median imputation** | Numeric and ordinal (1–5), so the same argument applies. |
| `in_stock` | **drop the row** | Boolean — there is no median to impute. Defaulting an unknown stock status to either `True` or `False` would fabricate a business fact, so the row is dropped instead. |

**On this run, all 99 rows parsed cleanly** — 0 imputations, 0 drops. Because
the failure path therefore never fires against live data, `clean.py` ends with a
short block calling each parser on deliberately malformed input
(`"price on request"`, `"Eleven"`, `"???"`), demonstrating that all three
degrade to `None` instead of crashing the pipeline.

### `in_stock` is uniformly `True`

Every book on books.toscrape.com's listing pages shows `"In stock"`; the site
has no out-of-stock items at listing level. `parse_in_stock` still handles
`"Out of stock"` and `"unavailable"`, but on this dataset the resulting column
is constant. This is a property of the source data, not a parsing failure.

### Schema

```sql
categories(category_id   INTEGER PRIMARY KEY,
           category_name TEXT UNIQUE NOT NULL)

books(book_id     INTEGER PRIMARY KEY,
      title       TEXT NOT NULL,
      price_gbp   REAL,
      price_inr   REAL,
      rating      INTEGER,
      in_stock    INTEGER,
      category_id INTEGER REFERENCES categories(category_id))
```

Category names are stored once in `categories` and referenced by foreign key,
rather than repeated on every book row. `PRAGMA foreign_keys = ON` is set on the
connection, since SQLite does not enforce foreign keys by default. `in_stock` is
stored as `INTEGER` 0/1 — SQLite has no native boolean type — and is read back
as a proper bool in pandas.

`build_db.py` drops and recreates both tables on every run, so `books.db` is
exactly reproducible from `books_clean.csv`. Both the database file and the
script that regenerates it are committed.

## SQL coverage

Six queries in `run_queries.py`, with full output in
[`queries_output.md`](queries_output.md):

| Query | Clause demonstrated | Rows |
|---|---|---|
| Q1 | `SELECT` / `WHERE` | 35 |
| Q2 | `ORDER BY` + `LIMIT` | 10 |
| Q3 | `DISTINCT` | 5 |
| Q4 | `BETWEEN` | 28 |
| Q5 | `IN` | 22 |
| Q6 | `JOIN` across both tables | 10 |

The brief asks for `IN` *or* `BETWEEN`; both are included.

## pandas equivalence

`pandas_check.py` (output in [`pandas_check_output.md`](pandas_check_output.md)):

- Reads three query results back with `pd.read_sql(...)` — the brief asks for at
  least two.
- Reproduces the Q6 join with `pd.merge(...)`. The two tables are loaded with
  plain `SELECT * FROM <table>`, and the join, filter, sort and limit are then
  all performed by pandas — no SQL `JOIN` is involved.
- Prints both results side by side and asserts equivalence with
  `pd.testing.assert_frame_equal`, which raises on any difference in value,
  ordering or dtype. **All 10 rows and dtypes match.**

The sort uses `kind="mergesort"` so that ties break in the same stable order
SQLite produces; the query's `ORDER BY` also includes `title` as an explicit
tiebreaker, making both results deterministic rather than coincidentally equal.
