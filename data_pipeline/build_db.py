"""Load books_clean.csv into a normalized two-table SQLite database.

Schema (PK/FK relationship required by the brief):

    categories(category_id INTEGER PRIMARY KEY,
               category_name TEXT UNIQUE NOT NULL)

    books(book_id     INTEGER PRIMARY KEY,
          title       TEXT NOT NULL,
          price_gbp   REAL,
          price_inr   REAL,
          rating      INTEGER,
          in_stock    INTEGER,
          category_id INTEGER REFERENCES categories(category_id))

The database is dropped and rebuilt on every run, so books.db is always exactly
reproducible from books_clean.csv.
"""

import pathlib
import sqlite3

import pandas as pd

INPUT_CSV = "books_clean.csv"
DB_PATH = "books.db"

SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL,
    price_inr   REAL,
    rating      INTEGER,
    in_stock    INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);
"""


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    print(f"Read {len(df)} cleaned rows from {INPUT_CSV}")

    # Start from a genuinely fresh file. Dropping tables inside an existing
    # database leaves freed pages behind, so the resulting file differs
    # byte-for-byte between runs even when its contents are identical -- which
    # shows up as a spurious diff on every rebuild of the committed books.db.
    pathlib.Path(DB_PATH).unlink(missing_ok=True)

    connection = sqlite3.connect(DB_PATH)
    # SQLite does not enforce foreign keys unless this pragma is set per connection.
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)

    # One row per distinct category; category_id is assigned by SQLite.
    categories = sorted(df["category"].unique())
    connection.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(name,) for name in categories],
    )

    category_ids = dict(
        connection.execute("SELECT category_name, category_id FROM categories")
    )

    connection.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(row.in_stock),  # SQLite has no BOOLEAN type; stored as 0/1
                category_ids[row.category],
            )
            for row in df.itertuples(index=False)
        ],
    )
    connection.commit()

    book_count = connection.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    category_count = connection.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    orphans = connection.execute(
        """SELECT COUNT(*) FROM books b
           LEFT JOIN categories c ON b.category_id = c.category_id
           WHERE c.category_id IS NULL"""
    ).fetchone()[0]

    print(f"Loaded {category_count} categories and {book_count} books -> {DB_PATH}")
    print(f"Foreign-key integrity check: {orphans} orphaned book rows")

    connection.close()


if __name__ == "__main__":
    main()
