"""Clean raw_books.csv into typed columns and write books_clean.csv.

Type conversions
    price   "£45.17"  -> price_gbp  float
    rating  "Three"        -> rating     int 1-5
    avail.  "In stock"     -> in_stock   bool
                           -> price_inr  float (fixed rate, see GBP_TO_INR)

Messy-row policy (see README for the written justification)
    numeric fields (price_gbp, rating) that fail to parse -> median imputation
    in_stock, which is boolean and cannot be median-imputed -> drop the row
"""

import pandas as pd

INPUT_CSV = "raw_books.csv"
OUTPUT_CSV = "books_clean.csv"

# Project-defined fixed baseline conversion rate for this assignment.
# This is an artificial constant, not a live or historical market rate, so it
# needs no API call, no network access and no date reference.
GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(text):
    """'£45.17' -> 45.17. Returns None if there is no parseable number."""
    try:
        return float(str(text).strip().lstrip("£$€").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_rating(text):
    """'Three' -> 3. Returns None for any word outside One..Five."""
    return RATING_WORDS.get(str(text).strip().title())


def parse_in_stock(text):
    """'In stock' -> True, 'Out of stock' -> False. Returns None if unrecognised."""
    normalised = str(text).strip().lower()
    if "in stock" in normalised:
        return True
    if "out of stock" in normalised or "unavailable" in normalised:
        return False
    return None


def median_impute(series, label):
    """Fill unparseable numerics with the column median, reporting what it did."""
    failures = int(series.isna().sum())
    if failures == 0:
        print(f"  {label}: 0 unparseable values")
        return series

    median = series.median()
    print(f"  {label}: {failures} unparseable -> median-imputed with {median}")
    return series.fillna(median)


def clean(raw):
    df = pd.DataFrame(
        {
            "title": raw["title"].astype(str).str.strip(),
            "category": raw["category"].astype(str).str.strip(),
            "price_gbp": raw["price"].map(parse_price),
            "rating": raw["star_rating"].map(parse_rating),
            "in_stock": raw["availability"].map(parse_in_stock),
        }
    )

    print("Messy-value handling:")
    df["price_gbp"] = median_impute(df["price_gbp"], "price_gbp")
    df["rating"] = median_impute(df["rating"], "rating").astype(int)

    # in_stock is boolean: there is no median to impute, and guessing a stock
    # status would fabricate a business fact, so unparseable rows are dropped.
    unparseable_stock = int(df["in_stock"].isna().sum())
    print(f"  in_stock: {unparseable_stock} unparseable -> row(s) dropped")
    df = df.dropna(subset=["in_stock"]).copy()
    df["in_stock"] = df["in_stock"].astype(bool)

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]


def demo_messy_inputs():
    """Show the parsers degrading gracefully instead of crashing.

    The live site data parses 100% cleanly, so this block exists to make the
    failure path visible to a reader without needing to corrupt the real CSV.
    """
    print("\nParser behaviour on deliberately malformed inputs:")
    for value in ["£45.17", "price on request", ""]:
        print(f"  parse_price({value!r}) -> {parse_price(value)}")
    for value in ["Three", "Eleven"]:
        print(f"  parse_rating({value!r}) -> {parse_rating(value)}")
    for value in ["In stock", "Out of stock", "???"]:
        print(f"  parse_in_stock({value!r}) -> {parse_in_stock(value)}")


def main():
    raw = pd.read_csv(INPUT_CSV, encoding="utf-8")
    print(f"Read {len(raw)} raw rows from {INPUT_CSV}\n")

    df = clean(raw)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nCleaned dtypes:\n{df.dtypes.to_string()}")
    print(f"\n{df.head(3).to_string(index=False)}")
    print(f"\nWrote {len(df)} rows across {df['category'].nunique()} categories -> {OUTPUT_CSV}")
    print(f"Conversion applied: 1 GBP = {GBP_TO_INR} INR (fixed project constant)")

    demo_messy_inputs()


if __name__ == "__main__":
    main()
