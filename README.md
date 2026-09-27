# Zepto Data & AI Platform

Capstone project — Certificate Program in Artificial Intelligence and Machine
Learning. One repository, three internally-linked modules:

| Module | Marks | What it does | Status |
|---|---|---|---|
| [`/data_pipeline`](data_pipeline/) | 25 | Scrapes catalogue data, cleans it, converts currency, loads it into a normalized SQLite database, and queries it with both SQL and pandas | ✅ Complete |
| [`/analytics`](analytics/) | 50 | Profiles the Titanic dataset end to end, then builds and evaluates a full predictive-modeling pipeline on it | ✅ Complete |
| [`/support_assistant`](support_assistant/) | 25 | A grounded GenAI support assistant: document corpus → embeddings → ChromaDB → LangGraph router → FastAPI service | ✅ Complete |

The original assignment brief is preserved at [`docs/brief.md`](docs/brief.md).

## Setup

Dependencies are managed as **one consolidated `requirements.txt`** at the
repository root, rather than one file per module. Installing it covers all
three modules.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Developed against Python 3.12.

> `support_assistant/requirements.txt` also exists, but it is **not** a
> per-module dependency file — it is the smaller runtime subset the Docker
> image installs, so the container does not ship pandas, seaborn,
> scikit-learn or BeautifulSoup. For local development you only ever need
> the root file above.

## Running each module

### Module 1 — `/data_pipeline`

```bash
cd data_pipeline
python scrape.py        # books.toscrape.com -> raw_books.csv       (needs network)
python clean.py         # raw_books.csv      -> books_clean.csv
python build_db.py      # books_clean.csv    -> books.db
python run_queries.py   # books.db           -> queries_output.md
python pandas_check.py  # books.db           -> pandas_check_output.md
```

Every intermediate output is committed, so each stage can be re-run
independently. Only `scrape.py` needs network access.

### Module 2 — `/analytics`

```bash
cd analytics
python 01_eda.py        # sns.load_dataset -> titanic.csv, charts/, eda_output.md   (needs network once)
python 02_modeling.py   # titanic.csv -> charts/, modeling_output.md, best_pipeline.joblib
```

`01_eda.py` performs the module's single raw dataset load and snapshots it to
the committed `titanic.csv`; `02_modeling.py` reads that file and never reloads
from seaborn. The module is therefore gradeable offline.

### Module 3 — `/support_assistant`

```bash
cd support_assistant
python ingest.py                         # embed the corpus into ChromaDB (first run downloads MiniLM)
python graph.py                          # self-check: one query per routing branch
uvicorn main:app --reload --port 7860    # serve POST /ask
```

Runs fully offline — no signup, no API key, no LLM provider call. `ingest.py`
must run once before the service starts.

```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query": "What is your refund policy for damaged items?"}'
```

## Design decisions

### Module 1 — Data Pipeline

Full write-up in [`data_pipeline/README.md`](data_pipeline/README.md). In brief:

- **Scope.** 99 books across 5 categories (Mystery, Historical Fiction,
  Classics, Travel, Philosophy), exceeding the required 60 books / 3 categories.
  Pagination is followed automatically, so multi-page categories are captured in
  full.
- **Currency.** `price_inr = price_gbp × 105.50`. **1 GBP = 105.50 INR** is the
  fixed project-defined baseline constant from the brief — an artificial rate,
  not a market rate, requiring no API call and no date reference.
- **Encoding.** The source site serves UTF-8 without declaring a charset, so
  `requests` guesses ISO-8859-1 and mangles the pound sign. The scraper forces
  `response.encoding = "utf-8"` before parsing.
- **Messy rows.** Numeric fields (`price_gbp`, `rating`) that fail to parse are
  median-imputed; unparseable `in_stock` values cause the row to be dropped,
  because a boolean has no median and guessing stock status would fabricate a
  fact. All 99 rows parsed cleanly on this run, so `clean.py` demonstrates the
  failure path explicitly against malformed sample inputs.
- **Schema.** Two tables with a PK/FK relationship — category names are stored
  once in `categories` and referenced from `books`, not repeated per row.
  `PRAGMA foreign_keys = ON` is set, since SQLite does not enforce them by
  default.
- **Verification.** The SQL join and its `pd.merge` reproduction are compared
  with `pd.testing.assert_frame_equal`, which raises on any difference in value,
  ordering or dtype.

### Module 2 — Analytics

Full write-up in [`analytics/README.md`](analytics/README.md), with every
required interpretation in [`analytics/eda_output.md`](analytics/eda_output.md)
and [`analytics/modeling_output.md`](analytics/modeling_output.md). In brief:

- **One load.** `sns.load_dataset('titanic')` is called exactly once, in
  `01_eda.py`, which immediately writes the committed `titanic.csv` that the
  modeling stage reads. No second load exists anywhere in the module.
- **Missing values.** `deck` 77.22% → column dropped (its recording is
  non-random and largely proxies "not first class", which `pclass` already
  captures); `age` 19.87% → median imputed; `embarked` 0.22% → 2 rows dropped.
- **Leakage control is structural.** All preprocessing sits in a
  `ColumnTransformer` inside a `Pipeline`, so nothing can fit on the test split.
  SMOTE uses `imblearn.pipeline.Pipeline`, which resamples during `fit` only —
  a plain sklearn Pipeline would resample the test data too.
- **Excluded features.** `alive` is a verbatim copy of the target; `class`,
  `embark_town`, `who`, `adult_male` and `alone` are all derived from columns
  already present.
- **Recommendation.** Random Forest, F1 0.7328 / AUC 0.8267 on a 179-row
  stratified test set. Tuning did not beat the default on this split, and the
  report says so rather than glossing over it. The saved `best_pipeline.joblib`
  is whichever model actually won on F1, verified by reloading and predicting on
  raw rows including one with a missing `age`.
- **Scripts, not notebooks.** The brief states these are equally acceptable;
  this keeps the module consistent with Module 1 and captures every printed
  result in committed Markdown.

### Module 3 — Support Assistant

Full write-up, both example call transcripts and the pipeline architecture
diagram are in [`support_assistant/README.md`](support_assistant/README.md).
In brief:

- **Offline by default.** `MOCK_LLM` unset or `1` is the graded path and makes
  no LLM call. Only the generation step inside each node branches on it —
  routing never does, and retrieval always runs for real, since embedding and
  ChromaDB need no key and no network.
- **The corpus is extracted, not retyped.** `docs/doc_01..08.txt` were
  generated by parsing `docs/brief.md` programmatically, so they are provably
  the brief's exact text.
- **One chunk per document.** Allowed by the brief, and at 330–560 characters
  these sit inside MiniLM's 256-token window; splitting would scatter one
  coherent policy across several vectors.
- **Cosine similarity is explicit.** ChromaDB defaults to squared L2, so the
  collection is created with `metadata={"hnsw:space": "cosine"}`.
- **Separate container requirements.** The image ships only the RAG runtime
  subset and CPU-only torch, not the consolidated local-dev list.
- **Docker is not build-verified** — Docker is not installed on the development
  machine. The Dockerfile is written but has not been executed.
- **Both optional extensions skipped** (real Groq LLM, Hugging Face Spaces) —
  explicitly ungraded. The retry-on-validation-failure logic for the real-LLM
  path is present in `schema.py` regardless.
