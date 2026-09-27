# Module 3 — Support Assistant

A grounded GenAI service over Zepto's own policy documents: ingestion →
embedding → retrieval → generation, orchestrated by LangGraph and served by
FastAPI.

**Everything here runs fully offline.** No signup, no API key, no network call
to any LLM provider. `MOCK_LLM` left at its default is the graded path.

## Run it

```bash
cd support_assistant
python ingest.py                                   # embed the corpus into ChromaDB (first run downloads MiniLM)
python graph.py                                    # self-check: one query per routing branch
uvicorn main:app --reload --port 7860              # serve POST /ask
```

`ingest.py` must run once before the service starts, to create the ChromaDB
collection. The `.chroma/` directory is gitignored and rebuilt from the
committed `docs/` corpus.

## Example calls

Both recorded with `MOCK_LLM` left at its default — no LLM call was made.

**1 — routed to `retrieve_and_answer`** (contains "refund", a policy keyword):

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your refund policy for damaged items?"}'
```

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
```

Retrieval landed on the right documents: `doc_02` is Returns & Refunds and
`doc_06` is Damaged or Missing Items — exactly what the question asked about.

**2 — routed to `direct_answer`** (no policy keyword):

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

`sources` is empty and no retrieval ran, as the brief specifies for
`general_question`.

**Health check** — `GET /` also reports which mode is active:

```json
{
  "status": "ok",
  "mock_llm": true,
  "mode": "mock (deterministic, no LLM calls)",
  "MOCK_LLM": null
}
```

## Architecture

```
docs/doc_01..08.txt
        │
        │  ingest.py :: load_chunks()          ── INGESTION
        ▼
   8 text chunks (one per document)
        │
        │  ingest.py :: build_index()          ── EMBEDDING
        │  SentenceTransformer("all-MiniLM-L6-v2").encode()
        ▼
   ChromaDB collection "zepto_policies"  (.chroma/, cosine similarity)
        │
        ▼
POST /ask  ──►  main.py :: ask_endpoint()
                      │
                      ▼
              graph.py :: compiled StateGraph
                      │
              ┌───────┴────────┐
              │ classify_intent│  keyword heuristic over 8 policy keywords
              └───────┬────────┘
            conditional edge on state["intent"]
              ┌───────┴──────────────────┐
              ▼                          ▼
   retrieve_and_answer            direct_answer      ── RETRIEVAL + GENERATION
   top-3 cosine query             fixed canned string
   + templated answer             sources = []
              └───────┬──────────────────┘
                      ▼
          schema.py :: AnswerResponse   (Pydantic-validated)
                      ▼
              JSON {answer, sources, confidence}
```

### Stage by stage

| Stage | Handled by | What happens |
|---|---|---|
| **Ingestion** | `ingest.py :: load_chunks()` | Reads `docs/doc_*.txt`, one chunk per document, id = filename stem (`doc_01`) |
| **Embedding** | `ingest.py :: build_index()` | `SentenceTransformer("all-MiniLM-L6-v2").encode()`, upserted into the ChromaDB collection `zepto_policies` persisted at `.chroma/` |
| **Retrieval** | `graph.py :: retrieve_and_answer()` | Embeds the query with the same model, queries the collection for the top 3 chunks by cosine similarity |
| **Generation** | `graph.py :: retrieve_and_answer()` / `direct_answer()` | Mock: canned templates. Real-LLM: `prompt.py :: ANSWER_PROMPT` sent via `schema.py :: call_llm()` |
| **Validation** | `schema.py :: AnswerResponse` | Every response is a Pydantic-validated `{answer, sources, confidence}` |

Data flows as a `State` TypedDict — `query` in, `intent` added by node 1, then
`chunks` and `response` added by node 2 or 3. `main.py` returns `state["response"]`.

### What `MOCK_LLM` changes

**Only the generation step inside each node branches on it.** Routing never
does, and retrieval always runs for real — embedding and ChromaDB need no API
key and no network call.

| Stage | Default (`MOCK_LLM` unset or `1`) — graded | `MOCK_LLM=0` — optional, not attempted |
|---|---|---|
| `classify_intent` | Keyword heuristic over `delivery, return, refund, membership, tracking, cancel, gift card, support hours`. No LLM call. | LLM classifies via `CLASSIFY_PROMPT` |
| Retrieval | **Runs for real** | **Runs for real** — unchanged |
| `retrieve_and_answer` generation | `f"Based on the retrieved context: {top_chunk[:200]}"` | LLM prompted with `ANSWER_PROMPT`, grounded only in retrieved chunks |
| `direct_answer` generation | Fixed canned string | LLM prompted directly, no retrieval |
| Schema population | Deterministic in code: `sources` = retrieved ids, `confidence` = 1.0 | Validated via `parse_or_retry`, up to 2 corrective retries |

The `MOCK_LLM=0` path is implemented but was **not attempted** — it is an
explicitly ungraded extension. `schema.py :: parse_or_retry()` contains the
retry-on-validation-failure logic regardless, and its self-check exercises the
degradation path without needing an LLM.

## Docker

```bash
cd support_assistant
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
# then POST http://127.0.0.1:7860/ask as above
```

The image runs `python ingest.py` at **build** time, so the MiniLM weights and
the Chroma index are baked in and the container needs no network at runtime.
`MOCK_LLM=1` is set explicitly in the image.

> ⚠️ **Not build-verified.** Docker is not installed on the machine this was
> developed on, so `docker build` has not been executed. The Dockerfile is
> written against the same commands used successfully outside the container,
> but it has not been proven to build.

The optional Hugging Face Spaces deployment was **not attempted** — explicitly
ungraded.

## Design decisions

### One chunk per document

The brief allows it, and these documents are 330–560 characters — comfortably
inside MiniLM's 256-token window. Splitting them would scatter a single coherent
policy across several vectors and make retrieval worse, not better.

### Cosine similarity must be requested explicitly

ChromaDB defaults to squared L2 distance. The collection is created with
`metadata={"hnsw:space": "cosine"}` because the brief specifies cosine.

### The corpus is extracted, not retyped

`docs/doc_*.txt` were generated by parsing `docs/brief.md` programmatically, so
the corpus is provably the brief's exact text with no transcription drift.

### A separate `requirements.txt` for the image

The repository root holds the consolidated local-development list across all
three modules. The container has no reason to carry pandas, seaborn,
scikit-learn or BeautifulSoup, so `support_assistant/requirements.txt` is the
smaller runtime subset — and it installs CPU-only torch rather than the
multi-GB CUDA wheel.

### Self-checks over a test framework

`schema.py` and `graph.py` each run assertions under `__main__`: both routing
branches behave as specified, retrieval reaches the correct source document,
and the retry loop degrades to a marked error response rather than raising.
