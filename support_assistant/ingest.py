"""Ingestion + embedding: docs/*.txt -> all-MiniLM-L6-v2 -> ChromaDB.

Runs entirely locally. No API key, no account, no network call to any LLM
provider -- sentence-transformers downloads the MiniLM weights once on first
use and caches them.

Chunking is one chunk per document. The brief allows this explicitly, and at
330-560 characters each the policy documents sit well inside MiniLM's 256-token
window, so splitting them would only scatter a single coherent policy across
several vectors and make retrieval worse.
"""

import pathlib

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = pathlib.Path(__file__).parent / "docs"
CHROMA_DIR = str(pathlib.Path(__file__).parent / ".chroma")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Load MiniLM once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_chunks():
    """Read the corpus. Returns (ids, texts) with ids like 'doc_01'."""
    paths = sorted(DOCS_DIR.glob("doc_*.txt"))
    if not paths:
        raise SystemExit(f"No corpus documents found in {DOCS_DIR}")
    return (
        [p.stem for p in paths],
        [p.read_text(encoding="utf-8").strip() for p in paths],
    )


def get_collection():
    """Open (or create) the ChromaDB collection, configured for cosine similarity."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Chroma defaults to squared L2; the brief asks for cosine similarity.
        metadata={"hnsw:space": "cosine"},
    )


def build_index():
    """Embed every document and upsert it into the collection."""
    ids, texts = load_chunks()
    embeddings = get_model().encode(texts).tolist()

    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": f"{doc_id}.txt"} for doc_id in ids],
    )
    return collection, ids


def main():
    collection, ids = build_index()
    print(f"Embedded {len(ids)} documents with {EMBEDDING_MODEL} -> "
          f"collection '{COLLECTION_NAME}' ({collection.count()} vectors)")
    for doc_id in ids:
        print(f"  {doc_id}")

    # Smoke check: a delivery question must retrieve the delivery policy.
    probe = "how long does delivery take and is it free"
    result = collection.query(
        query_embeddings=get_model().encode([probe]).tolist(), n_results=1
    )
    top_id = result["ids"][0][0]
    print(f"\nRetrieval check - {probe!r} -> {top_id}")
    assert top_id == "doc_01", f"expected doc_01 (delivery policy), got {top_id}"
    print("OK")


if __name__ == "__main__":
    main()
