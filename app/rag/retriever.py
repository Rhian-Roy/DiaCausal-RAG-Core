"""Hybrid retrieval engine combining dense (BGE-M3 / MiniLM) and sparse (BM25s) search.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both
retrieval strategies into a single candidate set.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Ensure single-threaded CPU execution to prevent OpenMP/tqdm worker segfaults on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import chromadb
import numpy as np

from app.rag.ingest import ingest_directory
from app.schemas.contracts import MedicalChunk, RAGEvidencePayload

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma_db")
COLLECTION_NAME = "clinical_guidelines"
RRF_K = 60
TOP_DENSE = 20
TOP_SPARSE = 20
TOP_FUSED = 15

# Dense model selection: defaults to CPU-friendly MiniLM; set
# DIACAUSAL_DENSE_MODEL=BAAI/bge-m3 for higher-quality medical retrieval.
import os as _os

_DENSE_MODEL_NAME: str | None = None
_EMBEDDING_FN: Any = None
_BM25_INDEX: Any = None
_CHUNK_STORE: dict[str, MedicalChunk] = {}
_CHROMA_COLLECTION: Any = None

# Default to lightweight model; BGE-M3 can be opted in via env var
_DEFAULT_DENSE_MODEL = _os.environ.get("DIACAUSAL_DENSE_MODEL", "all-MiniLM-L6-v2")


def _get_dense_model_name() -> str:
    """Select the dense embedding model (configurable via DIACAUSAL_DENSE_MODEL)."""
    global _DENSE_MODEL_NAME
    if _DENSE_MODEL_NAME is not None:
        return _DENSE_MODEL_NAME
    _DENSE_MODEL_NAME = _DEFAULT_DENSE_MODEL
    return _DENSE_MODEL_NAME


def _get_embedding_function() -> Any:
    """Lazy-load the sentence-transformer embedding function for ChromaDB."""
    global _EMBEDDING_FN
    if _EMBEDDING_FN is not None:
        return _EMBEDDING_FN

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    model_name = _get_dense_model_name()
    _EMBEDDING_FN = SentenceTransformerEmbeddingFunction(model_name=model_name)
    return _EMBEDDING_FN


# ---------------------------------------------------------------------------
# Index Building
# ---------------------------------------------------------------------------

def build_index(chunks: list[MedicalChunk] | None = None, force_rebuild: bool = False) -> None:
    """Build or refresh both the ChromaDB dense index and the BM25s sparse index."""
    global _BM25_INDEX, _CHUNK_STORE, _CHROMA_COLLECTION

    if chunks is None:
        chunks = ingest_directory()

    if not chunks:
        raise ValueError("No chunks available to index. Check data/raw/ directory.")

    # Store chunks for later retrieval
    _CHUNK_STORE = {c.chunk_id: c for c in chunks}
    texts = [c.content for c in chunks]
    ids = [c.chunk_id for c in chunks]

    # --- ChromaDB Dense Index ---
    ef = _get_embedding_function()
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
    except (AttributeError, ValueError, TypeError, Exception):
        client = chromadb.Client(
            chromadb.config.Settings(
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
            )
        )

    # Delete existing collection if force rebuild
    try:
        if force_rebuild:
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _CHROMA_COLLECTION = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Only upsert if collection is empty or rebuilding
    if force_rebuild or _CHROMA_COLLECTION.count() == 0:
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metas = [
                {
                    "doc_title": chunks[j].doc_title,
                    "section_path": chunks[j].section_path,
                    "page_number": chunks[j].page_number,
                    "authority_tier": chunks[j].authority_tier,
                }
                for j in range(i, min(i + batch_size, len(texts)))
            ]
            _CHROMA_COLLECTION.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
            )

    # --- BM25s Sparse Index ---
    import bm25s

    corpus_tokens = bm25s.tokenize(texts, stemmer=None)
    _BM25_INDEX = bm25s.BM25()
    _BM25_INDEX.index(corpus_tokens)


def _ensure_index() -> None:
    """Ensure indexes are built, building them on first call."""
    if _BM25_INDEX is None or _CHROMA_COLLECTION is None:
        build_index()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _dense_search(query: str, top_k: int = TOP_DENSE) -> list[tuple[str, float]]:
    """Query ChromaDB for the top-k dense results. Returns (chunk_id, distance)."""
    _ensure_index()
    count = _CHROMA_COLLECTION.count()
    if count == 0:
        return []

    results = _CHROMA_COLLECTION.query(
        query_texts=[query],
        n_results=min(top_k, count),
    )
    ids = results["ids"][0] if results.get("ids") and len(results["ids"]) > 0 else []
    distances = (
        results["distances"][0]
        if results.get("distances") and len(results["distances"]) > 0
        else [0.0] * len(ids)
    )
    return list(zip(ids, distances))


def _sparse_search(query: str, top_k: int = TOP_SPARSE) -> list[tuple[str, float]]:
    """Query BM25s for the top-k sparse results. Returns (chunk_id, score)."""
    _ensure_index()
    if not _CHUNK_STORE or _BM25_INDEX is None:
        return []

    import bm25s

    query_tokens = bm25s.tokenize([query], stemmer=None)
    chunk_ids = list(_CHUNK_STORE.keys())
    effective_top_k = min(top_k, len(chunk_ids))
    if effective_top_k == 0:
        return []

    results, scores = _BM25_INDEX.retrieve(query_tokens, k=effective_top_k, corpus=chunk_ids)
    return [(str(results[0][i]), float(scores[0][i])) for i in range(len(results[0]))]


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = RRF_K,
    top_n: int = TOP_FUSED,
) -> list[tuple[str, float]]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF).

    score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
    """
    rrf_scores: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def hybrid_retrieve(query: str, top_n: int = TOP_FUSED) -> RAGEvidencePayload:
    """Run hybrid dense+sparse retrieval with RRF fusion.

    Returns an *RAGEvidencePayload* (status may be INSUFFICIENT_EVIDENCE
    if no chunks are found).
    """
    start_ms = time.time() * 1000

    dense_hits = _dense_search(query)
    sparse_hits = _sparse_search(query)

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=top_n)

    chunks: list[MedicalChunk] = []
    for doc_id, score in fused:
        chunk = _CHUNK_STORE.get(doc_id)
        if chunk is not None:
            chunks.append(chunk)

    elapsed_ms = time.time() * 1000 - start_ms

    if not chunks:
        return RAGEvidencePayload(
            query=query,
            status="INSUFFICIENT_EVIDENCE",
            chunks=[],
            retrieval_latency_ms=round(elapsed_ms, 2),
        )

    return RAGEvidencePayload(
        query=query,
        status="SUCCESS",
        chunks=chunks,
        retrieval_latency_ms=round(elapsed_ms, 2),
    )
