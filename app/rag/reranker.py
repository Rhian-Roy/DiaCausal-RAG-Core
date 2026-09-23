"""Cross-encoder reranker with abstention gate.

Re-scores candidate chunks from the hybrid retriever using a cross-encoder
model. If the highest-scoring chunk falls below the abstention threshold
TAU, the pipeline returns INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from typing import Any

from app.schemas.contracts import MedicalChunk, RAGEvidencePayload

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TAU = 0.35  # Abstention cutoff threshold
TOP_K_RERANKED = 5

# Cross-encoder model (lazy-loaded)
_CROSS_ENCODER: Any = None
_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> Any:
    """Lazy-load the cross-encoder model."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER

    try:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(_CROSS_ENCODER_MODEL_NAME)
    except Exception:
        _CROSS_ENCODER = None
    return _CROSS_ENCODER


def _score_with_cross_encoder(
    query: str, chunks: list[MedicalChunk]
) -> list[tuple[MedicalChunk, float]]:
    """Score each chunk against the query using the cross-encoder."""
    encoder = _get_cross_encoder()
    if encoder is None:
        # Fallback: use a simple keyword overlap heuristic
        return _fallback_score(query, chunks)

    try:
        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores = encoder.predict(pairs)

        # Normalize scores to [0, 1] using sigmoid if needed
        import numpy as np

        scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=float)))

        scored = list(zip(chunks, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    except Exception:
        return _fallback_score(query, chunks)


def _fallback_score(
    query: str, chunks: list[MedicalChunk]
) -> list[tuple[MedicalChunk, float]]:
    """Keyword-overlap fallback when no cross-encoder is available."""
    query_terms = set(query.lower().split())
    scored: list[tuple[MedicalChunk, float]] = []
    for chunk in chunks:
        chunk_terms = set(chunk.content.lower().split())
        overlap = len(query_terms & chunk_terms)
        # Normalize by query length
        score = overlap / max(len(query_terms), 1)
        scored.append((chunk, min(score, 1.0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    evidence: RAGEvidencePayload,
    tau: float = TAU,
    top_k: int = TOP_K_RERANKED,
) -> RAGEvidencePayload:
    """Re-score and filter the evidence payload using a cross-encoder.

    If the top chunk's score is below *tau*, sets status to
    ``INSUFFICIENT_EVIDENCE`` and returns an empty chunk list.
    """
    if evidence.status != "SUCCESS" or not evidence.chunks:
        return evidence

    scored = _score_with_cross_encoder(evidence.query, evidence.chunks)

    if not scored:
        return RAGEvidencePayload(
            query=evidence.query,
            status="INSUFFICIENT_EVIDENCE",
            chunks=[],
            retrieval_latency_ms=evidence.retrieval_latency_ms,
        )

    # Abstention gate
    top_score = scored[0][1]
    if top_score < tau:
        return RAGEvidencePayload(
            query=evidence.query,
            status="INSUFFICIENT_EVIDENCE",
            chunks=[],
            retrieval_latency_ms=evidence.retrieval_latency_ms,
        )

    # Take top-k reranked chunks
    reranked_chunks = [chunk for chunk, _ in scored[:top_k]]

    return RAGEvidencePayload(
        query=evidence.query,
        status="SUCCESS",
        chunks=reranked_chunks,
        retrieval_latency_ms=evidence.retrieval_latency_ms,
    )
