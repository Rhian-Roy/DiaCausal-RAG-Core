"""Tests for the hybrid retrieval pipeline.

Verifies:
- Synthetic guideline seeding populates data/raw/.
- Ingestion produces well-formed MedicalChunk instances.
- Hybrid search for "sitagliptin renal dose" returns JANUVIA section 2.2
  with renal dose adjustments (100/50/25 mg).
- RRF fusion produces correctly merged scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.ingest import ingest_directory, seed_synthetic_guidelines
from app.schemas.contracts import MedicalChunk


# =========================================================================
# Ingestion Tests
# =========================================================================


class TestIngestion:
    """Tests for the document ingestion pipeline."""

    def test_seed_synthetic_guidelines(self, tmp_path: Path) -> None:
        """Verify synthetic guidelines are written correctly."""
        paths = seed_synthetic_guidelines(tmp_path)
        assert len(paths) == 5
        for p in paths:
            assert p.exists()
            assert p.suffix == ".md"
            content = p.read_text()
            assert len(content) > 100

    def test_ingest_produces_chunks(self, tmp_path: Path) -> None:
        """Verify ingestion produces well-formed MedicalChunk instances."""
        seed_synthetic_guidelines(tmp_path)
        chunks = ingest_directory(tmp_path)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, MedicalChunk)
            assert len(chunk.content) > 0
            assert chunk.chunk_id
            assert chunk.doc_title
            assert 1 <= chunk.authority_tier <= 5

    def test_januvia_chunks_present(self, tmp_path: Path) -> None:
        """Verify JANUVIA label chunks are ingested with correct metadata."""
        seed_synthetic_guidelines(tmp_path)
        chunks = ingest_directory(tmp_path)
        januvia_chunks = [
            c for c in chunks
            if "JANUVIA" in c.doc_title or "Sitagliptin" in c.doc_title
        ]
        assert len(januvia_chunks) > 0
        assert any("2.2" in c.section_path for c in januvia_chunks)


# =========================================================================
# Hybrid Retrieval Tests
# =========================================================================


class TestHybridRetrieval:
    """Tests for the hybrid dense+sparse retrieval with RRF fusion."""

    @pytest.fixture(autouse=True)
    def _setup_index(self, tmp_path: Path) -> None:
        """Build a fresh index from synthetic guidelines."""
        from app.rag import retriever

        seed_synthetic_guidelines(tmp_path)
        chunks = ingest_directory(tmp_path)
        # Reset global state
        retriever._BM25_INDEX = None
        retriever._CHROMA_COLLECTION = None
        retriever._CHUNK_STORE = {}
        retriever.CHROMA_PERSIST_DIR = str(tmp_path / "chroma_test")
        retriever.build_index(chunks, force_rebuild=True)

    def test_sitagliptin_renal_dose_retrieval(self) -> None:
        """Verify hybrid search for sitagliptin renal dose returns JANUVIA §2.2."""
        from app.rag.retriever import hybrid_retrieve

        result = hybrid_retrieve("sitagliptin renal dose adjustment")
        assert result.status == "SUCCESS"
        assert len(result.chunks) > 0

        # At least one chunk should be from JANUVIA label section 2.2
        januvia_renal_chunks = [
            c for c in result.chunks
            if ("JANUVIA" in c.doc_title or "Sitagliptin" in c.doc_title)
            and "2.2" in c.section_path
        ]
        assert len(januvia_renal_chunks) > 0, (
            f"Expected JANUVIA §2.2 chunk. Got docs: "
            f"{[(c.doc_title, c.section_path) for c in result.chunks]}"
        )

        # Verify dose information is present in the content
        combined_content = " ".join(c.content for c in januvia_renal_chunks)
        assert "100" in combined_content, "Missing 100 mg dose info"
        assert "50" in combined_content, "Missing 50 mg dose info"
        assert "25" in combined_content, "Missing 25 mg dose info"

    def test_dapagliflozin_egfr_retrieval(self) -> None:
        """Verify search for dapagliflozin eGFR returns FARXIGA label."""
        from app.rag.retriever import hybrid_retrieve

        result = hybrid_retrieve("dapagliflozin eGFR renal impairment")
        assert result.status == "SUCCESS"
        assert any(
            "FARXIGA" in c.doc_title or "Dapagliflozin" in c.doc_title
            for c in result.chunks
        )

    def test_retrieval_latency_recorded(self) -> None:
        """Verify retrieval latency is tracked."""
        from app.rag.retriever import hybrid_retrieve

        result = hybrid_retrieve("metformin first line therapy")
        assert result.retrieval_latency_ms > 0


# =========================================================================
# RRF Fusion Tests
# =========================================================================


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion score computation."""

    def test_rrf_basic_fusion(self) -> None:
        from app.rag.retriever import reciprocal_rank_fusion

        dense = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.7)]
        sparse = [("doc_b", 5.0), ("doc_a", 4.0), ("doc_d", 3.0)]

        fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=3)

        ids = [doc_id for doc_id, _ in fused]
        # doc_a and doc_b appear in both lists so should rank highest
        assert "doc_a" in ids[:2]
        assert "doc_b" in ids[:2]

    def test_rrf_scores_positive(self) -> None:
        from app.rag.retriever import reciprocal_rank_fusion

        dense = [("x", 0.5)]
        sparse = [("x", 1.0)]

        fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=1)
        assert fused[0][1] > 0
