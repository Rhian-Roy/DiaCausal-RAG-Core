#!/usr/bin/env python3
"""DiaCausal-RAG-Core — Clinical Copilot Launch Script.

Run with:
    streamlit run diacausal_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.bridge.causal_connector import estimate_all_treatments
from app.rag.fusion import fuse
from app.rag.ingest import ingest_directory, seed_synthetic_guidelines
from app.rag.reranker import rerank
from app.rag.retriever import build_index, hybrid_retrieve
from app.schemas.contracts import PatientContext, UnifiedCDSSResponse
from app.security.guardrails import run_all_guards
from app.ui.dashboard import render_dashboard


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

@st.cache_resource
def _initialize_pipeline() -> bool:
    """Seed guidelines and build retrieval indexes (runs once)."""
    seed_synthetic_guidelines()
    chunks = ingest_directory()
    build_index(chunks)
    return True


def run_pipeline(
    patient: PatientContext,
) -> tuple[UnifiedCDSSResponse, dict[str, float]]:
    """Execute the full CDSS pipeline and return response + stage timings."""
    stage_times: dict[str, float] = {}

    # 1. Input Guard
    t0 = time.time()
    guard_result = run_all_guards(
        query=f"Treatment options for patient with HbA1c {patient.hba1c}%, eGFR {patient.egfr}",
        hba1c=patient.hba1c,
        egfr=patient.egfr,
        age=patient.age,
    )
    stage_times["Input Guard"] = (time.time() - t0) * 1000

    if not guard_result.passed:
        from app.schemas.contracts import RAGEvidencePayload
        empty_rag = RAGEvidencePayload(
            query="",
            status="OUT_OF_SCOPE" if guard_result.status == "OUT_OF_SCOPE" else "INSUFFICIENT_EVIDENCE",
            chunks=[],
            retrieval_latency_ms=0.0,
        )
        return (
            UnifiedCDSSResponse(
                patient=patient,
                eligible_options=[],
                flagged_options={},
                causal_estimates=[],
                rag_evidence=empty_rag,
                synthesized_advice=f"Pipeline halted: {guard_result.message}",
                safety_warnings=[guard_result.message],
            ),
            stage_times,
        )

    # 2. Parallel RAG & Causal Execution
    from concurrent.futures import ThreadPoolExecutor

    t1 = time.time()
    query = (
        f"Type 2 diabetes treatment options for patient with "
        f"HbA1c {patient.hba1c}%, eGFR {patient.egfr} mL/min/1.73m², "
        f"BMI {patient.bmi}, age {patient.age}"
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rag_future = executor.submit(hybrid_retrieve, query)
        causal_future = executor.submit(estimate_all_treatments, patient)
        rag_evidence = rag_future.result()
        causal_estimates = causal_future.result()

    stage_times["Parallel RAG & Causal Execution"] = (time.time() - t1) * 1000

    # 3. RRF Fusion (already done inside hybrid_retrieve)
    stage_times["RRF Fusion"] = 0  # Included in step 2

    # 4. Reranker
    t3 = time.time()
    rag_evidence = rerank(rag_evidence)
    stage_times["Reranker"] = (time.time() - t3) * 1000

    # 5. Fusion Safety Gate
    t4 = time.time()
    response = fuse(patient, rag_evidence, causal_estimates)
    stage_times["Fusion Safety Gate"] = (time.time() - t4) * 1000

    # 6. Final Synthesis
    stage_times["Final Synthesis"] = 0  # Included in fusion

    return response, stage_times


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    _initialize_pipeline()
    render_dashboard(run_pipeline_fn=run_pipeline)


if __name__ == "__main__":
    main()
