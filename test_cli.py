#!/usr/bin/env python3
"""Quick CLI test of the full DiaCausal-RAG-Core pipeline.

Run with:  python test_cli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.bridge.causal_connector import estimate_all_treatments
from app.rag.fusion import fuse
from app.rag.ingest import ingest_directory, seed_synthetic_guidelines
from app.rag.reranker import rerank
from app.rag.retriever import build_index, hybrid_retrieve
from app.schemas.contracts import PatientContext
from app.security.guardrails import run_all_guards


def main() -> None:
    # --- 1. Initialize ---
    print("=" * 70)
    print("DiaCausal-RAG-Core — CLI Pipeline Test")
    print("=" * 70)

    print("\n[1/6] Seeding synthetic guidelines...")
    seed_synthetic_guidelines()

    print("[2/6] Ingesting documents & building index...")
    chunks = ingest_directory()
    build_index(chunks)
    print(f"      Indexed {len(chunks)} chunks from {len(set(c.doc_title for c in chunks))} documents.")

    # --- 2. Define patient ---
    patient = PatientContext(
        age=55,
        bmi=27.5,
        egfr=38.0,
        hba1c=8.6,
        prior_hypo=False,
        current_meds=["Metformin"],
    )
    print(f"\n[3/6] Patient: Age {patient.age}, BMI {patient.bmi}, "
          f"eGFR {patient.egfr}, HbA1c {patient.hba1c}%, "
          f"Meds: {patient.current_meds}")

    # --- 3. Guardrails ---
    guard = run_all_guards(
        f"Treatment for T2DM patient HbA1c {patient.hba1c}%",
        hba1c=patient.hba1c, egfr=patient.egfr, age=patient.age,
    )
    print(f"\n[4/6] Guardrails: {guard.status} — {guard.message}")
    if not guard.passed:
        print("      ❌ Pipeline halted by guardrails.")
        return

    # --- 4. RAG Retrieval + Reranking ---
    query = (f"Type 2 diabetes treatment eGFR {patient.egfr} HbA1c {patient.hba1c}")
    print(f"\n[5/6] Running hybrid retrieval for: '{query}'")
    rag_evidence = hybrid_retrieve(query)
    rag_evidence = rerank(rag_evidence)
    print(f"      Status: {rag_evidence.status} | "
          f"Chunks: {len(rag_evidence.chunks)} | "
          f"Latency: {rag_evidence.retrieval_latency_ms:.0f}ms")
    for i, c in enumerate(rag_evidence.chunks[:3], 1):
        print(f"      [{i}] {c.doc_title} — §{c.section_path} (p.{c.page_number})")

    # --- 5. Causal Estimation ---
    print("\n[6/6] Running causal estimation (LinearDML)...")
    causal_estimates = estimate_all_treatments(patient)

    # --- 6. Fusion ---
    response = fuse(patient, rag_evidence, causal_estimates)

    # --- Print Results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n✅ Eligible: {response.eligible_options}")
    print(f"🚫 Flagged:  {list(response.flagged_options.keys())}")

    print("\n--- Causal Estimates ---")
    for est in response.causal_estimates:
        flag = "🚫" if est.drug_name in response.flagged_options else "✅"
        print(f"  {flag} {est.drug_name}: "
              f"CATE {est.cate_hba1c_delta:+.2f}% "
              f"[95% CI: {est.ci_lower:+.2f}, {est.ci_upper:+.2f}%] "
              f"({est.overlap_status})")

    if response.flagged_options:
        print("\n--- Flagged Details ---")
        for drug, reason in response.flagged_options.items():
            print(f"  🚫 {drug}: {reason}")

    print("\n--- Safety Warnings ---")
    for w in response.safety_warnings:
        print(f"  ⚠️  {w}")

    print("\n" + "=" * 70)
    print("⚕️  Research prototype — not for unsupervised clinical use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
