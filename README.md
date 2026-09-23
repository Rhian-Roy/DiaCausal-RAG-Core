# DiaCausal-RAG-Core

> **Hybrid RAG + Causal Inference Clinical Decision Support System for Type 2 Diabetes**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

DiaCausal-RAG-Core is a clinical decision support system that combines **hybrid Retrieval-Augmented Generation (RAG)** with **causal inference** to provide evidence-grounded, safety-checked treatment recommendations for adult Type 2 Diabetes management.

### Key Features

- **Hybrid Retrieval (Dense + Sparse)**: Combines BAAI/bge-m3 dense embeddings with BM25s exact matching via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-scores candidate chunks with an abstention gate (τ = 0.35) — returns "INSUFFICIENT_EVIDENCE" when confidence is too low.
- **Deterministic Safety Guardrails**: Rule-based scope filtering (blocks Type 1, pediatric, pregnancy, emergency), identifier redaction (phone, PAN, Aadhaar, email), and clinical plausibility validation.
- **Causal Inference Bridge**: Interfaces with [DiaCausal-CDSS](https://github.com/Rhian-Roy/DiaCausal-CDSS) or uses a built-in Double Machine Learning (LinearDML) fallback calibrated to published clinical trial effects.
- **Evidence Fusion Layer**: Deterministic contraindication checks (eGFR-based SGLT2i/DPP-4i rules), conflict reconciliation (guideline overrides causal estimates), and grounded citation synthesis.
- **Streamlit Clinical Copilot UI**: Three-panel split view with patient input, decision cards (color-coded: Teal/Amber/Red), and a pipeline inspector with evidence sources.

## Architecture

```
┌────────────┐     ┌──────────────────────────────────────────────────┐
│  Clinician │────>│              Input Guardrails                    │
│   Input    │     │  (Scope Filter, PII Redaction, Plausibility)     │
└────────────┘     └──────────┬───────────────────────┬───────────────┘
                              │                       │
                    ┌─────────▼─────────┐   ┌─────────▼─────────┐
                    │    RAG Pipeline    │   │   Causal Engine    │
                    │  Dense + BM25s    │   │   (LinearDML /     │
                    │  RRF Fusion       │   │    DiaCausal-CDSS) │
                    │  Cross-Encoder    │   │                    │
                    └─────────┬─────────┘   └─────────┬──────────┘
                              │                       │
                    ┌─────────▼───────────────────────▼──────────┐
                    │         Evidence Fusion Layer               │
                    │  Contraindication checks, Conflict         │
                    │  reconciliation, Citation synthesis         │
                    └─────────────────────┬──────────────────────┘
                                          │
                    ┌─────────────────────▼──────────────────────┐
                    │       Unified CDSS Response                │
                    │  Eligible options, Flagged drugs,          │
                    │  CATE estimates, Safety warnings           │
                    └────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Rhian-Roy/DiaCausal-RAG-Core.git
cd DiaCausal-RAG-Core
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Dashboard

```bash
streamlit run diacausal_app.py
```

The Clinical Copilot will open in your browser at `http://localhost:8501`.

## Project Structure

```
DiaCausal-RAG-Core/
├── app/
│   ├── schemas/
│   │   └── contracts.py         # Strict Pydantic v2 data models
│   ├── security/
│   │   └── guardrails.py        # Deterministic input & scope checks
│   ├── rag/
│   │   ├── ingest.py            # PDF/Markdown parsing & chunking
│   │   ├── retriever.py         # Dense + BM25s hybrid search with RRF
│   │   ├── reranker.py          # Cross-encoder scoring & abstention gate
│   │   └── fusion.py            # Evidence Fusion (RAG + Causal)
│   ├── bridge/
│   │   └── causal_connector.py  # Bridge to DiaCausal-CDSS causal engine
│   └── ui/
│       └── dashboard.py         # Streamlit Clinical Copilot Split View
├── data/
│   ├── raw/                     # Clinical guidelines & drug labels
│   └── processed/chroma_db/     # Vector store persistence
├── tests/
│   ├── test_guardrails.py       # Safety guardrail tests
│   ├── test_retrieval.py        # Hybrid retrieval tests
│   └── test_full_pipeline.py    # End-to-end clinical case tests
├── requirements.txt
├── diacausal_app.py             # Root launch script
└── README.md
```

## Clinical Drug Coverage

| Drug Class | Drug | eGFR Threshold | Key Safety Rule |
|---|---|---|---|
| SGLT2i | Dapagliflozin (FARXIGA) | Not recommended < 45 | Blocked on dialysis |
| SGLT2i | Empagliflozin (JARDIANCE) | Not recommended < 30 | Contraindicated on dialysis |
| DPP-4i | Sitagliptin (JANUVIA) | Dose adjusted by band | 100/50/25 mg by eGFR |
| Sulfonylurea | Glimepiride | Start 1 mg in renal impairment | High hypoglycemia risk |

## Causal Inference

The system estimates Conditional Average Treatment Effects (CATE) for each drug using Double Machine Learning (LinearDML). Calibration targets are derived from published clinical trials:

| Drug | Mean HbA1c Δ | 95% CI Width |
|---|---|---|
| SGLT2i (Dapagliflozin) | -0.86% | ±0.18% |
| SGLT2i (Empagliflozin) | -0.84% | ±0.19% |
| DPP-4i (Sitagliptin) | -0.68% | ±0.15% |
| Sulfonylurea (Glimepiride) | -1.10% | ±0.20% |

## Testing

```bash
pytest tests/ -v
```

## Companion Repository

This system is designed to integrate with [DiaCausal-CDSS](https://github.com/Rhian-Roy/DiaCausal-CDSS), which provides the full causal inference engine. If the companion repo is available locally, DiaCausal-RAG-Core will automatically import and use its estimation logic.

## Disclaimer

⚕️ **This is a research prototype for clinician evaluation. It is not a marketed medical device and is not intended for unsupervised clinical use.**

## License

MIT License — see [LICENSE](LICENSE) for details.
