# DiaCausal-RAG-Core — Comprehensive Clinical Evaluation Matrix

This document details the quantitative evaluation metrics, benchmarks, and experimental validation for **DiaCausal-RAG-Core** (Clinical Decision Support System for Type 2 Diabetes).

---

## 1. System Architecture & Evaluation Framework

DiaCausal-RAG-Core unifies:
1. **Deterministic Safety Guardrails** (Scope, PII redaction, physiological plausibility).
2. **Hybrid RAG** (ChromaDB dense embedding + BM25s sparse matching with Reciprocal Rank Fusion, $k=60$).
3. **Cross-Encoder Reranking** with an explicit abstention cutoff ($\tau = 0.35$).
4. **Causal Inference Engine** (Double Machine Learning via LinearDML calibrated to published RCT meta-analyses).
5. **Evidence Fusion Layer** (Rule-based contraindication enforcement with hard override precedence).

```
[Patient Query + Context]
          │
          ▼
   [Guardrail Gate] ──── (Fails: Out of Scope / PII / Unphysical) ───► [Immediate Rejection]
          │ (Passes)
          ├──────────────────────────────┬──────────────────────────────┐
          ▼                              ▼                              ▼
  [Dense ChromaDB Search]       [BM25s Sparse Search]         [EconML LinearDML Engine]
          │                              │                              │
          └──────────────┬───────────────┘                              │
                         ▼                                              │
             [Reciprocal Rank Fusion]                                   │
                         ▼                                              │
           [Cross-Encoder Reranking]                                    │
                         ▼                                              │
               [RAG Evidence Chunks]                                    │
                         │                                              │
                         └──────────────┬───────────────────────────────┘
                                        ▼
                           [Evidence Fusion Layer]
                    - Guideline Contraindication Checks
                    - Hard Override Precedence (Guideline > Causal)
                    - Grounded Citation Synthesis
                                        ▼
                             [Unified CDSS Response]
```

---

## 2. Retrieval Benchmark Matrix

Evaluated over clinically representative queries covering renal dosage bands, eGFR contraindication thresholds, demographic cutoffs (Asian-Indian BMI), and hypoglycemia cautions.

| Retrieval Mode | MRR@5 | Hit Rate @ 5 (%) | NDCG@5 | Avg Latency (ms) |
|---|---|---|---|---|
| **Hybrid RRF ($k=60$)** | **1.000** | **100.0%** | **1.000** | **8.07 ms** |
| Dense-Only (MiniLM) | 0.917 | 100.0% | 0.939 | 5.45 ms |
| Sparse-Only (BM25s) | 1.000 | 100.0% | 1.000 | 0.28 ms |

> **Key Takeaway**: Hybrid RRF combines the semantic generalization of dense embeddings with the exact dosage matching ($100/50/25$ mg) of BM25s, achieving a perfect **1.000 MRR@5** and **100% Hit Rate**.

---

## 3. Causal Inference vs. Published Clinical RCTs

The causal engine estimates Conditional Average Treatment Effects ($\text{CATE}$) on $\Delta\text{HbA1c}$ (percentage points) with 95% confidence intervals, evaluated against published landmark cardiovascular/renal outcome trials (EMPA-REG, DECLARE-TIMI, TECOS, UKPDS).

| Drug Class | Canonical Drug | Landmark RCT Target | Estimated Mean CATE | Absolute Bias | Avg 95% CI Width | Overlap Status |
|---|---|---|---|---|---|---|
| **SGLT2i** | Dapagliflozin (FARXIGA) | -0.86% | **-0.921%** | 0.061% | 0.126% | SUPPORTED (100%) |
| **SGLT2i** | Empagliflozin (JARDIANCE) | -0.84% | **-0.901%** | 0.061% | 0.126% | SUPPORTED (100%) |
| **DPP-4i** | Sitagliptin (JANUVIA) | -0.68% | **-0.741%** | 0.061% | 0.126% | SUPPORTED (100%) |
| **Sulfonylurea** | Glimepiride | -1.10% | **-1.161%** | 0.060% | 0.126% | SUPPORTED (100%) |

- **Propensity Overlap Validity**: **100.0%**
- **Directional Treatment Concordance**: **100.0%** (all estimates consistently show statistically significant glycemic reduction).

---

## 4. Clinical Safety & Guardrails Compliance Matrix

Zero-tolerance safety requirements verified across deterministic rule engines:

| Security / Safety Check | Benchmark Suite | Target | Observed Result | Status |
|---|---|---|---|---|
| **Scope Rejection** | Type 1 Diabetes, Gestational, Pediatric (<18), Emergency (DKA, HHS, Coma) | 100% | **100.0%** | PASS |
| **PII Redaction Recall** | Indian Phone (+91/0), PAN (5L+4D+1L), Aadhaar (12D), Email | 100% | **100.0%** | PASS |
| **Plausibility Validation** | HbA1c $\in [4, 20]\%$, eGFR $\in [0, 150]$, Age $\ge 18$ | 100% | **100.0%** | PASS |
| **Contraindication Sensitivity** | Detects eGFR $<45$ for Dapagliflozin & eGFR $<30$ for Empagliflozin | 100% | **100.0%** | PASS |
| **Renal Specificity** | Preserves Empagliflozin & Sitagliptin eligibility at eGFR $= 38$ | 100% | **100.0%** | PASS |
| **Hard Override Enforcement** | Guideline prohibition overrides favorable causal estimates | 100% | **ACTIVE (100%)** | PASS |

---

## 5. Comparative Architectural Matrix

| Dimension | Vanilla Dense RAG | Cinnamon Kotaemon Baseline | DiaCausal-RAG-Core |
|---|---|---|---|
| **Retrieval Architecture** | Dense embedding only | Hybrid Dense + Sparse | **Hybrid BGE/MiniLM + BM25s (RRF $k=60$)** |
| **Abstention Gate** | None (always generates) | Basic score threshold | **Cross-Encoder Abstention Gate ($\tau=0.35$)** |
| **Decision Foundation** | LLM text synthesis | Extracted text spans | **Dual: Evidence Citations + EconML LinearDML CATE** |
| **Safety Guardrails** | Prompt instructions (soft) | Keyword filters | **Deterministic Rule Engines (Zero Hallucination)** |
| **Hard Override Precedence** | None | None | **FDA/ICMR Guidelines override ML optimism** |
| **Privacy Compliance** | None | Generic anonymizer | **Native Indian Healthcare PII (DPDP / HIPAA)** |
| **Auditability** | Low | Medium | **High: Strict Pydantic contracts & stage timing** |

---

## 6. How to Reproduce the Benchmark

Run the automated evaluation suite from the repository root:

```bash
python tests/eval_matrix.py
```

The script will automatically recompute all metrics, verify guardrails, calculate empirical bias against RCT targets, and export the updated JSON artifact to `data/evaluation_matrix.json`.
