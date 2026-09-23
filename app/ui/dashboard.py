"""Streamlit Clinical Copilot Split-View Dashboard.

Features:
- Three top-level tabs:
  1. 🩺 Clinical Copilot (Split-view with Patient Context, Decision Matrix, and Pipeline Inspector)
  2. 📊 Evaluation Matrix (Live & persisted benchmarks: Retrieval, Causal vs RCT, Safety)
  3. 📑 Guideline & Label Repository (Full text of ingested FDA labels & ICMR workflows)
- Preset Clinical Cases for 1-click evaluation.
- Real-time pipeline latency tracking and grounded citation display.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app.schemas.contracts import (
    CausalEstimatePayload,
    PatientContext,
    RAGEvidencePayload,
    UnifiedCDSSResponse,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_MATRIX_PATH = _PROJECT_ROOT / "data" / "evaluation_matrix.json"
_RAW_DATA_PATH = _PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Styling Helpers
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    """Return an HTML badge for a drug status."""
    colors = {
        "eligible": ("#0d9488", "#f0fdfa"),  # Teal
        "caution": ("#d97706", "#fffbeb"),   # Amber
        "blocked": ("#dc2626", "#fef2f2"),   # Red
    }
    bg, fg_bg = colors.get(status, ("#6b7280", "#f9fafb"))
    label = status.upper()
    return (
        f'<span style="background:{bg}; color:white; padding:3px 12px; '
        f'border-radius:12px; font-size:0.85em; font-weight:700;">{label}</span>'
    )


def _format_cate(est: CausalEstimatePayload) -> str:
    """Format a CATE estimate with 95% CI."""
    return (
        f"{est.cate_hba1c_delta:+.2f}% "
        f"[95% CI: {est.ci_lower:+.2f}, {est.ci_upper:+.2f}%]"
    )


# ---------------------------------------------------------------------------
# Patient Input Panel with Clinical Presets
# ---------------------------------------------------------------------------

PRESET_CASES = {
    "Custom Input": None,
    "Preset 1: Moderate CKD (eGFR 38, HbA1c 8.6%)": {
        "age": 55,
        "bmi": 27.5,
        "egfr": 38.0,
        "hba1c": 8.6,
        "prior_hypo": False,
        "current_meds": "Metformin",
        "description": "Tests Dapagliflozin cutoff (eGFR < 45 blocked), Empagliflozin eligible (cutoff 30), Sitagliptin dose-adjusted to 50mg.",
    },
    "Preset 2: Severe CKD / Dialysis (eGFR 22)": {
        "age": 63,
        "bmi": 26.2,
        "egfr": 22.0,
        "hba1c": 9.4,
        "prior_hypo": False,
        "current_meds": "Metformin, Dialysis",
        "description": "Tests SGLT2i dialysis hard contraindication and Sitagliptin 25mg renal band.",
    },
    "Preset 3: Normal Renal Profile (eGFR 88, HbA1c 9.1%)": {
        "age": 48,
        "bmi": 29.0,
        "egfr": 88.0,
        "hba1c": 9.1,
        "prior_hypo": False,
        "current_meds": "Metformin",
        "description": "Unrestricted renal function — all drug classes pass baseline eligibility.",
    },
    "Preset 4: High Hypoglycemia Risk (Prior Hypo)": {
        "age": 67,
        "bmi": 24.8,
        "egfr": 58.0,
        "hba1c": 8.2,
        "prior_hypo": True,
        "current_meds": "Metformin",
        "description": "Tests Glimepiride high hypoglycemia risk safety warning & clinician caution.",
    },
}


def render_patient_panel() -> PatientContext | None:
    """Render the left-panel patient context form with preset shortcuts."""
    st.markdown("### 🏥 Patient Context")

    selected_preset_name = st.selectbox(
        "Clinical Benchmark Presets",
        options=list(PRESET_CASES.keys()),
        index=1,  # Default to Preset 1 for immediate out-of-the-box demo
        help="Select a benchmark patient or choose 'Custom Input' to type manually.",
    )

    preset_data = PRESET_CASES.get(selected_preset_name)
    if preset_data:
        st.info(f"💡 **Case Profile**: {preset_data['description']}")
        default_age = preset_data["age"]
        default_bmi = preset_data["bmi"]
        default_egfr = preset_data["egfr"]
        default_hba1c = preset_data["hba1c"]
        default_hypo = preset_data["prior_hypo"]
        default_meds = preset_data["current_meds"]
    else:
        default_age = 55
        default_bmi = 27.5
        default_egfr = 65.0
        default_hba1c = 8.2
        default_hypo = False
        default_meds = "Metformin"

    with st.form("patient_form"):
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=default_age)
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=80.0,
            value=default_bmi,
            step=0.1,
            help="Asian-Indian cutoffs: ≥23 overweight, ≥25 obese",
        )
        egfr = st.number_input(
            "eGFR (mL/min/1.73m²)",
            min_value=0.0,
            max_value=150.0,
            value=default_egfr,
            step=1.0,
        )
        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=4.0,
            max_value=20.0,
            value=default_hba1c,
            step=0.1,
        )
        prior_hypo = st.checkbox("Prior Hypoglycemia Episodes", value=default_hypo)
        current_meds_str = st.text_input(
            "Current Medications (comma-separated)",
            value=default_meds,
        )

        submitted = st.form_submit_button("🔍 Analyze Treatment Options", type="primary", use_container_width=True)

    if submitted:
        current_meds = [m.strip() for m in current_meds_str.split(",") if m.strip()]
        try:
            return PatientContext(
                age=age,
                bmi=bmi,
                egfr=egfr,
                hba1c=hba1c,
                prior_hypo=prior_hypo,
                current_meds=current_meds,
            )
        except Exception as e:
            st.error(f"Invalid patient data: {e}")
    return None


# ---------------------------------------------------------------------------
# Decision Cards Panel
# ---------------------------------------------------------------------------

def render_decision_cards(response: UnifiedCDSSResponse) -> None:
    """Render the center panel with treatment decision cards."""
    st.markdown("### 💊 Treatment Decision Matrix")
    st.markdown(
        '<p style="color:#6b7280; font-size:0.85em;">'
        "Multi-agent causal evaluation comparing SGLT2i vs DPP-4i vs Sulfonylurea with guideline safety gates.</p>",
        unsafe_allow_html=True,
    )

    for estimate in response.causal_estimates:
        drug = estimate.drug_name
        is_eligible = drug in response.eligible_options
        is_flagged = drug in response.flagged_options

        if is_flagged:
            status = "blocked"
        elif is_eligible and estimate.overlap_status == "NOT_APPLICABLE":
            status = "caution"
        elif is_eligible:
            status = "eligible"
        else:
            status = "caution"

        with st.container():
            st.markdown("---")
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown(f"#### {drug}")
                st.markdown(f"CATE (ΔHbA1c): **{_format_cate(estimate)}**")
                st.caption(f"Propensity Overlap: `{estimate.overlap_status}`")
            with cols[1]:
                st.markdown(_status_badge(status), unsafe_allow_html=True)
                if is_flagged:
                    st.error(response.flagged_options[drug])
                elif is_eligible:
                    st.success("✅ Eligible based on current renal function & guideline rules.")

    # Safety warnings
    if response.safety_warnings:
        st.markdown("### ⚠️ Clinical Safety Warnings")
        for warning in response.safety_warnings:
            st.warning(warning)

    # Disclaimer
    st.info(
        "⚕️ **Clinical Disclaimer**: Research prototype for clinician evaluation. "
        "Not a marketed medical device. Not for unsupervised clinical use."
    )


# ---------------------------------------------------------------------------
# Pipeline Inspector Panel
# ---------------------------------------------------------------------------

def render_pipeline_inspector(
    response: UnifiedCDSSResponse,
    stage_times: dict[str, float] | None = None,
) -> None:
    """Render the right panel showing pipeline trace and evidence drawer."""
    st.markdown("### 🔬 Pipeline Inspector")

    # Stage timeline
    stages = [
        "Input Guard",
        "Parallel RAG & Causal Execution",
        "RRF Fusion",
        "Reranker",
        "Fusion Safety Gate",
        "Final Synthesis",
    ]
    times = stage_times or {}

    st.markdown("**Execution Pipeline:**")
    for s in stages:
        t = times.get(s, 0.0)
        st.markdown(f"- `{s}`: **{t:.1f} ms**")

    total_ms = sum(times.values()) if times else 0
    st.caption(f"⏱️ **Total Pipeline Latency:** `{total_ms:.1f} ms`")

    # Evidence drawer
    st.markdown("---")
    st.markdown("### 📚 Grounded Evidence Drawer")
    rag = response.rag_evidence
    st.caption(
        f"Status: **{rag.status}** | "
        f"Retrieval Latency: `{rag.retrieval_latency_ms:.1f} ms` | "
        f"Chunks: `{len(rag.chunks)}`"
    )

    if not rag.chunks:
        st.caption("No chunks retrieved or query was abstained by reranker threshold.")

    for i, chunk in enumerate(rag.chunks, 1):
        with st.expander(f"📄 Source {i}: {chunk.doc_title} — §{chunk.section_path}"):
            st.markdown(f"**Document:** {chunk.doc_title}")
            st.markdown(f"**Section:** {chunk.section_path}")
            st.markdown(f"**Page:** {chunk.page_number}")
            st.markdown(f"**Issuing Authority:** {chunk.issuing_body} ({chunk.publication_year})")
            st.markdown(f"**Authority Tier:** Tier {chunk.authority_tier}")
            st.markdown("---")
            st.markdown(f"> {chunk.content}")


# ---------------------------------------------------------------------------
# Overview Panel (when no patient submitted yet)
# ---------------------------------------------------------------------------

def render_overview_panel() -> None:
    """Render informative welcome screen with benchmark launcher when no query has run."""
    st.markdown("### 🩺 Clinical Copilot Overview")
    st.markdown(
        """
        Welcome to **DiaCausal-RAG-Core**, a decision-support system for adult **Type 2 Diabetes** pharmacotherapy add-on selection.
        
        #### How it Works:
        1. **Deterministic Guardrails**: Filters out-of-scope cases (Type 1, pregnancy, pediatric, emergencies) and redacts PII.
        2. **Hybrid RAG Retrieval**: Combines dense semantic search (`all-MiniLM-L6-v2`) with sparse medical keyword matching (`BM25s`) via **Reciprocal Rank Fusion ($k=60$)**.
        3. **Cross-Encoder Reranker**: Scores candidates with an abstention threshold ($\tau = 0.35$).
        4. **Causal Inference Engine**: Evaluates heterogeneous treatment effects using **Double Machine Learning (LinearDML)** calibrated to published clinical trials.
        5. **Evidence Fusion Layer**: Implements a strict clinical hierarchy — **FDA/ICMR guideline contraindications ALWAYS override algorithmic optimism**.
        """
    )

    st.markdown("---")
    st.markdown("### 🚀 Quick Benchmark Cases")
    st.markdown("Select a preset from the left panel and click **'🔍 Analyze Treatment Options'** to execute the pipeline.")


# ---------------------------------------------------------------------------
# Evaluation Matrix Tab
# ---------------------------------------------------------------------------

def render_evaluation_matrix_tab() -> None:
    """Render the comprehensive quantitative evaluation matrix tab."""
    st.markdown("## 📊 Comprehensive Clinical Evaluation Matrix")
    st.caption("Quantifies Retrieval Performance, Causal Inference Calibration, and Clinical Safety Guardrails.")

    matrix_data: dict[str, Any] | None = None
    if _EVAL_MATRIX_PATH.exists():
        try:
            matrix_data = json.loads(_EVAL_MATRIX_PATH.read_text(encoding="utf-8"))
        except Exception:
            matrix_data = None

    if st.button("🔄 Re-run Live Benchmark Evaluation", type="secondary"):
        with st.spinner("Executing evaluation matrix benchmark suite across retrieval, causal, and safety suites…"):
            try:
                from tests.eval_matrix import run_full_evaluation_matrix
                matrix_data = run_full_evaluation_matrix()
                st.success("✅ Live benchmark completed and exported!")
            except Exception as e:
                st.error(f"Benchmark run failed: {e}")

    if not matrix_data:
        st.info("Evaluation matrix JSON not found. Click 'Re-run Live Benchmark Evaluation' to generate.")
        return

    st.markdown(f"**Last Benchmark Timestamp:** `{matrix_data.get('timestamp')}` | **Duration:** `{matrix_data.get('benchmark_duration_seconds')}s`")

    # 1. Retrieval Performance
    st.markdown("### 1. Retrieval Performance Matrix")
    ret_perf = matrix_data.get("retrieval_performance", {})
    cols = st.columns(len(ret_perf))
    for col, (mode, vals) in zip(cols, ret_perf.items()):
        with col:
            st.metric(label=f"{mode} MRR@5", value=f"{vals.get('MRR@5'):.3f}")
            st.caption(f"Hit Rate @ 5: **{vals.get('HitRate@5')}%**")
            st.caption(f"NDCG@5: **{vals.get('NDCG@5'):.3f}**")
            st.caption(f"Avg Latency: **{vals.get('Avg_Latency_ms')} ms**")

    # 2. Safety & Guardrail Compliance
    st.markdown("---")
    st.markdown("### 2. Clinical Safety & Guardrail Compliance")
    safety = matrix_data.get("safety_and_guardrails", {})
    s_cols = st.columns(3)
    with s_cols[0]:
        st.metric("Scope Rejection Accuracy", f"{safety.get('Scope_Rejection_Accuracy_%')}%")
        st.metric("PII Redaction Recall", f"{safety.get('PII_Redaction_Recall_%')}%")
    with s_cols[1]:
        st.metric("Plausibility Validation", f"{safety.get('Plausibility_Validation_Accuracy_%')}%")
        st.metric("Contraindication Sensitivity", f"{safety.get('Contraindication_Sensitivity_%')}%")
    with s_cols[2]:
        st.metric("Renal Preservation Specificity", f"{safety.get('Renal_Preservation_Specificity_%')}%")
        override_status = "ACTIVE ✅" if safety.get("Hard_Contraindication_Override_Active") else "INACTIVE ❌"
        st.metric("Hard Override Gate", override_status)

    # 3. Causal Estimation vs. Landmark RCTs
    st.markdown("---")
    st.markdown("### 3. Causal Treatment Effects vs. Published Landmark RCTs")
    st.caption("Validates LinearDML CATE estimations against published clinical trial effect sizes (DECLARE-TIMI, EMPA-REG, TECOS, UKPDS).")
    causal = matrix_data.get("causal_inference_performance", {}).get("drug_eval", {})
    
    rows = []
    for drug, data in causal.items():
        rows.append({
            "Drug Class / Name": drug,
            "RCT Meta-Analysis Target": f"{data.get('RCT_Target_Mean_%'):.2f}%",
            "LinearDML Mean CATE": f"{data.get('Estimated_Mean_CATE_%'):.3f}%",
            "Absolute Bias": f"{data.get('Absolute_Bias_%'):.3f}%",
            "Avg 95% CI Width": f"{data.get('Avg_95CI_Width'):.3f}%",
            "Directional Concordance": "100% ✅" if data.get("Directional_Accuracy_%") == 100 else "0% ❌",
        })
    st.dataframe(rows, use_container_width=True)

    # 4. Comparative Architectural Matrix
    st.markdown("---")
    st.markdown("### 4. Comparative Architectural Strengths")
    comp = matrix_data.get("comparative_architecture", [])
    if comp:
        st.dataframe(comp, use_container_width=True)


# ---------------------------------------------------------------------------
# Guideline Repository Tab
# ---------------------------------------------------------------------------

def render_guidelines_tab() -> None:
    """Render the guideline repository explorer."""
    st.markdown("## 📑 Clinical Guideline & FDA Prescribing Information Library")
    st.caption("Browse all ingested regulatory documents and clinical guidelines.")

    if not _RAW_DATA_PATH.exists():
        st.info("Raw guidelines directory not found.")
        return

    md_files = sorted(list(_RAW_DATA_PATH.glob("*.md")))
    if not md_files:
        st.info("No guidelines currently seeded in data/raw/.")
        return

    for path in md_files:
        with st.expander(f"📖 {path.stem.replace('_', ' ')}"):
            st.markdown(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(run_pipeline_fn: Any = None) -> None:
    """Render the full three-column Clinical Copilot dashboard with tab navigation."""
    st.set_page_config(
        page_title="DiaCausal CDSS — Clinical Copilot",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        "<h1 style='text-align:center;'>🩺 DiaCausal Clinical Copilot</h1>"
        "<p style='text-align:center; color:#6b7280;'>"
        "Evidence-Grounded Hybrid RAG + Causal Inference Decision Support for Type 2 Diabetes</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab_copilot, tab_matrix, tab_guidelines = st.tabs([
        "🩺 Clinical Copilot",
        "📊 Evaluation Matrix",
        "📑 Clinical Guidelines",
    ])

    with tab_copilot:
        left, center, right = st.columns([1, 2, 1.5])

        with left:
            patient = render_patient_panel()

        if patient and run_pipeline_fn:
            with st.spinner("Running CDSS pipeline (Guardrails → Hybrid RAG → LinearDML → Evidence Fusion)…"):
                response, stage_times = run_pipeline_fn(patient)

            with center:
                render_decision_cards(response)

            with right:
                render_pipeline_inspector(response, stage_times)

            # Synthesized advice below
            st.markdown("---")
            st.markdown("### 📋 Synthesized Clinical Advice")
            st.markdown(response.synthesized_advice)
        else:
            with center:
                render_overview_panel()

            with right:
                st.markdown("### ⚙️ System Status")
                st.success("Dense Index: ChromaDB (`all-MiniLM-L6-v2`) ✅")
                st.success("Sparse Index: BM25s Exact Terminology ✅")
                st.success("Causal Engine: EconML LinearDML ✅")
                st.success("Guardrails: Deterministic Scope & PII Gates ✅")
                st.info("Select a patient preset on the left and click 'Analyze Treatment Options' to start.")

    with tab_matrix:
        render_evaluation_matrix_tab()

    with tab_guidelines:
        render_guidelines_tab()
