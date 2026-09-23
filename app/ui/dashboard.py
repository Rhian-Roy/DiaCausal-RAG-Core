"""Streamlit Clinical Copilot Split-View Dashboard.

Layout:
- Left Panel: Patient context input form.
- Center Panel: Decision cards with CATE estimates and color-coded badges.
- Right Panel: Pipeline inspector and evidence drawer.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app.schemas.contracts import (
    CausalEstimatePayload,
    PatientContext,
    RAGEvidencePayload,
    UnifiedCDSSResponse,
)


# ---------------------------------------------------------------------------
# Styling Helpers
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    """Return an HTML badge for a drug status."""
    colors = {
        "eligible": ("#0d9488", "#f0fdfa"),   # Teal
        "caution": ("#d97706", "#fffbeb"),     # Amber
        "blocked": ("#dc2626", "#fef2f2"),     # Red
    }
    bg, fg_bg = colors.get(status, ("#6b7280", "#f9fafb"))
    label = status.upper()
    return (
        f'<span style="background:{bg}; color:white; padding:2px 10px; '
        f'border-radius:12px; font-size:0.85em; font-weight:600;">{label}</span>'
    )


def _format_cate(est: CausalEstimatePayload) -> str:
    """Format a CATE estimate with 95% CI."""
    return (
        f"{est.cate_hba1c_delta:+.2f}% "
        f"[95% CI: {est.ci_lower:+.2f}, {est.ci_upper:+.2f}%]"
    )


# ---------------------------------------------------------------------------
# Patient Input Panel
# ---------------------------------------------------------------------------

def render_patient_panel() -> PatientContext | None:
    """Render the left-panel patient context form. Returns PatientContext or None."""
    st.markdown("### 🏥 Patient Context")

    with st.form("patient_form"):
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=55)
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0, max_value=80.0, value=27.5, step=0.1,
            help="Asian-Indian cutoffs: ≥23 overweight, ≥25 obese",
        )
        egfr = st.number_input(
            "eGFR (mL/min/1.73m²)",
            min_value=0.0, max_value=150.0, value=65.0, step=1.0,
        )
        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=4.0, max_value=20.0, value=8.2, step=0.1,
        )
        prior_hypo = st.checkbox("Prior Hypoglycemia Episodes")
        current_meds_str = st.text_input(
            "Current Medications (comma-separated)",
            value="Metformin",
        )

        submitted = st.form_submit_button("🔍 Analyze Treatment Options", type="primary")

    if submitted:
        current_meds = [
            m.strip() for m in current_meds_str.split(",") if m.strip()
        ]
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
        "Comparing SGLT2i vs DPP-4i vs Sulfonylurea based on patient profile.</p>",
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
                st.markdown(f"CATE: **{_format_cate(estimate)}**")
            with cols[1]:
                st.markdown(_status_badge(status), unsafe_allow_html=True)
                if is_flagged:
                    st.caption(response.flagged_options[drug])
                elif is_eligible:
                    st.caption("Eligible for this patient")

    # Safety warnings
    if response.safety_warnings:
        st.markdown("### ⚠️ Safety Warnings")
        for warning in response.safety_warnings:
            st.warning(warning)

    # Disclaimer
    st.info(
        "⚕️ **Disclaimer**: Research prototype for clinician evaluation. "
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
    pipeline_display = " → ".join(
        f"`{s}` ({times.get(s, 0):.0f}ms)" for s in stages
    )
    st.markdown(pipeline_display)

    total_ms = sum(times.values()) if times else 0
    st.caption(f"Total pipeline latency: {total_ms:.0f}ms")

    # Evidence drawer
    st.markdown("### 📚 Evidence Sources")
    rag = response.rag_evidence
    st.caption(
        f"Retrieval status: **{rag.status}** | "
        f"Latency: {rag.retrieval_latency_ms:.0f}ms | "
        f"Chunks retrieved: {len(rag.chunks)}"
    )

    for i, chunk in enumerate(rag.chunks, 1):
        with st.expander(f"📄 Source {i}: {chunk.doc_title} — §{chunk.section_path}"):
            st.markdown(f"**Document:** {chunk.doc_title}")
            st.markdown(f"**Section:** {chunk.section_path}")
            st.markdown(f"**Page:** {chunk.page_number}")
            st.markdown(f"**Issuing Body:** {chunk.issuing_body} ({chunk.publication_year})")
            st.markdown(f"**Authority Tier:** {chunk.authority_tier}")
            st.markdown("---")
            st.markdown(f"> {chunk.content}")


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(run_pipeline_fn: Any = None) -> None:
    """Render the full three-column Clinical Copilot dashboard.

    *run_pipeline_fn* should accept a PatientContext and return
    (UnifiedCDSSResponse, dict[str, float]).
    """
    st.set_page_config(
        page_title="DiaCausal CDSS — Clinical Copilot",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown(
        "<h1 style='text-align:center;'>🩺 DiaCausal Clinical Copilot</h1>"
        "<p style='text-align:center; color:#6b7280;'>"
        "Hybrid RAG + Causal Inference Decision Support for Type 2 Diabetes</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    left, center, right = st.columns([1, 2, 1.5])

    with left:
        patient = render_patient_panel()

    if patient and run_pipeline_fn:
        with st.spinner("Running CDSS pipeline…"):
            response, stage_times = run_pipeline_fn(patient)

        with center:
            render_decision_cards(response)

        with right:
            render_pipeline_inspector(response, stage_times)

        # Synthesized advice below
        st.markdown("---")
        st.markdown("### 📋 Synthesized Clinical Advice")
        st.markdown(response.synthesized_advice)
    elif patient:
        with center:
            st.info("Pipeline function not configured. Displaying patient context only.")
            st.json(patient.model_dump())
