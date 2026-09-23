"""Evidence Fusion Layer: merge RAG evidence with Causal estimates.

Applies deterministic contraindication checks and conflict reconciliation.
When a causal engine reports a favourable treatment effect for a drug that
guideline rules prohibit, the Fusion Layer issues a HARD CONTRAINDICATION
override and drops the drug from eligible options.
"""

from __future__ import annotations

from app.schemas.contracts import (
    CausalEstimatePayload,
    MedicalChunk,
    PatientContext,
    RAGEvidencePayload,
    UnifiedCDSSResponse,
)


# ---------------------------------------------------------------------------
# Deterministic Contraindication Rules
# ---------------------------------------------------------------------------

def _check_contraindications(
    patient: PatientContext,
    causal_estimates: list[CausalEstimatePayload],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Apply deterministic contraindication checks.

    Returns:
        eligible: list of drug names that pass all checks.
        flagged: dict mapping drug name -> reason for flagging.
        warnings: safety warning strings.
    """
    eligible: list[str] = []
    flagged: dict[str, str] = {}
    warnings: list[str] = []

    on_dialysis = any(
        med.lower() in ("dialysis", "hemodialysis", "peritoneal dialysis")
        for med in patient.current_meds
    )

    for estimate in causal_estimates:
        drug = estimate.drug_name.lower()

        # --- SGLT2 Inhibitors ---
        if drug in ("dapagliflozin", "farxiga"):
            if on_dialysis:
                flagged[estimate.drug_name] = (
                    "HARD CONTRAINDICATION: All SGLT2 inhibitors are blocked for "
                    "patients on dialysis."
                )
                warnings.append(
                    f"{estimate.drug_name} is contraindicated on dialysis "
                    f"(FDA label)."
                )
            elif patient.egfr < 45:
                flagged[estimate.drug_name] = (
                    f"NOT RECOMMENDED for glycemic control: eGFR {patient.egfr:.0f} "
                    f"< 45 mL/min/1.73m² (FDA FARXIGA label §2.2)."
                )
                warnings.append(
                    f"{estimate.drug_name} not recommended for glycemic control "
                    f"with eGFR < 45 mL/min/1.73m²."
                )
            else:
                eligible.append(estimate.drug_name)

        elif drug in ("empagliflozin", "jardiance"):
            if on_dialysis:
                flagged[estimate.drug_name] = (
                    "HARD CONTRAINDICATION: Empagliflozin is contraindicated in "
                    "patients on dialysis (FDA JARDIANCE label §2.2)."
                )
                warnings.append(
                    f"{estimate.drug_name} is contraindicated on dialysis."
                )
            elif patient.egfr < 30:
                flagged[estimate.drug_name] = (
                    f"NOT RECOMMENDED for glycemic control: eGFR {patient.egfr:.0f} "
                    f"< 30 mL/min/1.73m² (FDA JARDIANCE label §2.2)."
                )
                warnings.append(
                    f"{estimate.drug_name} not recommended with eGFR < 30."
                )
            else:
                eligible.append(estimate.drug_name)

        # --- DPP-4 Inhibitors ---
        elif drug in ("sitagliptin", "januvia"):
            if patient.egfr >= 45:
                dose_note = "100 mg once daily (no adjustment)"
            elif patient.egfr >= 30:
                dose_note = "50 mg once daily (dose adjusted for eGFR 30-<45)"
            else:
                dose_note = "25 mg once daily (dose adjusted for eGFR <30/ESRD)"
            eligible.append(estimate.drug_name)
            warnings.append(
                f"{estimate.drug_name}: Recommended dose based on renal function: "
                f"{dose_note} (FDA JANUVIA label §2.2)."
            )

        # --- Sulfonylureas ---
        elif drug in ("glimepiride", "sulfonylurea"):
            eligible.append(estimate.drug_name)
            hypo_risk = "HIGH" if patient.prior_hypo else "MODERATE"
            warnings.append(
                f"{estimate.drug_name}: Hypoglycemia risk is {hypo_risk}. "
                f"Start at 1 mg/day in renal impairment. Max 8 mg/day."
            )
            if patient.prior_hypo:
                warnings.append(
                    f"CAUTION: Patient has prior hypoglycemia history. "
                    f"Consider alternative to {estimate.drug_name}."
                )

        else:
            # Unknown drug -> pass through but flag for review
            eligible.append(estimate.drug_name)
            warnings.append(
                f"{estimate.drug_name}: No specific contraindication rules found; "
                f"clinician review recommended."
            )

    return eligible, flagged, warnings


# ---------------------------------------------------------------------------
# Citation Synthesis
# ---------------------------------------------------------------------------

def _synthesize_advice(
    patient: PatientContext,
    eligible: list[str],
    flagged: dict[str, str],
    causal_estimates: list[CausalEstimatePayload],
    rag_chunks: list[MedicalChunk],
    warnings: list[str],
) -> str:
    """Generate grounded clinical advice with inline source citations."""
    lines: list[str] = []
    lines.append("## Clinical Recommendation Summary\n")
    lines.append(
        f"Patient profile: Age {patient.age}, BMI {patient.bmi:.1f}, "
        f"eGFR {patient.egfr:.0f} mL/min/1.73m², HbA1c {patient.hba1c:.1f}%.\n"
    )

    if eligible:
        lines.append("### Eligible Treatment Options\n")
        for drug_name in eligible:
            est = next((e for e in causal_estimates if e.drug_name == drug_name), None)
            if est:
                lines.append(
                    f"- **{drug_name}**: Expected HbA1c change "
                    f"{est.cate_hba1c_delta:+.2f}% "
                    f"[95% CI: {est.ci_lower:+.2f}, {est.ci_upper:+.2f}%]"
                )
                if est.overlap_status == "NOT_APPLICABLE":
                    lines.append(
                        f"  ⚠ Causal estimate may not be reliable "
                        f"(insufficient propensity overlap)."
                    )

    if flagged:
        lines.append("\n### Blocked / Flagged Options\n")
        for drug_name, reason in flagged.items():
            lines.append(f"- **{drug_name}**: {reason}")

    if rag_chunks:
        lines.append("\n### Supporting Evidence\n")
        for i, chunk in enumerate(rag_chunks[:5], 1):
            lines.append(
                f"{i}. [{chunk.doc_title}] §{chunk.section_path}, "
                f"p.{chunk.page_number} — \"{chunk.content[:120]}...\""
            )

    lines.append(
        "\n---\n"
        "⚕ *Disclaimer: This is a research prototype for clinician evaluation. "
        "Not a marketed medical device. Not for unsupervised clinical use.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse(
    patient: PatientContext,
    rag_evidence: RAGEvidencePayload,
    causal_estimates: list[CausalEstimatePayload],
) -> UnifiedCDSSResponse:
    """Fuse RAG evidence and causal estimates into a unified CDSS response.

    Applies deterministic contraindication checks, conflict reconciliation,
    and generates grounded clinical advice.
    """
    eligible, flagged, warnings = _check_contraindications(patient, causal_estimates)

    # --- Conflict Reconciliation ---
    # If causal engine says a drug is effective but guidelines block it,
    # the guideline rule wins with a HARD CONTRAINDICATION.
    for drug_name in list(flagged.keys()):
        est = next((e for e in causal_estimates if e.drug_name == drug_name), None)
        if est and est.cate_hba1c_delta < 0:  # Favourable effect
            flagged[drug_name] = (
                f"HARD CONTRAINDICATION OVERRIDE: Causal model shows favourable "
                f"effect ({est.cate_hba1c_delta:+.2f}%) but guideline rules "
                f"prohibit this drug. Guideline takes precedence. "
                f"Original reason: {flagged[drug_name]}"
            )

    rag_chunks = rag_evidence.chunks if rag_evidence.status == "SUCCESS" else []

    advice = _synthesize_advice(
        patient, eligible, flagged, causal_estimates, rag_chunks, warnings
    )

    return UnifiedCDSSResponse(
        patient=patient,
        eligible_options=eligible,
        flagged_options=flagged,
        causal_estimates=causal_estimates,
        rag_evidence=rag_evidence,
        synthesized_advice=advice,
        safety_warnings=warnings,
    )
