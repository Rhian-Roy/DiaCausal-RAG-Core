"""Strict Pydantic v2 data contracts for the DiaCausal-RAG-Core pipeline.

Every payload exchanged between pipeline stages is validated through these
schemas, guaranteeing type-safe, auditable data flow from ingestion to
final clinical synthesis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MedicalChunk(BaseModel):
    """A single evidence chunk extracted from a clinical document."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk.")
    doc_title: str = Field(..., description="Title of the source document.")
    issuing_body: str = Field(
        ..., description="Organisation that published the document (e.g. FDA, ICMR)."
    )
    publication_year: int = Field(
        ..., ge=1900, le=2100, description="Year the document was published."
    )
    section_path: str = Field(
        ..., description="Hierarchical section path (e.g. '2.2 Dosage in Renal Impairment')."
    )
    page_number: int = Field(..., ge=0, description="Page number in the source PDF.")
    content: str = Field(..., min_length=1, description="Full text of the chunk.")
    cited_span: str | None = Field(
        default=None,
        description="Exact substring used as an inline citation.",
    )
    authority_tier: int = Field(
        ...,
        ge=1,
        le=5,
        description="Evidence authority tier (1=FDA label, 2=major guideline, 3=consensus, 4=observational, 5=expert opinion).",
    )


class RAGEvidencePayload(BaseModel):
    """Aggregated retrieval result returned by the RAG pipeline."""

    query: str = Field(..., min_length=1, description="Original user query.")
    status: Literal["SUCCESS", "INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"] = Field(
        ..., description="Outcome status of the retrieval."
    )
    chunks: list[MedicalChunk] = Field(
        default_factory=list, description="Ranked evidence chunks."
    )
    retrieval_latency_ms: float = Field(
        ..., ge=0, description="Wall-clock retrieval latency in milliseconds."
    )


class CausalEstimatePayload(BaseModel):
    """Treatment-effect estimate produced by the causal inference engine."""

    drug_name: str = Field(..., description="Canonical drug name.")
    cate_hba1c_delta: float = Field(
        ..., description="Conditional Average Treatment Effect on HbA1c (percentage points)."
    )
    ci_lower: float = Field(..., description="Lower bound of 95% confidence interval.")
    ci_upper: float = Field(..., description="Upper bound of 95% confidence interval.")
    overlap_status: Literal["SUPPORTED", "NOT_APPLICABLE"] = Field(
        ..., description="Whether propensity-score overlap is sufficient."
    )

    @field_validator("ci_upper")
    @classmethod
    def ci_upper_gte_lower(cls, v: float, info) -> float:
        if "ci_lower" in info.data and v < info.data["ci_lower"]:
            raise ValueError("ci_upper must be >= ci_lower")
        return v


class PatientContext(BaseModel):
    """Structured patient profile supplied by the clinician."""

    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    bmi: float = Field(..., gt=5, lt=100, description="Body mass index (kg/m²).")
    egfr: float = Field(
        ..., ge=0, le=150, description="Estimated Glomerular Filtration Rate (mL/min/1.73m²)."
    )
    hba1c: float = Field(
        ..., ge=4.0, le=20.0, description="Glycated haemoglobin (%%)."
    )
    prior_hypo: bool = Field(
        ..., description="History of hypoglycemic episodes."
    )
    current_meds: list[str] = Field(
        default_factory=list, description="List of current medications."
    )


class UnifiedCDSSResponse(BaseModel):
    """Final clinical decision-support response combining RAG evidence and causal estimates."""

    patient: PatientContext
    eligible_options: list[str] = Field(
        default_factory=list,
        description="Drug names that pass all safety and eligibility checks.",
    )
    flagged_options: dict[str, str] = Field(
        default_factory=dict,
        description="Drug name -> reason for flagging/blocking.",
    )
    causal_estimates: list[CausalEstimatePayload] = Field(
        default_factory=list,
        description="CATE estimates for each candidate drug.",
    )
    rag_evidence: RAGEvidencePayload
    synthesized_advice: str = Field(
        ..., description="Clinician-facing grounded recommendation text."
    )
    safety_warnings: list[str] = Field(
        default_factory=list,
        description="Critical safety warnings that must be surfaced.",
    )
