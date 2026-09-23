"""Full integration test for the DiaCausal-RAG-Core pipeline.

Clinical test case:
    Patient: Age 55, HbA1c 8.6%, eGFR 38 mL/min/1.73m², on Metformin.

Expected outcomes:
    - Dapagliflozin: FLAGGED (eGFR < 45, not recommended for glycemic control).
    - Empagliflozin: ELIGIBLE (eGFR 38 >= 30 cutoff).
    - Sitagliptin: ELIGIBLE (dose adjusted to 50 mg for eGFR 30-<45).
    - Glimepiride: ELIGIBLE (with hypoglycemia caution).
    - Causal estimates: all drugs produce CATE outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.contracts import (
    CausalEstimatePayload,
    PatientContext,
    RAGEvidencePayload,
    UnifiedCDSSResponse,
)


# =========================================================================
# Test Patient Fixture
# =========================================================================


@pytest.fixture
def clinical_patient() -> PatientContext:
    """Age 55, HbA1c 8.6%, eGFR 38, on Metformin."""
    return PatientContext(
        age=55,
        bmi=27.5,
        egfr=38.0,
        hba1c=8.6,
        prior_hypo=False,
        current_meds=["Metformin"],
    )


# =========================================================================
# Causal Estimation Tests
# =========================================================================


class TestCausalEstimation:
    """Test the causal bridge produces estimates for all drugs."""

    def test_all_drugs_estimated(self, clinical_patient: PatientContext) -> None:
        from app.bridge.causal_connector import estimate_all_treatments

        estimates = estimate_all_treatments(clinical_patient)
        assert len(estimates) == 4

        drug_names = {e.drug_name for e in estimates}
        assert "Dapagliflozin" in drug_names
        assert "Empagliflozin" in drug_names
        assert "Sitagliptin" in drug_names
        assert "Glimepiride" in drug_names

    def test_cate_values_negative(self, clinical_patient: PatientContext) -> None:
        """All drugs should show HbA1c reduction (negative delta)."""
        from app.bridge.causal_connector import estimate_all_treatments

        estimates = estimate_all_treatments(clinical_patient)
        for est in estimates:
            assert est.cate_hba1c_delta < 0, (
                f"{est.drug_name} should reduce HbA1c but got {est.cate_hba1c_delta}"
            )

    def test_confidence_intervals_valid(self, clinical_patient: PatientContext) -> None:
        from app.bridge.causal_connector import estimate_all_treatments

        estimates = estimate_all_treatments(clinical_patient)
        for est in estimates:
            assert est.ci_lower <= est.cate_hba1c_delta <= est.ci_upper, (
                f"{est.drug_name}: CATE {est.cate_hba1c_delta} outside "
                f"CI [{est.ci_lower}, {est.ci_upper}]"
            )


# =========================================================================
# Fusion Safety Gate Tests
# =========================================================================


class TestFusionSafetyGate:
    """Test the Evidence Fusion Layer's contraindication logic."""

    def _get_test_estimates(self) -> list[CausalEstimatePayload]:
        """Pre-built estimates for fusion testing."""
        return [
            CausalEstimatePayload(
                drug_name="Dapagliflozin",
                cate_hba1c_delta=-0.86,
                ci_lower=-1.04,
                ci_upper=-0.68,
                overlap_status="SUPPORTED",
            ),
            CausalEstimatePayload(
                drug_name="Empagliflozin",
                cate_hba1c_delta=-0.84,
                ci_lower=-1.03,
                ci_upper=-0.65,
                overlap_status="SUPPORTED",
            ),
            CausalEstimatePayload(
                drug_name="Sitagliptin",
                cate_hba1c_delta=-0.68,
                ci_lower=-0.83,
                ci_upper=-0.53,
                overlap_status="SUPPORTED",
            ),
            CausalEstimatePayload(
                drug_name="Glimepiride",
                cate_hba1c_delta=-1.10,
                ci_lower=-1.30,
                ci_upper=-0.90,
                overlap_status="SUPPORTED",
            ),
        ]

    def test_dapagliflozin_flagged_egfr_below_45(
        self, clinical_patient: PatientContext
    ) -> None:
        """Dapagliflozin should be flagged when eGFR < 45."""
        from app.rag.fusion import fuse

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        response = fuse(clinical_patient, rag, self._get_test_estimates())

        assert "Dapagliflozin" in response.flagged_options, (
            f"Dapagliflozin should be flagged. Eligible: {response.eligible_options}, "
            f"Flagged: {response.flagged_options}"
        )
        assert "Dapagliflozin" not in response.eligible_options

    def test_empagliflozin_eligible_egfr_38(
        self, clinical_patient: PatientContext
    ) -> None:
        """Empagliflozin should remain eligible when eGFR=38 (>=30 cutoff)."""
        from app.rag.fusion import fuse

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        response = fuse(clinical_patient, rag, self._get_test_estimates())

        assert "Empagliflozin" in response.eligible_options, (
            f"Empagliflozin should be eligible with eGFR=38. "
            f"Eligible: {response.eligible_options}"
        )

    def test_sitagliptin_eligible_dose_adjusted(
        self, clinical_patient: PatientContext
    ) -> None:
        """Sitagliptin should be eligible with dose adjustment for eGFR 30-<45."""
        from app.rag.fusion import fuse

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        response = fuse(clinical_patient, rag, self._get_test_estimates())

        assert "Sitagliptin" in response.eligible_options
        # Should mention 50 mg dose in warnings
        dose_warnings = [
            w for w in response.safety_warnings if "Sitagliptin" in w and "50" in w
        ]
        assert len(dose_warnings) > 0, (
            f"Expected 50 mg dose warning for Sitagliptin. "
            f"Warnings: {response.safety_warnings}"
        )

    def test_glimepiride_eligible_with_warning(
        self, clinical_patient: PatientContext
    ) -> None:
        """Glimepiride should be eligible but with hypoglycemia warning."""
        from app.rag.fusion import fuse

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        response = fuse(clinical_patient, rag, self._get_test_estimates())

        assert "Glimepiride" in response.eligible_options
        hypo_warnings = [
            w for w in response.safety_warnings if "Glimepiride" in w
        ]
        assert len(hypo_warnings) > 0

    def test_causal_estimates_present(
        self, clinical_patient: PatientContext
    ) -> None:
        """All causal CATE outputs should be present in the response."""
        from app.rag.fusion import fuse

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        response = fuse(clinical_patient, rag, self._get_test_estimates())

        assert len(response.causal_estimates) == 4
        drug_names = {e.drug_name for e in response.causal_estimates}
        assert drug_names == {
            "Dapagliflozin",
            "Empagliflozin",
            "Sitagliptin",
            "Glimepiride",
        }

    def test_hard_contraindication_override(self) -> None:
        """If causal shows favourable effect but guidelines block, guideline wins."""
        from app.rag.fusion import fuse

        patient = PatientContext(
            age=65,
            bmi=28.0,
            egfr=25.0,  # Very low eGFR
            hba1c=9.0,
            prior_hypo=False,
            current_meds=["Metformin"],
        )

        estimates = [
            CausalEstimatePayload(
                drug_name="Empagliflozin",
                cate_hba1c_delta=-0.90,  # Favourable
                ci_lower=-1.10,
                ci_upper=-0.70,
                overlap_status="SUPPORTED",
            ),
        ]

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        response = fuse(patient, rag, estimates)

        assert "Empagliflozin" in response.flagged_options
        assert "HARD CONTRAINDICATION" in response.flagged_options["Empagliflozin"]
        assert "Empagliflozin" not in response.eligible_options


# =========================================================================
# Synthesized Advice Tests
# =========================================================================


class TestSynthesizedAdvice:
    """Test the synthesized clinical advice output."""

    def test_advice_contains_disclaimer(
        self, clinical_patient: PatientContext
    ) -> None:
        from app.rag.fusion import fuse

        rag = RAGEvidencePayload(
            query="test",
            status="SUCCESS",
            chunks=[],
            retrieval_latency_ms=10.0,
        )

        estimates = [
            CausalEstimatePayload(
                drug_name="Sitagliptin",
                cate_hba1c_delta=-0.68,
                ci_lower=-0.83,
                ci_upper=-0.53,
                overlap_status="SUPPORTED",
            ),
        ]

        response = fuse(clinical_patient, rag, estimates)

        assert "research prototype" in response.synthesized_advice.lower()
        assert "not for unsupervised clinical use" in response.synthesized_advice.lower()
