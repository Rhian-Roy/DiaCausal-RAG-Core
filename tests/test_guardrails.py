"""Tests for deterministic safety guardrails.

Verifies:
- Out-of-scope queries (Type 1 diabetes, pregnancy, pediatric, emergency) are blocked.
- Sensitive identifiers (phone, PAN, Aadhaar, email) are redacted.
- Implausible clinical inputs are rejected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.security.guardrails import (
    GuardrailResult,
    check_scope,
    redact_identifiers,
    run_all_guards,
    validate_clinical_inputs,
)


# =========================================================================
# Scope Filter Tests
# =========================================================================


class TestScopeFilter:
    """Tests for the scope filter."""

    @pytest.mark.parametrize(
        "query",
        [
            "Treatment for Type 1 Diabetes in a 30 year old male",
            "T1DM management guidelines",
            "juvenile diabetes insulin regimen",
            "Pediatric diabetes management",
            "Diabetes management in pregnancy",
            "Gestational diabetes screening",
            "DKA management protocol",
            "Patient in hyperosmolar hyperglycemic state",
            "Seizure management in diabetic patient",
            "Patient is unconscious with low blood sugar",
            "Emergency management of diabetic coma",
            "ICU protocol for diabetic crisis",
        ],
    )
    def test_out_of_scope_blocked(self, query: str) -> None:
        result = check_scope(query)
        assert not result.passed
        assert result.status == "OUT_OF_SCOPE"

    @pytest.mark.parametrize(
        "query",
        [
            "Treatment options for Type 2 Diabetes with HbA1c 8.5%",
            "Sitagliptin dose adjustment for eGFR 38",
            "SGLT2 inhibitor efficacy in T2DM",
            "Metformin first line therapy for adult diabetes",
            "Dapagliflozin cardiovascular benefits",
        ],
    )
    def test_in_scope_passes(self, query: str) -> None:
        result = check_scope(query)
        assert result.passed
        assert result.status == "OK"


# =========================================================================
# Identifier Redaction Tests
# =========================================================================


class TestIdentifierRedaction:
    """Tests for sensitive identifier redaction."""

    def test_phone_number_redacted(self) -> None:
        result = redact_identifiers("Call patient at 9876543210 for followup")
        assert result.status == "REDACTED"
        assert "9876543210" not in result.sanitized_text
        assert "[Phone Redacted]" in result.sanitized_text

    def test_phone_with_country_code(self) -> None:
        result = redact_identifiers("Contact: +91 9876543210")
        assert result.status == "REDACTED"
        assert "[Phone Redacted]" in result.sanitized_text

    def test_pan_number_redacted(self) -> None:
        result = redact_identifiers("Patient PAN: ABCDE1234F")
        assert result.status == "REDACTED"
        assert "ABCDE1234F" not in result.sanitized_text
        assert "[PAN Redacted]" in result.sanitized_text

    def test_aadhaar_redacted(self) -> None:
        result = redact_identifiers("Aadhaar: 1234 5678 9012")
        assert result.status == "REDACTED"
        assert "[Aadhaar Redacted]" in result.sanitized_text

    def test_email_redacted(self) -> None:
        result = redact_identifiers("Send report to doctor@hospital.org")
        assert result.status == "REDACTED"
        assert "doctor@hospital.org" not in result.sanitized_text
        assert "[Email Redacted]" in result.sanitized_text

    def test_clean_text_unchanged(self) -> None:
        text = "Patient has Type 2 Diabetes with HbA1c 9.2%"
        result = redact_identifiers(text)
        assert result.status == "OK"
        assert result.sanitized_text == text


# =========================================================================
# Plausibility Validator Tests
# =========================================================================


class TestPlausibilityValidator:
    """Tests for clinical input plausibility checks."""

    def test_valid_inputs_pass(self) -> None:
        result = validate_clinical_inputs(hba1c=8.5, egfr=65.0, age=55)
        assert result.passed
        assert result.status == "OK"

    def test_hba1c_too_low(self) -> None:
        result = validate_clinical_inputs(hba1c=3.0)
        assert not result.passed
        assert result.status == "INVALID_INPUT"
        assert "HbA1c" in result.message

    def test_hba1c_too_high(self) -> None:
        result = validate_clinical_inputs(hba1c=21.0)
        assert not result.passed
        assert "HbA1c" in result.message

    def test_egfr_negative(self) -> None:
        result = validate_clinical_inputs(egfr=-5.0)
        assert not result.passed
        assert "eGFR" in result.message

    def test_egfr_too_high(self) -> None:
        result = validate_clinical_inputs(egfr=200.0)
        assert not result.passed

    def test_glucose_too_low(self) -> None:
        result = validate_clinical_inputs(fasting_glucose=10.0)
        assert not result.passed
        assert "glucose" in result.message.lower()

    def test_glucose_too_high(self) -> None:
        result = validate_clinical_inputs(fasting_glucose=900.0)
        assert not result.passed

    def test_pediatric_age_blocked(self) -> None:
        result = validate_clinical_inputs(age=12)
        assert not result.passed
        assert "Pediatric" in result.message


# =========================================================================
# Combined Guard Tests
# =========================================================================


class TestCombinedGuards:
    """Tests for the combined guardrail runner."""

    def test_clean_query_passes(self) -> None:
        result = run_all_guards(
            "Sitagliptin dose for eGFR 38",
            hba1c=8.5,
            egfr=38.0,
            age=55,
        )
        assert result.passed

    def test_out_of_scope_short_circuits(self) -> None:
        result = run_all_guards("Type 1 Diabetes management", hba1c=8.0, age=25)
        assert not result.passed
        assert result.status == "OUT_OF_SCOPE"

    def test_invalid_input_caught(self) -> None:
        result = run_all_guards("T2DM treatment", hba1c=2.0, age=55)
        assert not result.passed
        assert result.status == "INVALID_INPUT"

    def test_redaction_with_valid_inputs(self) -> None:
        result = run_all_guards(
            "Patient email: test@example.com needs T2DM treatment",
            hba1c=8.0,
            age=55,
        )
        assert result.passed
        assert result.status == "REDACTED"
        assert "[Email Redacted]" in result.sanitized_text
