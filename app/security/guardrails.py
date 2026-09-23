"""Deterministic input validation and scope enforcement guardrails.

All checks are rule-based (no ML) to guarantee reproducibility and
auditability in a clinical decision-support context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of a guardrail check."""

    passed: bool
    status: Literal["OK", "OUT_OF_SCOPE", "INVALID_INPUT", "REDACTED"]
    message: str
    sanitized_text: str | None = None


# ---------------------------------------------------------------------------
# Scope Filter
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btype\s*1\s*diabetes\b", re.IGNORECASE),
    re.compile(r"\bT1D(?:M)?\b", re.IGNORECASE),
    re.compile(r"\bjuvenile\s*diabetes\b", re.IGNORECASE),
    re.compile(r"\bpediatric\b|\bpaediatric\b|\bchild(?:ren)?\b|\binfant\b|\bneonat\w*\b", re.IGNORECASE),
    re.compile(r"\bpregnancy\b|\bpregnant\b|\bgestational\b|\bmaternal\b", re.IGNORECASE),
    re.compile(r"\bDKA\b|\bdiabetic\s*ketoacidosis\b", re.IGNORECASE),
    re.compile(r"\bHHS\b|\bhyperosmolar\b", re.IGNORECASE),
    re.compile(r"\bseizure\b|\bunconscious(?:ness)?\b|\bcoma\b", re.IGNORECASE),
    re.compile(r"\bemergency\b|\bICU\b|\bcritical\s*care\b", re.IGNORECASE),
]


def check_scope(query: str) -> GuardrailResult:
    """Return OUT_OF_SCOPE if the query mentions excluded clinical domains."""
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        match = pattern.search(query)
        if match:
            return GuardrailResult(
                passed=False,
                status="OUT_OF_SCOPE",
                message=(
                    f"Query is out of scope for this system. "
                    f"Matched exclusion pattern: '{match.group()}'. "
                    f"This system only supports adult Type 2 Diabetes management "
                    f"in non-emergency outpatient settings."
                ),
            )
    return GuardrailResult(passed=True, status="OK", message="Query is in scope.")


# ---------------------------------------------------------------------------
# Sensitive Identifier Redaction
# ---------------------------------------------------------------------------

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Indian mobile numbers: 10 digits starting with 6-9, optional +91 / 0 prefix
    (re.compile(r"(?:\+91[\s-]?|0)?[6-9]\d{9}\b"), "[Phone Redacted]"),
    # PAN number: 5 letters + 4 digits + 1 letter (Indian tax ID)
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN Redacted]"),
    # Aadhaar-like: 12 consecutive digits (possibly space/dash separated in groups of 4)
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[Aadhaar Redacted]"),
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[Email Redacted]"),
]


def redact_identifiers(text: str) -> GuardrailResult:
    """Redact sensitive personal identifiers from the input text."""
    sanitized = text
    redacted = False
    for pattern, placeholder in _REDACTION_PATTERNS:
        if pattern.search(sanitized):
            redacted = True
            sanitized = pattern.sub(placeholder, sanitized)

    if redacted:
        return GuardrailResult(
            passed=True,
            status="REDACTED",
            message="Sensitive identifiers were redacted from input.",
            sanitized_text=sanitized,
        )
    return GuardrailResult(
        passed=True,
        status="OK",
        message="No sensitive identifiers detected.",
        sanitized_text=text,
    )


# ---------------------------------------------------------------------------
# Plausibility Validator
# ---------------------------------------------------------------------------

def validate_clinical_inputs(
    *,
    hba1c: float | None = None,
    egfr: float | None = None,
    fasting_glucose: float | None = None,
    age: int | None = None,
) -> GuardrailResult:
    """Reject physiologically implausible clinical values."""
    violations: list[str] = []

    if hba1c is not None and not (4.0 <= hba1c <= 20.0):
        violations.append(f"HbA1c={hba1c}% is outside plausible range [4.0, 20.0].")

    if egfr is not None and not (0.0 <= egfr <= 150.0):
        violations.append(f"eGFR={egfr} is outside plausible range [0, 150].")

    if fasting_glucose is not None and not (20.0 <= fasting_glucose <= 800.0):
        violations.append(
            f"Fasting glucose={fasting_glucose} mg/dL is outside plausible range [20, 800]."
        )

    if age is not None and age < 18:
        violations.append(
            f"Age={age} is below 18. Pediatric patients are out of scope."
        )

    if violations:
        return GuardrailResult(
            passed=False,
            status="INVALID_INPUT",
            message="Clinical input validation failed: " + "; ".join(violations),
        )
    return GuardrailResult(
        passed=True, status="OK", message="Clinical inputs are plausible."
    )


# ---------------------------------------------------------------------------
# Combined Pre-Processing Guard
# ---------------------------------------------------------------------------

def run_all_guards(
    query: str,
    *,
    hba1c: float | None = None,
    egfr: float | None = None,
    fasting_glucose: float | None = None,
    age: int | None = None,
) -> GuardrailResult:
    """Execute all guardrail checks in sequence.

    Order: scope -> identifier redaction -> plausibility.
    Short-circuits on the first hard failure.
    """
    # 1. Scope check
    scope_result = check_scope(query)
    if not scope_result.passed:
        return scope_result

    # 2. Redact identifiers
    redaction_result = redact_identifiers(query)
    working_text = redaction_result.sanitized_text or query

    # 3. Clinical plausibility
    plausibility_result = validate_clinical_inputs(
        hba1c=hba1c, egfr=egfr, fasting_glucose=fasting_glucose, age=age
    )
    if not plausibility_result.passed:
        return plausibility_result

    # Return redaction result if identifiers were sanitized
    if redaction_result.status == "REDACTED":
        return redaction_result

    return GuardrailResult(
        passed=True,
        status="OK",
        message="All guardrail checks passed.",
        sanitized_text=working_text,
    )
