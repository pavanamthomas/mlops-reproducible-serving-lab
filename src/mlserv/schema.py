"""Feature schema for the synthetic serving laboratory.

Problem: a serving contract must name the features, their order, their
types, and their admissible ranges so that a record can be rejected
before it is passed to a fitted Pipeline.
Assumptions: the DGP in ``mlserv.data`` draws inside these bounds; the
API Pydantic models encode the same bounds.
Why this method: a single Python source of truth is cheaper to check
than a separate training schema and a separate OpenAPI schema that can
diverge.
Alternative: JSON Schema or a protobuf contract. Either would work; they
add tooling without changing the estimand.
What can go wrong: the API range and this module drift apart; an unseen
category is silently one-hot encoded to zeros.
Independent check: ``tests/test_schemas.py`` compares Pydantic bounds to
these constants; the preprocessor uses ``handle_unknown='error'``.
Can conclude: a record that fails this schema is not a valid input to
the laboratory model.
Cannot conclude: that a record inside the bounds is drawn from any real
applicant population.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SCHEMA_VERSION = "1.0"

NUMERIC_FEATURES: tuple[str, ...] = ("age", "income", "credit_score")
CATEGORICAL_FEATURES: tuple[str, ...] = ("segment",)
FEATURE_ORDER: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SEGMENT_VALUES: tuple[str, ...] = ("A", "B", "C")

AGE_MIN, AGE_MAX = 18.0, 90.0
INCOME_MIN, INCOME_MAX = 1.0, 10_000_000.0
CREDIT_MIN, CREDIT_MAX = 0.0, 1.0


@dataclass(frozen=True)
class SchemaViolation:
    """One failed check against the recorded feature schema."""

    column: str
    reason: str


def validate_frame(frame: pd.DataFrame) -> list[SchemaViolation]:
    """Return schema violations; an empty list means the frame is admissible.

    Extra columns are reported. They are not silently dropped here, because
    dropping them in serving is a common source of unnoticed schema drift.
    """
    violations: list[SchemaViolation] = []
    columns = list(frame.columns)
    missing = [name for name in FEATURE_ORDER if name not in columns]
    unexpected = [name for name in columns if name not in FEATURE_ORDER]
    for name in missing:
        violations.append(SchemaViolation(name, "missing"))
    for name in unexpected:
        violations.append(SchemaViolation(name, "unexpected"))
    if missing:
        return violations

    age = pd.to_numeric(frame["age"], errors="coerce")
    income = pd.to_numeric(frame["income"], errors="coerce")
    credit = pd.to_numeric(frame["credit_score"], errors="coerce")
    if age.isna().any():
        violations.append(SchemaViolation("age", "non-numeric"))
    elif ((age < AGE_MIN) | (age > AGE_MAX)).any():
        violations.append(SchemaViolation("age", "out-of-range"))
    if income.isna().any():
        violations.append(SchemaViolation("income", "non-numeric"))
    elif ((income < INCOME_MIN) | (income > INCOME_MAX)).any():
        violations.append(SchemaViolation("income", "out-of-range"))
    if credit.isna().any():
        violations.append(SchemaViolation("credit_score", "non-numeric"))
    elif ((credit < CREDIT_MIN) | (credit > CREDIT_MAX)).any():
        violations.append(SchemaViolation("credit_score", "out-of-range"))

    segment = frame["segment"].astype(str)
    if (~segment.isin(SEGMENT_VALUES)).any():
        violations.append(SchemaViolation("segment", "unseen-category"))
    return violations


def require_valid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if the frame violates the schema."""
    violations = validate_frame(frame)
    if violations:
        detail = "; ".join(f"{item.column}:{item.reason}" for item in violations)
        raise ValueError(f"schema {SCHEMA_VERSION} violated: {detail}")
    return frame.loc[:, list(FEATURE_ORDER)]
