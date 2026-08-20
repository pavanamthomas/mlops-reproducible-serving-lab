"""Schema drift: missing fields, new fields, and type changes.

Problem: the production record shape can change while the model artifact
still expects schema_version 1.0.
Assumptions: the laboratory schema is the tuple ``FEATURE_ORDER``;
unexpected columns are violations, not extras to drop.
Why this method: compare column sets and dtypes before calling the
Pipeline, so a 422/ValueError happens instead of a silent column drop.
Alternative: a schema registry (Protobuf, Avro). Same estimand, more
infrastructure.
What can go wrong: ``remainder='drop'`` in ColumnTransformer hides new
fields if they never reach this check.
Independent check: ``tests/test_failures.py`` for missing, extra, and
bad category.
Can conclude: this record does not match schema 1.0.
Cannot conclude: how the upstream producer should version its events.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlserv.schema import FEATURE_ORDER, validate_frame


@dataclass(frozen=True)
class SchemaDriftReport:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    other: tuple[str, ...]
    ok: bool


def schema_drift_report(frame: pd.DataFrame) -> SchemaDriftReport:
    violations = validate_frame(frame)
    missing = tuple(v.column for v in violations if v.reason == "missing")
    unexpected = tuple(v.column for v in violations if v.reason == "unexpected")
    other = tuple(f"{v.column}:{v.reason}" for v in violations if v.reason not in {"missing", "unexpected"})
    return SchemaDriftReport(
        missing=missing,
        unexpected=unexpected,
        other=other,
        ok=len(violations) == 0,
    )


def drop_column(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in frame.columns:
        raise ValueError(f"{name} is not in the frame")
    return frame.drop(columns=[name])


def add_column(frame: pd.DataFrame, name: str, value: object) -> pd.DataFrame:
    out = frame.copy()
    out[name] = value
    return out


def required_columns() -> tuple[str, ...]:
    return FEATURE_ORDER
