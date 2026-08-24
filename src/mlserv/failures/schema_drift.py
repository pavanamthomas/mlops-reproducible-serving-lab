"""Missing, extra, or retyped columns versus FEATURE_ORDER.

Fail before the Pipeline rather than dropping fields silently.
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
