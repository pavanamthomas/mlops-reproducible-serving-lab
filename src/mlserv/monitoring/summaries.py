"""Simple monitoring statistics for the synthetic serving laboratory.

Problem: after a model is served, one still needs numbers on missingness,
feature location, distribution change, prediction histograms, and
delayed labelled performance.
Assumptions: windows are batches, not streaming; tests are unadjusted;
labels may be missing. These are descriptive statistics, not a
monitoring product.
Why this method: each quantity is one function that a test can check
on a known shift.
Alternative: Prometheus + Grafana + a drift service. Out of scope; see
``docs/monitoring_limits.md``.
What can go wrong: alerting on KS p-values as if they were calibrated
Type I errors under dependence; treating delayed accuracy as online
skill.
Independent check: ``tests/test_monitoring.py``.
Can conclude: these summaries of these two batches.
Cannot conclude: operational readiness or a false-alarm warranty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mlserv.failures.concept_drift import DelayedScore, delayed_accuracy
from mlserv.failures.data_drift import UnivariateShift, numeric_shift_report
from mlserv.schema import FEATURE_ORDER, NUMERIC_FEATURES


@dataclass(frozen=True)
class FeatureSummary:
    name: str
    n: int
    missing_rate: float
    mean: float
    std: float
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class PredictionSummary:
    n: int
    mean_p1: float
    std_p1: float
    share_predicted_positive: float
    p10: float
    p50: float
    p90: float


def missingness(frame: pd.DataFrame) -> dict[str, float]:
    """Fraction missing per column, including columns that should exist."""
    rates: dict[str, float] = {}
    n = max(len(frame), 1)
    for name in FEATURE_ORDER:
        if name not in frame.columns:
            rates[name] = 1.0
        else:
            rates[name] = float(frame[name].isna().mean())
    for name in frame.columns:
        if name not in rates:
            rates[str(name)] = float(frame[name].isna().mean()) if n else 1.0
    return rates


def feature_summaries(frame: pd.DataFrame) -> list[FeatureSummary]:
    """Location and spread of numeric features. Categorical columns are skipped."""
    out: list[FeatureSummary] = []
    n = int(len(frame))
    for name in NUMERIC_FEATURES:
        if name not in frame.columns:
            out.append(
                FeatureSummary(
                    name=name,
                    n=n,
                    missing_rate=1.0,
                    mean=float("nan"),
                    std=float("nan"),
                    p10=float("nan"),
                    p50=float("nan"),
                    p90=float("nan"),
                )
            )
            continue
        series = pd.to_numeric(frame[name], errors="coerce")
        missing_rate = float(series.isna().mean()) if n else 1.0
        values = series.dropna().to_numpy(dtype=float)
        if values.size == 0:
            out.append(
                FeatureSummary(
                    name=name,
                    n=n,
                    missing_rate=missing_rate,
                    mean=float("nan"),
                    std=float("nan"),
                    p10=float("nan"),
                    p50=float("nan"),
                    p90=float("nan"),
                )
            )
            continue
        out.append(
            FeatureSummary(
                name=name,
                n=n,
                missing_rate=missing_rate,
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                p10=float(np.quantile(values, 0.10)),
                p50=float(np.quantile(values, 0.50)),
                p90=float(np.quantile(values, 0.90)),
            )
        )
    return out


def distribution_change(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> list[UnivariateShift]:
    """PSI and KS for each numeric feature. Not a sequential monitor."""
    reports: list[UnivariateShift] = []
    for name in NUMERIC_FEATURES:
        reports.append(
            numeric_shift_report(reference[name], current[name], feature=name)
        )
    return reports


def prediction_distribution(
    pipeline: Pipeline,
    frame: pd.DataFrame,
) -> PredictionSummary:
    p1 = np.asarray(pipeline.predict_proba(frame)[:, 1], dtype=float)
    return PredictionSummary(
        n=int(p1.size),
        mean_p1=float(np.mean(p1)),
        std_p1=float(np.std(p1, ddof=1)) if p1.size > 1 else 0.0,
        share_predicted_positive=float(np.mean(p1 >= 0.5)),
        p10=float(np.quantile(p1, 0.10)),
        p50=float(np.quantile(p1, 0.50)),
        p90=float(np.quantile(p1, 0.90)),
    )


def delayed_performance(
    y_true: NDArray,
    y_pred: NDArray,
    available: NDArray,
) -> DelayedScore:
    return delayed_accuracy(y_true, y_pred, available)
