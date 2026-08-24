"""Deliberate wrong transforms that still return a probability.

HTTP 200 does not identify equality with pipeline.predict_proba.
Swapped columns and pooled scalers are the constructed cases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlserv.api.serving import predict_proba_1
from mlserv.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEGMENT_VALUES, require_valid_frame

DEFAULT_SKEW_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SkewReport:
    """Comparison of correct Pipeline probabilities to a skewed transform."""

    max_abs_diff: float
    mean_abs_diff: float
    n: int
    detector_fired: bool
    kind: str
    tolerance: float


def probability_parity(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    other: NDArray[np.floating],
    *,
    tolerance: float = DEFAULT_SKEW_TOLERANCE,
    kind: str = "custom",
) -> SkewReport:
    """Compare Pipeline probabilities to another vector of P(y=1)."""
    correct = np.asarray(predict_proba_1(pipeline, frame), dtype=float).reshape(-1)
    other_arr = np.asarray(other, dtype=float).reshape(-1)
    if correct.shape != other_arr.shape:
        raise ValueError("probability vectors must have the same length")
    abs_diff = np.abs(correct - other_arr)
    max_abs = float(np.max(abs_diff))
    return SkewReport(
        max_abs_diff=max_abs,
        mean_abs_diff=float(np.mean(abs_diff)),
        n=int(correct.size),
        detector_fired=max_abs > tolerance,
        kind=kind,
        tolerance=tolerance,
    )


def serve_with_pooled_scaler(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_serve: pd.DataFrame,
) -> NDArray[np.float64]:
    """Scale numeric features with train+serve statistics, then apply the trained classifier.

    The encoder is the fitted one. Only the numeric scaler is wrong. This
    is a standard training-serving skew: production statistics leak into
    the transform.
    """
    X_tr = require_valid_frame(X_train)
    X_sv = require_valid_frame(X_serve)
    pooled = pd.concat([X_tr[list(NUMERIC_FEATURES)], X_sv[list(NUMERIC_FEATURES)]], axis=0)
    scaler = StandardScaler().fit(pooled)
    fitted_pre = pipeline.named_steps["preprocess"]
    encoder = fitted_pre.named_transformers_["cat"]
    clf = pipeline.named_steps["clf"]
    x_num = scaler.transform(X_sv[list(NUMERIC_FEATURES)])
    x_cat = encoder.transform(X_sv[list(CATEGORICAL_FEATURES)])
    xt = np.hstack([x_num, x_cat])
    return np.asarray(clf.predict_proba(xt)[:, 1], dtype=float)


def serve_with_batch_scaler(
    pipeline: Pipeline,
    X_serve: pd.DataFrame,
) -> NDArray[np.float64]:
    """Fit a StandardScaler on the serving batch only (no training statistics)."""
    X_sv = require_valid_frame(X_serve)
    scaler = StandardScaler().fit(X_sv[list(NUMERIC_FEATURES)])
    fitted_pre = pipeline.named_steps["preprocess"]
    encoder = fitted_pre.named_transformers_["cat"]
    clf = pipeline.named_steps["clf"]
    x_num = scaler.transform(X_sv[list(NUMERIC_FEATURES)])
    x_cat = encoder.transform(X_sv[list(CATEGORICAL_FEATURES)])
    xt = np.hstack([x_num, x_cat])
    return np.asarray(clf.predict_proba(xt)[:, 1], dtype=float)


def in_range_swap_frame(n: int, seed: int = 2026) -> pd.DataFrame:
    """Rows where age and income both lie in [18, 90].

    A header-mapping swap then still passes the schema. That is the
    interesting case: HTTP 422 does not fire, but predictions change.
    Income values this small are allowed by the contract; they are not
    typical of the training DGP.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(int(seed))
    frame = pd.DataFrame(
        {
            "age": rng.uniform(20.0, 80.0, size=n),
            "income": rng.uniform(20.0, 80.0, size=n),
            "credit_score": rng.uniform(0.1, 0.9, size=n),
            "segment": rng.choice(list(SEGMENT_VALUES), size=n),
        }
    )
    return require_valid_frame(frame)


def serve_with_swapped_age_income(
    pipeline: Pipeline,
    frame: pd.DataFrame,
) -> NDArray[np.float64]:
    """CSV-header mapping bug: income values written into the age column and conversely.

    Column names remain valid. If both values stay inside schema bounds,
    Pydantic still accepts the JSON. The Pipeline then scales the wrong
    numbers.
    """
    X = require_valid_frame(frame).copy()
    age = X["age"].to_numpy()
    X["age"] = X["income"].to_numpy()
    X["income"] = age
    return np.asarray(predict_proba_1(pipeline, X), dtype=float)


def detect_pooled_scale_skew(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_serve: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_SKEW_TOLERANCE,
) -> SkewReport:
    other = serve_with_pooled_scaler(pipeline, X_train, X_serve)
    return probability_parity(
        pipeline, X_serve, other, tolerance=tolerance, kind="pooled_scaler"
    )


def detect_batch_scale_skew(
    pipeline: Pipeline,
    X_serve: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_SKEW_TOLERANCE,
) -> SkewReport:
    other = serve_with_batch_scaler(pipeline, X_serve)
    return probability_parity(
        pipeline, X_serve, other, tolerance=tolerance, kind="batch_scaler"
    )


def detect_feature_swap_skew(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_SKEW_TOLERANCE,
) -> SkewReport:
    other = serve_with_swapped_age_income(pipeline, frame)
    return probability_parity(
        pipeline, frame, other, tolerance=tolerance, kind="age_income_swap"
    )
