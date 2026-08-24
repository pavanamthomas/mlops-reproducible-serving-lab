"""iid logistic DGP used by training, serving, and drift checks.

Labels are Bernoulli given age, log income, credit score, and segment.
Default seed 2026.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlserv._rng import DEFAULT_SEED, get_rng
from mlserv.schema import (
    AGE_MAX,
    AGE_MIN,
    CREDIT_MAX,
    CREDIT_MIN,
    FEATURE_ORDER,
    INCOME_MAX,
    INCOME_MIN,
    SEGMENT_VALUES,
    require_valid_frame,
)

DGP_NAME = "synthetic_logistic_segments_v1"

# Logit coefficients on the raw (unscaled) features. These are the
# laboratory's true parameters, not estimated quantities.
INTERCEPT = -6.0
COEF_AGE = 0.03
COEF_LOG_INCOME = 0.25
COEF_CREDIT = 2.4
COEF_SEGMENT = {"A": 0.0, "B": 0.35, "C": 0.85}


@dataclass(frozen=True)
class SyntheticBatch:
    """Features, labels, and the true conditional probability."""

    X: pd.DataFrame
    y: NDArray[np.int_]
    p_true: NDArray[np.float64]


def true_logit(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Linear predictor of the DGP. Identified here because we wrote it."""
    ordered = frame.loc[:, list(FEATURE_ORDER)]
    age = ordered["age"].to_numpy(dtype=float)
    income = ordered["income"].to_numpy(dtype=float)
    credit = ordered["credit_score"].to_numpy(dtype=float)
    segment = ordered["segment"].astype(str).to_numpy()
    logit = (
        INTERCEPT
        + COEF_AGE * age
        + COEF_LOG_INCOME * np.log(income)
        + COEF_CREDIT * credit
        + np.array([COEF_SEGMENT[str(s)] for s in segment], dtype=float)
    )
    return logit


def true_probability(frame: pd.DataFrame) -> NDArray[np.float64]:
    logit = true_logit(frame)
    return 1.0 / (1.0 + np.exp(-logit))


def generate_features(
    n: int,
    seed: int | np.random.Generator | None = DEFAULT_SEED,
) -> pd.DataFrame:
    """Draw features inside the schema bounds."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = get_rng(seed)
    age = rng.uniform(AGE_MIN, AGE_MAX, size=n)
    income = np.clip(np.exp(rng.normal(10.4, 0.55, size=n)), INCOME_MIN, INCOME_MAX)
    credit = np.clip(rng.beta(5.0, 2.2, size=n), CREDIT_MIN, CREDIT_MAX)
    segment = rng.choice(SEGMENT_VALUES, size=n, p=np.array([0.50, 0.30, 0.20]))
    frame = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit,
            "segment": segment,
        }
    )
    return require_valid_frame(frame)


def generate_dataset(
    n: int,
    seed: int | np.random.Generator | None = DEFAULT_SEED,
    *,
    coef_credit: float | None = None,
) -> SyntheticBatch:
    """Draw (X, y) from the logistic DGP.

    ``coef_credit`` overrides the DGP credit-score slope. Concept-drift
    experiments pass a different value; training uses the default.
    """
    rng = get_rng(seed)
    X = generate_features(n, seed=rng)
    p = true_probability(X)
    if coef_credit is not None:
        logit = true_logit(X)
        logit = logit - COEF_CREDIT * X["credit_score"].to_numpy(dtype=float)
        logit = logit + float(coef_credit) * X["credit_score"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p).astype(int)
    return SyntheticBatch(X=X, y=y, p_true=p)
