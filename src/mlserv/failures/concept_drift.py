"""Concept drift with delayed labels.

Problem: P(Y | X) can change while P(X) looks stable. Labels used to
score the model often arrive later than predictions.
Assumptions: the laboratory changes the credit-score slope in the DGP;
until a boolean mask says the label has arrived, accuracy is not
identified from the delayed sample.
Why this method: a known change in the DGP plus an explicit delay mask
makes the identification statement sharp.
Alternative: sequential detectors (ADWIN, Page-Hinkley). Not
implemented; they would require a different false-positive analysis.
What can go wrong: reading a stable prediction histogram as evidence
that the concept is unchanged; scoring delayed accuracy on the full
window including missing labels filled with zeros.
Independent check: after the slope sign flips, accuracy on arrived
labels falls relative to the original val split on this DGP.
Can conclude: under this DGP change and this delay, these scores were
observed.
Cannot conclude: a monitoring product, or that prediction-distribution
shift is a sufficient proxy for concept drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from mlserv.data import generate_dataset
from mlserv.evaluate import EvalResult, evaluate_classifier


@dataclass(frozen=True)
class DelayedScore:
    n_total: int
    n_available: int
    accuracy_available: float
    delay_fraction: float


def concept_shifted_dataset(
    n: int,
    seed: int,
    *,
    coef_credit: float = -2.4,
):
    """Same feature draw family, different label slope on credit_score."""
    return generate_dataset(n, seed=seed, coef_credit=coef_credit)


def delayed_label_mask(n: int, delay_fraction: float, seed: int) -> NDArray[np.bool_]:
    """Which labels have arrived. ``delay_fraction`` is P(label missing)."""
    if not 0.0 <= delay_fraction < 1.0:
        raise ValueError("delay_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    return rng.random(n) >= delay_fraction


def delayed_accuracy(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    available: NDArray[np.bool_],
) -> DelayedScore:
    """Accuracy using only rows whose labels have arrived."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_pred, dtype=int).reshape(-1)
    mask = np.asarray(available, dtype=bool).reshape(-1)
    if y.size != p.size or y.size != mask.size:
        raise ValueError("y_true, y_pred, and available must have the same length")
    n_avail = int(mask.sum())
    if n_avail == 0:
        acc = float("nan")
    else:
        acc = float(accuracy_score(y[mask], p[mask]))
    return DelayedScore(
        n_total=int(y.size),
        n_available=n_avail,
        accuracy_available=acc,
        delay_fraction=1.0 - n_avail / y.size,
    )


def score_on_shifted_concept(
    pipeline: Pipeline,
    n: int,
    seed: int,
    *,
    coef_credit: float = -2.4,
) -> EvalResult:
    batch = concept_shifted_dataset(n, seed, coef_credit=coef_credit)
    return evaluate_classifier(pipeline, batch.X, batch.y)


def prediction_mean(
    pipeline: Pipeline,
    frame: pd.DataFrame,
) -> float:
    """Mean predicted P(y=1). A stable mean does not identify a stable concept."""
    return float(np.mean(pipeline.predict_proba(frame)[:, 1]))
