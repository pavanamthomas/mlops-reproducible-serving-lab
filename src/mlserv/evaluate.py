"""Hold-out evaluation against a dummy baseline.

Problem: a trained model needs numbers on a split that was not used to
fit the Pipeline, and those numbers need a baseline so they are not
read as an absolute claim of skill.
Assumptions: labels are binary in {0, 1}; ``predict_proba`` is defined;
the val split is iid from the same DGP as training unless a drift
experiment says otherwise.
Why this method: accuracy, log loss, ROC AUC, and Brier score are
standard proper or ranking scores for a probabilistic classifier.
Alternative: expected calibration error, decision-curve analysis. Not
implemented; they would answer a different question.
What can go wrong: quoting val accuracy as a production SLO; claiming
the logistic model is better than the dummy outside this DGP.
Independent check: DummyClassifier is fitted on the same y_train;
tests check that both metric dicts are populated.
Can conclude: on this synthetic val split, these scores were observed.
Cannot conclude: generalization to a shifted or real population.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    log_loss: float
    roc_auc: float
    brier: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_classifier(model: object, X: pd.DataFrame, y: ArrayLike) -> EvalResult:
    """Scores of a fitted classifier on a labelled frame."""
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if y_arr.size == 0:
        raise ValueError("y must be non-empty")
    if set(np.unique(y_arr)).difference({0, 1}):
        raise ValueError("labels must be in {0, 1}")
    pred = np.asarray(model.predict(X), dtype=int).reshape(-1)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    p1 = proba[:, 1] if proba.ndim == 2 else proba
    acc = float(accuracy_score(y_arr, pred))
    ll = float(log_loss(y_arr, p1, labels=[0, 1]))
    brier = float(brier_score_loss(y_arr, p1))
    if np.unique(y_arr).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_arr, p1))
    return EvalResult(accuracy=acc, log_loss=ll, roc_auc=auc, brier=brier, n=int(y_arr.size))
