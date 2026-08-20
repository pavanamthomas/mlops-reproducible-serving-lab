"""Regression: extreme DGP rows keep the predicted class the model should assign."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlserv.api.serving import predict_proba_1
from mlserv.data import true_probability
from mlserv.schema import require_valid_frame


def _frame(age: float, income: float, credit: float, segment: str) -> pd.DataFrame:
    return require_valid_frame(
        pd.DataFrame(
            [{"age": age, "income": income, "credit_score": credit, "segment": segment}]
        )
    )


def test_extreme_high_row_has_high_probability(train_result):
    frame = _frame(88.0, 250000.0, 0.97, "C")
    p_hat = float(predict_proba_1(train_result.pipeline, frame)[0])
    p_true = float(true_probability(frame)[0])
    assert p_true > 0.85
    assert p_hat > 0.70
    assert abs(p_hat - p_true) < 0.35


def test_extreme_low_row_has_low_probability(train_result):
    frame = _frame(22.0, 12000.0, 0.12, "A")
    p_hat = float(predict_proba_1(train_result.pipeline, frame)[0])
    p_true = float(true_probability(frame)[0])
    assert p_true < 0.20
    assert p_hat < 0.35


def test_session_fixture_prediction_is_stable(train_result):
    frame = _frame(45.0, 40000.0, 0.6, "B")
    a = float(predict_proba_1(train_result.pipeline, frame)[0])
    b = float(predict_proba_1(train_result.pipeline, frame)[0])
    np.testing.assert_allclose(a, b, atol=1e-15)
    assert 0.0 < a < 1.0
