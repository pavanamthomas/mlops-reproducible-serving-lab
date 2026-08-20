"""Monitoring summaries on known missingness and known shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlserv.failures.data_drift import shift_income
from mlserv.monitoring.summaries import (
    delayed_performance,
    distribution_change,
    feature_summaries,
    missingness,
    prediction_distribution,
)


def test_missingness_is_zero_on_complete_val(train_result):
    rates = missingness(train_result.X_val)
    for name, rate in rates.items():
        assert rate == 0.0, name


def test_missingness_flags_absent_column(train_result):
    dropped = train_result.X_val.drop(columns=["income"])
    rates = missingness(dropped)
    assert rates["income"] == 1.0


def test_feature_summaries_match_pandas(train_result):
    summaries = {item.name: item for item in feature_summaries(train_result.X_val)}
    age = train_result.X_val["age"].to_numpy(dtype=float)
    np.testing.assert_allclose(summaries["age"].mean, age.mean(), atol=1e-12)
    np.testing.assert_allclose(summaries["age"].p50, np.quantile(age, 0.50), atol=1e-12)


def test_distribution_change_detects_income_shift(train_result):
    shifted = shift_income(train_result.X_val, log_shift=0.9)
    reports = {item.feature: item for item in distribution_change(train_result.X_val, shifted)}
    assert reports["income"].psi > reports["age"].psi
    assert reports["income"].ks_statistic > 0.2


def test_prediction_distribution_bounds(train_result):
    summary = prediction_distribution(train_result.pipeline, train_result.X_val)
    assert summary.n == len(train_result.X_val)
    assert 0.0 <= summary.mean_p1 <= 1.0
    assert 0.0 <= summary.share_predicted_positive <= 1.0


def test_delayed_performance_uses_only_available_labels():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0])
    available = np.array([True, False, True, True])
    score = delayed_performance(y_true, y_pred, available)
    assert score.n_available == 3
    assert score.accuracy_available == 1.0
