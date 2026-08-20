"""The skew detector must fire on a deliberately wrong transform."""

from __future__ import annotations

import numpy as np

from mlserv.api.serving import predict_proba_1
from mlserv.failures.training_serving_skew import (
    detect_batch_scale_skew,
    detect_feature_swap_skew,
    detect_pooled_scale_skew,
    in_range_swap_frame,
    probability_parity,
    serve_with_swapped_age_income,
)


def test_parity_on_correct_path_does_not_fire(train_result):
    X = train_result.X_val
    correct = predict_proba_1(train_result.pipeline, X)
    report = probability_parity(
        train_result.pipeline, X, correct, tolerance=1e-10, kind="identity"
    )
    assert report.detector_fired is False
    assert report.max_abs_diff < 1e-12


def test_pooled_scaler_detector_fires(train_result):
    from mlserv.failures.data_drift import shift_income

    X_serve = shift_income(train_result.X_val, log_shift=0.7)
    report = detect_pooled_scale_skew(
        train_result.pipeline,
        train_result.X_train,
        X_serve,
        tolerance=1e-6,
    )
    assert report.kind == "pooled_scaler"
    assert report.detector_fired is True
    assert report.max_abs_diff > 1e-4


def test_batch_scaler_detector_fires(train_result):
    report = detect_batch_scale_skew(
        train_result.pipeline, train_result.X_val, tolerance=1e-6
    )
    assert report.detector_fired is True
    assert report.max_abs_diff > 1e-4


def test_feature_swap_detector_fires(train_result):
    frame = in_range_swap_frame(80, seed=2026)
    report = detect_feature_swap_skew(
        train_result.pipeline, frame, tolerance=1e-6
    )
    assert report.detector_fired is True
    assert report.max_abs_diff > 0.01


def test_api_would_still_accept_swapped_values(train_result, client):
    """Pydantic checks names and ranges, not whether age and income were swapped."""
    X = in_range_swap_frame(1, seed=2026)
    swapped = serve_with_swapped_age_income(train_result.pipeline, X)
    correct = predict_proba_1(train_result.pipeline, X)
    assert abs(float(swapped[0]) - float(correct[0])) > 1e-6
    row = X.iloc[0]
    payload = {
        "age": float(row["income"]),
        "income": float(row["age"]),
        "credit_score": float(row["credit_score"]),
        "segment": str(row["segment"]),
    }
    response = client.post("/predict-proba", json=payload)
    assert response.status_code == 200
    expected = float(swapped[0])
    np.testing.assert_allclose(response.json()["probability_1"], expected, atol=1e-10)
