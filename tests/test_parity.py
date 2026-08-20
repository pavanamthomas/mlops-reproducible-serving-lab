"""Serving must call the fitted Pipeline; not a reimplemented transform."""

from __future__ import annotations

import numpy as np

from mlserv.api.serving import frame_from_records, predict_label, predict_proba_1
from mlserv.schema import FEATURE_ORDER


def test_serving_matches_pipeline_predict_proba(train_result):
    X = train_result.X_val
    pipeline = train_result.pipeline
    p_pipe = pipeline.predict_proba(X)[:, 1]
    p_serve = predict_proba_1(pipeline, X)
    np.testing.assert_allclose(p_pipe, p_serve, atol=1e-12)
    y_pipe = pipeline.predict(X)
    y_serve = predict_label(pipeline, X)
    np.testing.assert_array_equal(y_pipe, y_serve)


def test_frame_from_records_restores_feature_order(train_result):
    row = train_result.X_val.iloc[0]
    scrambled = {
        "segment": row["segment"],
        "credit_score": float(row["credit_score"]),
        "income": float(row["income"]),
        "age": float(row["age"]),
    }
    frame = frame_from_records([scrambled])
    assert list(frame.columns) == list(FEATURE_ORDER)
    p_direct = train_result.pipeline.predict_proba(train_result.X_val.iloc[[0]])[:, 1]
    p_from_records = predict_proba_1(train_result.pipeline, frame)
    np.testing.assert_allclose(p_direct, p_from_records, atol=1e-12)


def test_column_permutation_of_dataframe_does_not_change_named_serving(train_result):
    X = train_result.X_val
    permuted = X[["segment", "age", "credit_score", "income"]]
    p0 = predict_proba_1(train_result.pipeline, X)
    p1 = predict_proba_1(train_result.pipeline, permuted)
    np.testing.assert_allclose(p0, p1, atol=1e-12)
