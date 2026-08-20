"""Unit tests for the sklearn preprocessor and feature order."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mlserv.pipeline.preprocess import (
    build_model_pipeline,
    build_preprocessor,
    feature_names_in_order,
)
from mlserv.schema import FEATURE_ORDER, SEGMENT_VALUES


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30.0, 50.0, 70.0],
            "income": [20000.0, 40000.0, 80000.0],
            "credit_score": [0.2, 0.5, 0.9],
            "segment": ["A", "B", "C"],
        }
    )


def test_preprocessor_uses_named_columns_not_position():
    frame = _toy_frame()
    pre = build_preprocessor()
    pre.fit(frame)
    reordered = frame[["segment", "credit_score", "income", "age"]]
    xt = pre.transform(frame)
    xt_reordered = pre.transform(reordered)
    np.testing.assert_allclose(xt, xt_reordered)


def test_pipeline_records_training_feature_order():
    frame = _toy_frame()
    y = np.array([0, 1, 1])
    pipe = build_model_pipeline(random_state=2026)
    pipe.fit(frame, y)
    assert feature_names_in_order(pipe) == FEATURE_ORDER


def test_unseen_category_raises():
    frame = _toy_frame()
    pre = build_preprocessor()
    pre.fit(frame)
    bad = frame.copy()
    bad.loc[0, "segment"] = "Z"
    with pytest.raises(ValueError):
        pre.transform(bad)


def test_feature_names_in_order_requires_fit():
    pipe = build_model_pipeline()
    with pytest.raises(ValueError, match="feature_names_in_"):
        feature_names_in_order(pipe)


def test_numeric_means_are_zero_after_scale():
    frame = pd.concat([_toy_frame()] * 20, ignore_index=True)
    pre = build_preprocessor()
    xt = pre.fit_transform(frame)
    # first three transformed columns are the scaled numerics
    means = np.mean(xt[:, :3], axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-12)


def test_onehot_has_one_column_per_declared_segment():
    frame = _toy_frame()
    pre = build_preprocessor()
    pre.fit(frame)
    encoder = pre.named_transformers_["cat"]
    assert list(encoder.categories_[0]) == list(SEGMENT_VALUES)


def test_unfitted_pipeline_predict_raises():
    pipe = build_model_pipeline()
    with pytest.raises(NotFittedError):
        pipe.predict(_toy_frame())
