"""Serving helpers that call the fitted Pipeline on an ordered frame.

Problem: HTTP handlers must not reimplement preprocessing. The estimand
is equality of served g-hat and training g-hat for the same record x.
Assumptions: the loaded object is the fitted sklearn Pipeline; columns
are reindexed to ``FEATURE_ORDER`` before ``predict``.
Why this method: one function used by the API and by the parity test.
Alternative: a handwritten numpy transform in the handler. That is the
flagship failure mode.
What can go wrong: converting to a numpy array in the wrong column
order; fitting a new scaler on the request batch.
Independent check: ``tests/test_parity.py``.
Can conclude: this helper uses the fitted Pipeline as-is.
Cannot conclude: that every future handler will call this helper.
"""

from __future__ import annotations

import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mlserv.schema import FEATURE_ORDER, require_valid_frame


def frame_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build a DataFrame with the training feature order."""
    if not records:
        raise ValueError("records must be non-empty")
    frame = pd.DataFrame.from_records(records)
    return require_valid_frame(frame.loc[:, list(FEATURE_ORDER)])


def predict_proba_1(pipeline: Pipeline, frame: pd.DataFrame) -> NDArray:
    """P(y=1 | x) from the fitted Pipeline."""
    ordered = require_valid_frame(frame)
    proba = pipeline.predict_proba(ordered)
    return proba[:, 1]


def predict_label(pipeline: Pipeline, frame: pd.DataFrame) -> NDArray:
    ordered = require_valid_frame(frame)
    return pipeline.predict(ordered)
