"""Call the fitted Pipeline on a DataFrame in FEATURE_ORDER.

Handlers must not rebuild preprocessing. Parity tests use this helper.
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
