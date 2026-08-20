"""Schema module and Pydantic contract tests."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from mlserv.api.schemas import PredictRequest
from mlserv.schema import (
    AGE_MAX,
    AGE_MIN,
    CREDIT_MAX,
    CREDIT_MIN,
    FEATURE_ORDER,
    INCOME_MAX,
    INCOME_MIN,
    SCHEMA_VERSION,
    require_valid_frame,
    validate_frame,
)


def _ok_row() -> dict:
    return {"age": 40.0, "income": 35000.0, "credit_score": 0.6, "segment": "B"}


def test_valid_frame_is_reindexed_to_feature_order():
    frame = pd.DataFrame([{**_ok_row(), "age": 41.0}])[["segment", "income", "age", "credit_score"]]
    ordered = require_valid_frame(frame)
    assert list(ordered.columns) == list(FEATURE_ORDER)


def test_missing_column_is_a_violation():
    frame = pd.DataFrame([{"age": 40.0, "income": 35000.0, "credit_score": 0.6}])
    reasons = [v.reason for v in validate_frame(frame)]
    assert "missing" in reasons


def test_unexpected_column_is_a_violation():
    row = {**_ok_row(), "device": "phone"}
    frame = pd.DataFrame([row])
    reasons = [v.reason for v in validate_frame(frame)]
    assert "unexpected" in reasons


def test_out_of_range_age():
    frame = pd.DataFrame([{**_ok_row(), "age": AGE_MAX + 1}])
    assert any(v.column == "age" and v.reason == "out-of-range" for v in validate_frame(frame))


def test_unseen_segment():
    frame = pd.DataFrame([{**_ok_row(), "segment": "Z"}])
    assert any(v.reason == "unseen-category" for v in validate_frame(frame))


def test_pydantic_rejects_missing_field():
    payload = {"age": 40.0, "income": 35000.0, "credit_score": 0.6}
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(payload)


def test_pydantic_rejects_wrong_type():
    payload = {**_ok_row(), "age": "forty"}
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(payload)


def test_pydantic_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**_ok_row(), "credit_score": CREDIT_MAX + 0.2})
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**_ok_row(), "age": AGE_MIN - 1})
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**_ok_row(), "income": INCOME_MIN - 0.5})


def test_pydantic_rejects_unseen_category_and_extra_field():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**_ok_row(), "segment": "Z"})
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**_ok_row(), "extra": 1})


def test_pydantic_bounds_match_schema_constants():
    PredictRequest.model_validate(
        {"age": AGE_MIN, "income": INCOME_MIN, "credit_score": CREDIT_MIN, "segment": "A"}
    )
    PredictRequest.model_validate(
        {"age": AGE_MAX, "income": min(INCOME_MAX, 1_000_000.0), "credit_score": CREDIT_MAX, "segment": "C"}
    )
    assert SCHEMA_VERSION == "1.0"
