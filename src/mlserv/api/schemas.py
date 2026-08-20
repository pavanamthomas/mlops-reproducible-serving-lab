"""Pydantic request and response models for the laboratory API.

These bounds must match ``mlserv.schema``. Extra fields are forbidden
so that an unexpected production field is a 422, not a silent drop.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlserv.schema import (
    AGE_MAX,
    AGE_MIN,
    CREDIT_MAX,
    CREDIT_MIN,
    FEATURE_ORDER,
    INCOME_MAX,
    INCOME_MIN,
    SCHEMA_VERSION,
    SEGMENT_VALUES,
)

Segment = Literal["A", "B", "C"]
assert SEGMENT_VALUES == ("A", "B", "C")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=AGE_MIN, le=AGE_MAX)
    income: float = Field(ge=INCOME_MIN, le=INCOME_MAX)
    credit_score: float = Field(ge=CREDIT_MIN, le=CREDIT_MAX)
    segment: Segment


class PredictResponse(BaseModel):
    prediction: int
    model_version: str
    schema_version: str = SCHEMA_VERSION


class PredictProbaResponse(BaseModel):
    probability_1: float
    prediction: int
    model_version: str
    schema_version: str = SCHEMA_VERSION


class ModelInfoResponse(BaseModel):
    model_version: str
    schema_version: str
    feature_names: list[str]
    feature_order: list[str]
    python_version: str
    sklearn_version: str
    seed: int
    artifact_id: str
    dgp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def request_to_ordered_dict(req: PredictRequest) -> dict[str, float | str]:
    data = req.model_dump()
    return {name: data[name] for name in FEATURE_ORDER}
