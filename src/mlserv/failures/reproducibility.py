"""Fields required to reconstruct the served function.

Coefficients without preprocessor, feature order, seed, and library
versions are not enough for training-serving equality.
"""
from __future__ import annotations

REQUIRED_RETENTION: tuple[str, ...] = (
    "seed",
    "python_version",
    "sklearn_version",
    "numpy_version",
    "pandas_version",
    "feature_names",
    "feature_order",
    "schema_version",
    "model_version",
    "hyperparameters",
    "n_train",
    "n_val",
    "dgp",
    "artifact_id",
    "fitted_pipeline",
)


def missing_retention(provided: set[str] | dict[str, object]) -> tuple[str, ...]:
    """Return required names that are absent from ``provided``."""
    keys = set(provided) if not isinstance(provided, set) else provided
    return tuple(name for name in REQUIRED_RETENTION if name not in keys)
