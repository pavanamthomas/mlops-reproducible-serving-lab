"""What must be retained for a trained model to be reconstructable.

Problem: a pickle of coefficients is not enough. Serving equality
of g-hat at serve time and train time also needs the
preprocessor, feature order, schema version, seed, and library versions.
Assumptions: the laboratory trains on a synthetic DGP; a real system
would also retain training-row hashes or a snapshot URI.
Why this method: a closed list that tests can subtract against a
metadata dict.
Alternative: a full experiment database. Out of scope.
What can go wrong: retaining the seed but not the sklearn version;
retaining MLflow run ids without the artifact bytes.
Independent check: ``missing_retention`` on a stripped metadata dict.
Can conclude: which named items are absent from a given record.
Cannot conclude: that keeping the list makes a result scientifically
valid; it only makes it reconstructable in this lab.
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
