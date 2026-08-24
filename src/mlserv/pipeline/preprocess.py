"""StandardScaler and OneHotEncoder inside the sklearn Pipeline.

Unseen segments are rejected. Serving a separately fitted scaler is the
training-serving skew the laboratory constructs on purpose.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlserv.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEGMENT_VALUES


def build_preprocessor() -> ColumnTransformer:
    """Numeric z-scores and a one-hot encoder that errors on unseen levels."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(
                    categories=[list(SEGMENT_VALUES)],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    *,
    C: float = 1.0,
    max_iter: int = 200,
    random_state: int = 2026,
) -> Pipeline:
    """Preprocess and logistic regression as one fitted object."""
    if C <= 0:
        raise ValueError("C must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def feature_names_in_order(pipeline: Pipeline) -> tuple[str, ...]:
    """Names recorded by sklearn when the Pipeline was fitted on a DataFrame."""
    names = getattr(pipeline, "feature_names_in_", None)
    if names is None:
        raise ValueError("pipeline has no feature_names_in_; fit on a DataFrame")
    return tuple(str(name) for name in names)
