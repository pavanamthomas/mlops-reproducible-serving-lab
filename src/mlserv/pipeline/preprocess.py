"""sklearn preprocessing joined to the classifier.

Problem: a scaler or encoder fitted separately from the model is a
common way for serving code to silently use different statistics than
training.
Assumptions: training rows are a pandas DataFrame with the schema
feature names; the categorical encoder must reject unseen segments
rather than map them to a zero vector.
Why this method: ``sklearn.pipeline.Pipeline`` serialises one object
that serving can load. Joint fit is the contract.
Alternative: a feature store with explicit statistics snapshots. That
is a larger system; the failure mode is the same if the snapshot is not
the one used at train time.
What can go wrong: serving applies ``fit_transform`` on a production
batch; column order is taken from a CSV header; ``handle_unknown='ignore'``
hides a new category.
Independent check: ``tests/test_preprocess.py`` and the training-serving
parity test.
Can conclude: the fitted Pipeline is the unique intended transform for
this laboratory model.
Cannot conclude: that StandardScaler is the right transform for a
different DGP.
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
