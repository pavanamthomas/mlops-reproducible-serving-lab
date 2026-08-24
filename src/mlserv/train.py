"""Fit the sklearn Pipeline and write the versioned artifact.

Seed, feature order, schema version, and the fitted object are stored
together. Local MLflow logging is optional; CI does not start the UI.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlserv.artifacts import ArtifactBundle, save_bundle
from mlserv.config import TrainConfig, load_config
from mlserv.data import DGP_NAME, generate_dataset
from mlserv.evaluate import EvalResult, evaluate_classifier
from mlserv.pipeline import build_model_pipeline, feature_names_in_order
from mlserv.schema import FEATURE_ORDER, SCHEMA_VERSION, require_valid_frame


@dataclass
class TrainResult:
    pipeline: Pipeline
    baseline: DummyClassifier
    metadata: dict[str, Any]
    metrics: EvalResult
    baseline_metrics: EvalResult
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    artifact_path: Path | None = None

    @property
    def bundle(self) -> ArtifactBundle:
        return ArtifactBundle(
            pipeline=self.pipeline,
            metadata=self.metadata,
            path=self.artifact_path,
        )


def _library_versions() -> dict[str, str]:
    import numpy
    import pandas

    return {
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
    }


def _as_mlflow_uri(uri: str) -> str:
    """Windows ``C:\\...`` is not a valid MLflow scheme; coerce to file://."""
    if uri.startswith(("file:", "http:", "https:", "sqlite:", "databricks")):
        return uri
    path = Path(uri)
    return path.resolve().as_uri()


def _log_mlflow(
    config: TrainConfig,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    metrics: EvalResult,
    baseline_metrics: EvalResult,
    X_train: pd.DataFrame,
    artifact_path: Path | None,
) -> str:
    import os

    import mlflow
    import mlflow.sklearn

    # MLflow 3.x treats the filesystem backend as maintenance-mode. This
    # laboratory opts in: a local file store is the intended artefact, not a
    # hosted tracking server. CI does not start the UI.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = _as_mlflow_uri(config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "seed": config.seed,
                "model_type": config.model_type,
                "C": config.C,
                "max_iter": config.max_iter,
                "n_samples": config.n_samples,
                "test_size": config.test_size,
                "schema_version": config.schema_version,
                "model_version": config.model_version,
                "dgp": DGP_NAME,
            }
        )
        mlflow.log_metrics(
            {
                "val_accuracy": metrics.accuracy,
                "val_log_loss": metrics.log_loss,
                "val_roc_auc": metrics.roc_auc,
                "val_brier": metrics.brier,
                "dummy_val_accuracy": baseline_metrics.accuracy,
                "dummy_val_log_loss": baseline_metrics.log_loss,
                "dummy_val_brier": baseline_metrics.brier,
            }
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            pip_requirements=["scikit-learn", "pandas", "numpy"],
            input_example=X_train.head(2),
        )
        if artifact_path is not None and artifact_path.is_file():
            mlflow.log_artifact(str(artifact_path))
        metadata["mlflow_run_id"] = run.info.run_id
        # Keep artifact_id as the local UUID. The run id is tracking metadata,
        # not a replacement identity for rollback.
        return str(run.info.run_id)


def train(config: TrainConfig, *, log_mlflow: bool = False) -> TrainResult:
    """Fit Pipeline and DummyClassifier on a stratified split of the DGP."""
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"config schema_version {config.schema_version!r} != code {SCHEMA_VERSION!r}"
        )
    batch = generate_dataset(config.n_samples, seed=config.seed)
    X = require_valid_frame(batch.X)
    y = np.asarray(batch.y, dtype=int)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    pipeline = build_model_pipeline(
        C=config.C,
        max_iter=config.max_iter,
        random_state=config.seed,
    )
    pipeline.fit(X_train, y_train)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    metrics = evaluate_classifier(pipeline, X_val, y_val)
    baseline_metrics = evaluate_classifier(baseline, X_val, y_val)

    artifact_id = str(uuid.uuid4())
    metadata: dict[str, Any] = {
        "artifact_id": artifact_id,
        "model_version": config.model_version,
        "schema_version": config.schema_version,
        "seed": config.seed,
        "dgp": DGP_NAME,
        "model_type": config.model_type,
        "hyperparameters": {"C": config.C, "max_iter": config.max_iter},
        "feature_names": list(FEATURE_ORDER),
        "feature_order": list(feature_names_in_order(pipeline)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "test_size": config.test_size,
        "metrics": metrics.as_dict(),
        "baseline_metrics": baseline_metrics.as_dict(),
        "baseline": "DummyClassifier(strategy='most_frequent')",
        **_library_versions(),
    }
    if log_mlflow:
        run_id = _log_mlflow(
            config,
            pipeline,
            metadata,
            metrics,
            baseline_metrics,
            X_train,
            artifact_path=None,
        )
        metadata["mlflow_run_id"] = run_id

    return TrainResult(
        pipeline=pipeline,
        baseline=baseline,
        metadata=metadata,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
    )


def train_and_persist(
    config: TrainConfig,
    artifact_dir: str | Path,
    *,
    log_mlflow: bool = False,
) -> TrainResult:
    """Train, write joblib+JSON, optionally log to a local MLflow file store."""
    result = train(config, log_mlflow=False)
    dest = Path(artifact_dir) / "bundle.joblib"
    save_bundle(dest, result.pipeline, result.metadata)
    result.artifact_path = dest
    if log_mlflow:
        run_id = _log_mlflow(
            config,
            result.pipeline,
            result.metadata,
            result.metrics,
            result.baseline_metrics,
            result.X_train,
            artifact_path=dest,
        )
        result.metadata["mlflow_run_id"] = run_id
        save_bundle(dest, result.pipeline, result.metadata)
    return result


def train_from_config(
    config_path: str | Path | None = None,
    *,
    artifact_dir: str | Path = "models",
    n_samples: int | None = None,
    log_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> TrainResult:
    config = load_config(config_path)
    if n_samples is not None:
        config = config.replace(n_samples=n_samples)
    if tracking_uri is not None:
        config = config.replace(mlflow_tracking_uri=tracking_uri)
    return train_and_persist(config, artifact_dir=artifact_dir, log_mlflow=log_mlflow)
