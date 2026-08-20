"""Training integration: artifact round-trip and dummy baseline."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from mlserv.artifacts import load_bundle
from mlserv.config import TrainConfig, load_config
from mlserv.pipeline.preprocess import feature_names_in_order
from mlserv.schema import FEATURE_ORDER, SCHEMA_VERSION
from mlserv.train import train, train_and_persist


def test_artifact_reload_matches_in_memory_proba(persisted_result):
    loaded = load_bundle(persisted_result.artifact_path)
    X = persisted_result.X_val
    p_mem = persisted_result.pipeline.predict_proba(X)[:, 1]
    p_disk = loaded.pipeline.predict_proba(X)[:, 1]
    np.testing.assert_allclose(p_mem, p_disk, atol=1e-12)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.metadata["seed"] == 2026
    assert loaded.metadata["feature_order"] == list(FEATURE_ORDER)


def test_metadata_records_versions_and_split(train_result):
    meta = train_result.metadata
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["model_version"] == "1.0.0"
    assert "python_version" in meta
    assert "sklearn_version" in meta
    assert meta["n_train"] + meta["n_val"] == 240
    assert feature_names_in_order(train_result.pipeline) == FEATURE_ORDER


def test_logistic_beats_dummy_on_this_dgp(train_result):
    """Property of this DGP and this model class, not a general ranking."""
    assert train_result.metrics.accuracy >= train_result.baseline_metrics.accuracy
    assert train_result.metrics.brier <= train_result.baseline_metrics.brier + 1e-12


def test_same_seed_reproduces_val_proba():
    cfg = TrainConfig(seed=2026, n_samples=160, test_size=0.25)
    a = train(cfg, log_mlflow=False)
    b = train(cfg, log_mlflow=False)
    np.testing.assert_allclose(
        a.pipeline.predict_proba(a.X_val)[:, 1],
        b.pipeline.predict_proba(b.X_val)[:, 1],
        atol=1e-12,
    )


def test_load_config_rejects_unknown_model_type(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 2026\nn_samples: 80\nmodel:\n  type: xgboost\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported model type"):
        load_config(path)


def test_missing_artifact_raises(tmp_path):
    from mlserv.artifacts import load_bundle

    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")


def test_mlflow_run_id_does_not_replace_artifact_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = tmp_path / "mlruns"
    cfg = TrainConfig(
        seed=2026,
        n_samples=80,
        test_size=0.25,
        mlflow_tracking_uri=str(tracking),
        mlflow_experiment="identity-split",
    )
    result = train_and_persist(cfg, artifact_dir=tmp_path / "art", log_mlflow=True)
    uuid.UUID(str(result.metadata["artifact_id"]))
    assert result.metadata["mlflow_run_id"]
    assert result.metadata["artifact_id"] != result.metadata["mlflow_run_id"]
