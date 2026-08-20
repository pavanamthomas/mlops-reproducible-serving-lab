"""Shared fixtures: train a tiny model in-process so no pickle is committed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlserv.api.app import create_app
from mlserv.config import TrainConfig
from mlserv.train import train, train_and_persist


@pytest.fixture(scope="session")
def train_result():
    config = TrainConfig(seed=2026, n_samples=240, test_size=0.25, C=1.0, max_iter=200)
    return train(config, log_mlflow=False)


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory, train_result):
    root = tmp_path_factory.mktemp("artifact")
    path = root / "bundle.joblib"
    from mlserv.artifacts import save_bundle

    save_bundle(path, train_result.pipeline, train_result.metadata)
    bundle = train_result.bundle
    bundle.path = path
    return bundle


@pytest.fixture(scope="session")
def client(trained_bundle):
    app = create_app(trained_bundle)
    return TestClient(app)


@pytest.fixture
def persisted_result(tmp_path):
    config = TrainConfig(seed=2026, n_samples=160, test_size=0.25)
    return train_and_persist(config, artifact_dir=tmp_path, log_mlflow=False)
