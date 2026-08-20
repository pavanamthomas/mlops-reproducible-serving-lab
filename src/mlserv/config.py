"""Training configuration loaded from YAML.

The YAML file is the recorded experiment statement: seed, model type,
hyperparameters, schema version, and model version. Code must not
silently change these after the file is read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/train.yaml")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    schema_version: str = "1.0"
    model_version: str = "1.0.0"
    n_samples: int = 800
    test_size: float = 0.25
    model_type: str = "logistic"
    C: float = 1.0
    max_iter: int = 200
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "mlserv-synthetic"

    def replace(self, **kwargs: Any) -> TrainConfig:
        return replace(self, **kwargs)


def load_config(path: str | Path | None = None) -> TrainConfig:
    """Parse YAML into TrainConfig. Unknown model types raise ValueError."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config at {config_path} must be a mapping")
    model = raw.get("model") or {}
    mlflow_cfg = raw.get("mlflow") or {}
    model_type = str(model.get("type", "logistic"))
    if model_type != "logistic":
        raise ValueError(f"unsupported model type {model_type!r}; only logistic is in this laboratory")
    test_size = float(raw.get("test_size", 0.25))
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    n_samples = int(raw.get("n_samples", 800))
    if n_samples < 40:
        raise ValueError("n_samples must be at least 40 so a stratified split is defined")
    return TrainConfig(
        seed=int(raw.get("seed", 2026)),
        schema_version=str(raw.get("schema_version", "1.0")),
        model_version=str(raw.get("model_version", "1.0.0")),
        n_samples=n_samples,
        test_size=test_size,
        model_type=model_type,
        C=float(model.get("C", 1.0)),
        max_iter=int(model.get("max_iter", 200)),
        mlflow_tracking_uri=str(mlflow_cfg.get("tracking_uri", "file:./mlruns")),
        mlflow_experiment=str(mlflow_cfg.get("experiment_name", "mlserv-synthetic")),
    )
