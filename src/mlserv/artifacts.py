"""Joblib Pipeline plus JSON metadata for local rollback checks.

Not a model registry. Train, dump, load, and compare predict_proba.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass
class ArtifactBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    path: Path | None = None

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def schema_version(self) -> str:
        return str(self.metadata["schema_version"])

    @property
    def artifact_id(self) -> str:
        return str(self.metadata["artifact_id"])


def save_bundle(path: str | Path, pipeline: Pipeline, metadata: dict[str, Any]) -> Path:
    """Write joblib bundle and a JSON sidecar with the same metadata."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pipeline": pipeline, "metadata": metadata}
    joblib.dump(payload, dest)
    sidecar = dest.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dest


def load_bundle(path: str | Path) -> ArtifactBundle:
    """Load a bundle written by ``save_bundle``."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"artifact not found: {src}")
    payload = joblib.load(src)
    if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
        raise ValueError(f"artifact at {src} is not an mlserv bundle")
    pipeline = payload["pipeline"]
    metadata = dict(payload["metadata"])
    if not isinstance(pipeline, Pipeline):
        raise ValueError("bundle pipeline is not a sklearn Pipeline")
    return ArtifactBundle(pipeline=pipeline, metadata=metadata, path=src)
