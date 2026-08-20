"""Local artifact bundle: fitted Pipeline plus version metadata.

Problem: a serving process must load the same object that training
fitted, together with the seed, library versions, feature order, and
schema/model versions that define the contract.
Assumptions: joblib can serialise the sklearn Pipeline; metadata is JSON
with string keys; this is a filesystem registry, not a model store with
ACLs.
Why this method: one file for the Pipeline and one JSON sidecar (also
embedded in the joblib dict) is enough to test rollback and version
mismatch without introducing a database.
Alternative: MLflow model registry, S3 plus a pointer table. Those are
valid; they are out of scope for this laboratory.
What can go wrong: loading a pickle from an untrusted path; serving an
artifact whose schema_version does not match the API models.
Independent check: train, dump, load, compare predict_proba.
Can conclude: the bytes on disk reconstruct the fitted object in this
environment.
Cannot conclude: pickle compatibility across sklearn major versions.
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
