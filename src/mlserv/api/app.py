"""FastAPI application factory for local serving of the laboratory model.

The application is a contract check, not a production service. It loads
one artifact, rejects records that fail the Pydantic schema, and calls
the fitted Pipeline through ``mlserv.api.serving``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from mlserv.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictProbaResponse,
    PredictRequest,
    PredictResponse,
    request_to_ordered_dict,
)
from mlserv.api.serving import frame_from_records, predict_label, predict_proba_1
from mlserv.artifacts import ArtifactBundle, load_bundle
from mlserv.schema import SCHEMA_VERSION


def _bundle_from_env() -> ArtifactBundle | None:
    raw = os.environ.get("MLSERV_ARTIFACT_PATH")
    if not raw:
        default = Path("models/bundle.joblib")
        if default.is_file():
            return load_bundle(default)
        return None
    path = Path(raw)
    if not path.is_file():
        return None
    return load_bundle(path)


def create_app(bundle: ArtifactBundle | None = None) -> FastAPI:
    """Build the API. Tests inject a bundle; uvicorn uses env or models/."""
    app = FastAPI(
        title="mlserv laboratory",
        description="Local serving contract for a synthetic logistic classifier. Not a production service.",
        version="0.1.0",
    )
    app.state.bundle = bundle if bundle is not None else _bundle_from_env()

    def _require_bundle() -> ArtifactBundle:
        loaded = app.state.bundle
        if loaded is None:
            raise HTTPException(status_code=503, detail="no model artifact loaded")
        return loaded

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.bundle is not None
        return HealthResponse(status="ok", model_loaded=loaded)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        loaded = _require_bundle()
        meta = loaded.metadata
        return ModelInfoResponse(
            model_version=str(meta["model_version"]),
            schema_version=str(meta["schema_version"]),
            feature_names=list(meta["feature_names"]),
            feature_order=list(meta["feature_order"]),
            python_version=str(meta["python_version"]),
            sklearn_version=str(meta["sklearn_version"]),
            seed=int(meta["seed"]),
            artifact_id=str(meta["artifact_id"]),
            dgp=str(meta["dgp"]),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        loaded = _require_bundle()
        if loaded.schema_version != SCHEMA_VERSION:
            raise HTTPException(
                status_code=409,
                detail="schema_version mismatch between artifact and API",
            )
        frame = frame_from_records([request_to_ordered_dict(req)])
        label = int(predict_label(loaded.pipeline, frame)[0])
        return PredictResponse(
            prediction=label,
            model_version=loaded.model_version,
            schema_version=loaded.schema_version,
        )

    @app.post("/predict-proba", response_model=PredictProbaResponse)
    def predict_proba(req: PredictRequest) -> PredictProbaResponse:
        loaded = _require_bundle()
        if loaded.schema_version != SCHEMA_VERSION:
            raise HTTPException(
                status_code=409,
                detail="schema_version mismatch between artifact and API",
            )
        frame = frame_from_records([request_to_ordered_dict(req)])
        p1 = float(predict_proba_1(loaded.pipeline, frame)[0])
        label = int(p1 >= 0.5)
        return PredictProbaResponse(
            probability_1=p1,
            prediction=label,
            model_version=loaded.model_version,
            schema_version=loaded.schema_version,
        )

    return app


app = create_app()
