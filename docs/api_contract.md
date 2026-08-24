# API contract

Local FastAPI app, factory `mlserv.api.app.create_app`.

| Method | Path | Success | Typical failure |
| --- | --- | --- | --- |
| GET | `/health` | 200 `{status, model_loaded}` | none; 200 even if no artifact |
| GET | `/model-info` | 200 versions, feature order, seed | 503 if no artifact |
| POST | `/predict` | 200 `{prediction, model_version, schema_version}` | 422 bad record; 503 no artifact; 409 schema mismatch |
| POST | `/predict-proba` | 200 `{probability_1, prediction, ...}` | same |

Pydantic model `PredictRequest`: extra fields forbidden; `age` in [18, 90]; `income` in [1, 1e7]; `credit_score` in [0, 1]; `segment` in {A, B, C}. Missing fields, wrong types, out-of-range values, and unseen categories are 422.

The handler does not preprocess. It builds an ordered DataFrame and calls the fitted Pipeline (`mlserv.api.serving`).

## Local uvicorn (not production)

```bash
python scripts/train.py --skip-mlflow
export MLSERV_ARTIFACT_PATH=models/bundle.joblib   # Windows: set MLSERV_ARTIFACT_PATH=models/bundle.joblib
uvicorn mlserv.api.app:app --host 127.0.0.1 --port 8000
```

This is a contract check on a laptop. It is not a deployment.

## Docker (local image only)

The `Dockerfile` installs the package, trains a small model during the image build, and runs uvicorn. The image is a reproduction aid. It is not used in production in this laboratory, and no latency or uptime number is claimed.

```bash
docker build -t mlserv-lab .
docker run --rm -p 8000:8000 mlserv-lab
```

`.dockerignore` keeps tests, docs, and `mlruns/` out of the build context.

## What CI actually checks on the image

GitHub Actions builds the Docker image and runs `scripts/check_container_parity.py` against a live container. That script waits until `/health` reports `model_loaded`, then compares one schema-valid `/predict-proba` response with the offline fitted Pipeline, including schema and model versions.

The first end-to-end attempt failed with `ConnectionResetError` while the process was still binding the port. That is a readiness failure, not a prediction-parity failure. The probe retries bounded connection resets; numerical comparison happens only after readiness. The regression is `tests/test_container_probe.py`.

HTTP 200 is still not the estimand. A swapped-column request can return 200 with the wrong probability (`tests/test_skew.py`). The container check compares the served number with `pipeline.predict_proba` on the same fixture.
