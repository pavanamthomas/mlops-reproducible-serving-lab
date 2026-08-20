# Roadmap

Current as of August 2026.

## In scope now

- Synthetic binary classification DGP with a recorded schema.
- sklearn `Pipeline` that fits preprocessing jointly with logistic regression.
- Stratified train/val split with a recorded seed; DummyClassifier baseline on the same split.
- Local MLflow file store for params, metrics, and the fitted model. The UI is not started in CI.
- FastAPI contract: `/health`, `/model-info`, `/predict`, `/predict-proba`, with Pydantic rejection of bad records.
- Training-serving parity test and a deliberate skew detector.
- Simple PSI/KS covariate checks, a concept-drift DGP with delayed labels, schema drift, version mismatch, and a previous-artifact rollback selector.
- Docker files for a local image. The image is not used as a production system.
- CI: `pytest`, a training smoke via `scripts/train.py`, and `scripts/run_all.py`.

## Failures that are part of the design

- Serving that rescales with pooled or batch statistics while the classifier was trained on training-only statistics.
- Feature values placed in the wrong named columns (CSV header mapping).
- Covariate shift that moves PSI/KS without any claim of a complete monitoring platform.
- Label mechanism change that delayed accuracy can detect only after labels arrive.
- Serving a model whose `schema_version` or `model_version` does not match the request contract.

Details: `docs/failures_and_corrections.md` and `FLAGSHIP_TRAINING_SERVING_SKEW.md`.

## Open (issues)

1. Canary traffic and shadow-mode comparison are not implemented; rollback is a registry lookup, not a live router.
2. Calibration under shift is not estimated; Brier scores are in-sample or on the synthetic val split only.
3. The Docker image is not built on GitHub Actions (see `docs/api_contract.md` and the CI workflow comments).
4. joblib may warn under NumPy 2.x when serialising arrays; the laboratory treats that as a library warning, not a parity failure. Starlette's TestClient may warn about an httpx transport shortcut; the official FastAPI test client remains the sync path used here.

## Explicitly not in scope

- Kubernetes, service meshes, or cloud-vendor deployment theatre.
- Claiming that logistic regression beats the dummy classifier on tasks other than this DGP.
- Treating PSI, KS, or a prediction-histogram shift as a complete observability stack.
- Causal claims about `age`, `income`, `credit_score`, or `segment`.
- Invented latency SLOs, uptime, or production incident history.

Close an issue only with a test or a limitation sentence.
