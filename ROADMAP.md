# Roadmap

Serving and monitoring limits remaining after Docker prediction-parity landed in CI (August 2026).

## In scope now

- Synthetic binary classification DGP with a recorded schema.
- sklearn `Pipeline` that fits preprocessing jointly with logistic regression.
- Stratified train/validation split with a recorded seed and a DummyClassifier baseline on the same split.
- Local MLflow file store for parameters, metrics, and model artifacts; the UI is not started in CI.
- FastAPI contract: `/health`, `/model-info`, `/predict`, `/predict-proba`, with Pydantic rejection of invalid records.
- Training-serving parity tests and deliberate skew detectors.
- Docker image built and exercised in CI: health readiness, one schema-valid request, and served-versus-offline probability parity on a deterministic fixture.
- A regression test for transient connection reset during container startup, based on the first failed end-to-end CI attempt.
- PSI/KS covariate checks, a concept-drift DGP with delayed labels, schema drift, version mismatch, and a previous-artifact rollback selector.
- CI runs the test suite, a training smoke, the reproduction script, Docker build, and container prediction-parity check.

## Failures that are part of the design

- Serving that rescales with pooled or request-batch statistics while the classifier was trained with training-only preprocessing.
- Feature values placed in the wrong named columns while still satisfying basic JSON schema types.
- Treating HTTP 200 as proof that the numerical prediction matches the fitted training pipeline.
- Treating a transient container-startup connection reset as model-parity failure; readiness must be established before comparing predictions.
- Covariate shift that moves PSI/KS without implying a complete retraining policy.
- Label mechanism change that can be evaluated only after outcomes arrive.
- Serving an artifact whose schema or model version is incompatible with the request contract.

Details: `docs/failures_and_corrections.md`, `docs/container_parity.md`, and `FLAGSHIP_TRAINING_SERVING_SKEW.md`.

## Open (issues)

1. Canary traffic and shadow-mode comparison are not implemented; rollback remains an artifact-registry lookup rather than a live traffic router.
2. Calibration under shift is not estimated as a deployment policy; Brier scores are evaluated on synthetic splits.
3. The container parity check uses one deterministic fixture. Broader property-based request sampling across the valid schema is not implemented.
4. Hosted model registry, object-store provenance, signed artifacts, and access control are outside this local laboratory.
5. Cross-version model serialization remains bounded by sklearn/joblib compatibility; recorded versions make mismatch visible but do not guarantee forward compatibility.

## Explicitly not in scope

- Kubernetes, service meshes, or cloud-vendor deployment theatre.
- Claiming that logistic regression beats the dummy classifier on tasks other than the documented synthetic DGP.
- Treating PSI, KS, or a prediction-histogram shift as a complete observability stack.
- Causal claims about `age`, `income`, `credit_score`, or `segment`.
- Invented latency SLOs, uptime, production incident history, or traffic volume.

Close an issue only with executable evidence, a regression test, or a limitation sentence that narrows the claim.
