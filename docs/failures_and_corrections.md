# Failures and corrections

Serving and monitoring mistakes stay visible. A passing test often means the **wrong procedure still misbehaves** on a known DGP.

## 24 Aug 2026 — first container-parity CI run

The Docker image built. Tests, training, and `scripts/run_all.py` had already passed. The first `/health` probe then hit `ConnectionResetError` while uvicorn was still coming up. No probability comparison had happened yet.

I had treated that reset as a terminal parity failure. It was a readiness failure. Correction: classify startup connection reset as transient, retry health polling, and only then compare probabilities after `model_loaded=true`. Locked by `tests/test_container_probe.py` and the end-to-end CI container step. Still unknown: readiness under an orchestrator or real traffic.

The failed run did **not** show model-parity failure. After the retry logic, the container CI reached the probability comparison and passed.

## Designed skews (in since 20 Aug)

**Pooled or request-batch scaling.** Classifier was trained with training-only preprocessing. Predicted probabilities diverge from `pipeline.predict_proba`. Serve the fitted Pipeline; do not refit the scaler. `tests/test_skew.py`. Other skews (unit changes, training-only imputation) are not covered.

**Feature swap.** Writing income into the `age` field through a valid-looking mapping. API schema can still accept the record. Named columns from one schema module; parity test. `tests/test_skew.py`, `tests/test_parity.py`. Upstream feature-store joins are out of scope.

**HTTP 200 as evidence.** Health and schema validation do not test the prediction function. Parity test on known rows; route serving through the fitted Pipeline. `tests/test_parity.py`. Handlers that bypass the helper would still slip.

**PSI/KS on a mean-shifted income column, read as “retrain now”.** Two-sample descriptive statistics are not a retraining decision rule. State the object: the input distribution changed. `tests/test_failures.py`, `docs/monitoring_limits.md`. Sequential false-alarm rates untested.

**Concept shift.** Score the original Pipeline after flipping the DGP credit slope. Accuracy falls; prediction histogram may still look ordinary. Compute delayed accuracy only on arrived labels. `tests/test_failures.py`. Real delayed-label processes are not modelled.

**Schema.** Drop `segment` or add `device_id` → `ok=False`. Reject before predict. `tests/test_failures.py`, API 422. Producer-side event versioning is not here.

**Version mismatch.** Load `schema_version=1.0` against expected `2.0` → `VersionMismatchError`. Hard fail, then rollback selector. Semantic compatibility of two artifacts with the same version string is not proved.

**Dropped `feature_order`.** Coefficients without feature order: serving equality is no longer reconstructible. Retain the required metadata list. Cross-major sklearn pickle stability is not guaranteed.

**MLflow run id as artifact id.** Rollback keys would depend on whether tracking ran. Local UUID remains bundle identity; run id is tracking metadata. `tests/test_train.py::test_mlflow_run_id_does_not_replace_artifact_id`.

Process: `docs/lab_process.md`. Flagship: `FLAGSHIP_TRAINING_SERVING_SKEW.md`.
