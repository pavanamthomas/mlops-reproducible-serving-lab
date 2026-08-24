# Failures and corrections

The laboratory keeps serving and monitoring mistakes visible. A “successful” test here often means the **wrong procedure still misbehaves** under a known DGP.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| Scale numeric features with train+serve or serve-batch statistics, then apply the trained classifier | Predicted probabilities diverge from `pipeline.predict_proba` | max absolute probability difference | Serve the fitted Pipeline; do not refit the scaler | `tests/test_skew.py` | Other skews such as unit changes or training-only imputation |
| Write income into the `age` field through a valid-looking mapping | API schema can still accept the record; predictions are wrong | feature-swap detector | Named columns from one schema module; parity test | `tests/test_skew.py`, `tests/test_parity.py` | Upstream feature-store joins |
| Quote HTTP 200 as evidence that serving matches training | Health and schema validation do not test the prediction function | parity test on known rows | Route serving through the fitted Pipeline | `tests/test_parity.py` | Handlers that bypass the helper |
| Start a freshly built Docker container and treat the first TCP reset as a terminal parity failure | The first container-parity CI run failed with `ConnectionResetError` while the service was still starting; no probability comparison had yet occurred | CI log showed the failure inside `/health` polling, after tests, training, reproduction, and Docker build had passed | Classify startup connection reset as transient, retry health polling, then perform the probability check only after `model_loaded=true` | `tests/test_container_probe.py` plus the end-to-end CI container step | Readiness under orchestrators or real traffic |
| PSI/KS on a mean-shifted income column, read as “retrain now” | Two-sample descriptive statistics are not a retraining decision rule | `numeric_shift_report` | State the object: the input distribution changed | `tests/test_failures.py`, `docs/monitoring_limits.md` | Sequential false-alarm rates |
| Score the original Pipeline after flipping the DGP credit slope | Accuracy falls; prediction histogram may still look ordinary | concept-shift evaluation plus delayed labels | Compute delayed accuracy only on arrived labels | `tests/test_failures.py` | Real delayed-label processes |
| Drop `segment` or add `device_id` | Schema report returns `ok=False` | `schema_drift_report` | Reject before predict | `tests/test_failures.py`, API 422 | Producer-side event versioning |
| Load `schema_version=1.0` against expected `2.0` | `VersionMismatchError` | `check_versions` | Hard fail, then use rollback selector | `tests/test_failures.py` | Semantic compatibility of two artifacts with the same version string |
| Keep coefficients but drop `feature_order` or sklearn version | Serving equality is no longer reconstructible | `missing_retention` | Retain the required metadata list | `tests/test_failures.py` | Cross-major sklearn pickle stability |
| Alias `artifact_id` to the MLflow run id when logging is on | Rollback keys would depend on whether tracking ran | UUID versus `mlflow_run_id` | Local UUID remains bundle identity; run id is tracking metadata | `tests/test_train.py::test_mlflow_run_id_does_not_replace_artifact_id` | Hosted MLflow registry identifiers |

The startup-reset correction is intentionally recorded rather than hidden. The failed run did **not** show model-parity failure; it showed that readiness and prediction correctness are different checks. After the retry logic was added, the container CI reached the probability comparison and passed.

Process: `docs/lab_process.md`. Flagship narrative: `FLAGSHIP_TRAINING_SERVING_SKEW.md`.
