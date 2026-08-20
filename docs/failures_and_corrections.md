# Failures and corrections

The laboratory keeps serving and monitoring mistakes visible. A “successful” test here often means the **wrong procedure still misbehaves** under a known DGP.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| Scale numeric features with train+serve or serve-batch statistics, then apply the trained classifier | \(P(y=1)\) diverges from `pipeline.predict_proba` | max abs probability difference | Serve the fitted Pipeline; do not refit the scaler | `tests/test_skew.py` | Other skews (log vs linear units, training-time imputation) |
| Write income into the `age` field (CSV header mapping) | API schema can still accept the JSON; predictions are wrong | feature-swap detector | Named columns from a single schema module; parity test | `tests/test_skew.py`, `tests/test_parity.py` | Joins from a feature store |
| Quote HTTP 200 as evidence that serving matches training | Health and Pydantic do not test \(\hat g\) | parity test on val rows | `mlserv.api.serving` only | `tests/test_parity.py` | Handlers that bypass the helper |
| PSI/KS on a mean-shifted income column, read as “retrain now” | Two-sample descriptive stats, not a decision rule | `numeric_shift_report` | State the estimand: \(P(X)\) changed | `tests/test_failures.py`, `docs/monitoring_limits.md` | Sequential false-alarm rates |
| Score the original Pipeline after flipping the DGP credit slope | Accuracy falls; prediction histogram may still look ordinary | concept-shift eval + delayed labels | Delayed accuracy only on arrived labels | `tests/test_failures.py` | Real delayed-label processes |
| Drop `segment` or add `device_id` | Schema report `ok=False` | `schema_drift_report` | Reject before predict | `tests/test_failures.py`, API 422 | How producers version events |
| Load `schema_version=1.0` against expected `2.0` | `VersionMismatchError` | `check_versions` | Hard fail, then rollback selector | `tests/test_failures.py` | Semantic compatibility of two 1.0 artifacts |
| Keep coefficients but drop `feature_order` / sklearn version | Serving equality is not identified | `missing_retention` | Retain the list in `reproducibility.py` | `tests/test_failures.py` | Cross-major sklearn pickle stability |
| Alias `artifact_id` to the MLflow run id when logging is on | Rollback keys would depend on whether tracking ran | UUID vs `mlflow_run_id` | Local UUID is the bundle identity; run id is tracking metadata | `tests/test_train.py::test_mlflow_run_id_does_not_replace_artifact_id` | Hosted MLflow registry ids |

Process: `docs/lab_process.md`. Flagship narrative: `FLAGSHIP_TRAINING_SERVING_SKEW.md`.
