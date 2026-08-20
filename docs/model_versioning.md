# Model versioning

An artifact in this laboratory is a joblib bundle plus a JSON sidecar:

- `model_version` — recorded in `configs/train.yaml`, copied onto the bundle
- `schema_version` — must match `mlserv.schema.SCHEMA_VERSION` (`1.0`)
- `artifact_id` — local UUID of the joblib bundle (stable whether or not MLflow logging is on)
- `mlflow_run_id` — present only when logging is on; tracking metadata, not the rollback key
- library versions, seed, feature order, hyperparameters, val metrics, dummy metrics

Serving reads those strings. `mlserv.failures.version_mismatch.check_versions` raises if either string differs from the expected pair. The API returns HTTP 409 if the loaded artifact's `schema_version` is not the API's schema.

## What a version string does not mean

Two artifacts with `schema_version=1.0` can still differ in preprocessing if someone changes the Pipeline and forgets to bump the schema. The laboratory's protection is the parity test plus the recorded `feature_order`, not a formal compatibility proof.

`model_version` is not a semantic-version promise about accuracy. It is a name for a fitted object.

## MLflow

When logging is on, params, metrics, and the sklearn model are written to a **local file store** (`mlruns/`, gitignored). That is experiment tracking on disk. It is not a hosted MLflow server. CI does not start the MLflow UI.

## Replacement

A new artifact is a new `artifact_id`. Rollback is selection of the previous id in a linear registry (`docs/rollback.md`). There is no canary, no shadow traffic, and no model stage named Production.
