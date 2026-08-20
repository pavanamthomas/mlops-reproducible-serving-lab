# Rollback

`mlserv.failures.rollback.select_previous_artifact` takes a list of `ArtifactRecord` and a current `artifact_id`. Records are sorted by `created_at` then `artifact_id`. The function returns the immediately preceding record.

That is the whole mechanism.

## What it does not do

- It does not move traffic.
- It does not check that the previous artifact's `schema_version` matches the current API. Call `check_versions` separately.
- It does not run a canary or a shadow comparison.
- It does not talk to MLflow Model Registry stages.
- It cannot roll back the oldest record; that raises `ValueError`.

## Intended use in this laboratory

A test constructs three records and asks for the previous id of the latest one. `scripts/run_all.py` prints that result. A human replacing a bad `models/bundle.joblib` would copy the previous file into place and restart uvicorn. That operational step is outside the package.

See also `docs/model_versioning.md`.
