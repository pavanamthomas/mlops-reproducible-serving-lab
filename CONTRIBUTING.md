# Contributing to the serving lab

Useful work is a schema-valid request that returns HTTP 200 and the wrong probability, or a container check that compares `/predict-proba` to the offline Pipeline rather than to a health badge.

1. Open an issue naming the served transform, the schema, and the mismatch with the fitted `Pipeline`.
2. Add a failing test before a numerical or contract change.
3. Keep commits to one serving-contract claim.
4. Comment column order, version metadata, and readiness, not obvious FastAPI boilerplate.
5. Do not commit `mlruns/`, trained pickles, or `.env` files. Tests train a tiny model in a session fixture.

See `FLAGSHIP_TRAINING_SERVING_SKEW.md`, `docs/container_parity.md`, `ROADMAP.md`, and `.github/workflows/ci.yml`.
