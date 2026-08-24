# mlops-reproducible-serving-lab

[![CI](https://github.com/pavanamthomas/mlops-reproducible-serving-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/mlops-reproducible-serving-lab/actions)

How do we make a trained ML model reproducible, testable, versioned, servable, monitorable, and safely replaceable without pretending a small laboratory is a production platform?

A synthetic logistic classifier is trained inside an sklearn `Pipeline`, compared with a dummy baseline, stored with version metadata, served through FastAPI, and then broken on purpose. The flagship failure is training-serving skew: an API can return HTTP 200 while its numerical probability is not the probability from the fitted training pipeline.

Author: Dr. Pavanam Thomas ([GitHub](https://github.com/pavanamthomas), thomaspavanam@gmail.com).

The invariant is simple: for the same schema-valid input, serving must reproduce the fitted pipeline's prediction. Health, schema validation, and prediction parity are separate checks.

## Start here

1. [`FLAGSHIP_TRAINING_SERVING_SKEW.md`](FLAGSHIP_TRAINING_SERVING_SKEW.md) — the API works but a changed serving transform returns the wrong number.
2. [`docs/container_parity.md`](docs/container_parity.md) — Docker build, readiness, HTTP request, and served-versus-offline probability parity in CI.
3. [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md) — designed skews plus the first failed container-parity CI attempt and its correction.
4. [`tests/test_parity.py`](tests/test_parity.py), [`tests/test_skew.py`](tests/test_skew.py), [`tests/test_container_probe.py`](tests/test_container_probe.py), and [`tests/test_api.py`](tests/test_api.py) — property and regression checks.
5. [`docs/monitoring_limits.md`](docs/monitoring_limits.md) — what PSI, KS, prediction histograms, and delayed labels do and do not establish.
6. [`ROADMAP.md`](ROADMAP.md) — remaining serving and monitoring bounds.

Reproduce locally:

```bash
python -m pip install -e .
python -m pytest
python scripts/train.py --skip-mlflow
python scripts/run_all.py
```

Python 3.11 or newer. There is no observational dataset. Tests train small deterministic synthetic models; no stale fitted pickle is committed as evidence.

## Contract and artifacts

**Serving object.** Correct serving is the fitted sklearn `Pipeline` applied to a named DataFrame under the same feature contract used during training. Preprocessing is not reimplemented in the API.

**DGP.** Binary labels come from a documented synthetic logistic mean in age, log income, credit score, and segment. The model class matches the DGP deliberately so this laboratory can focus on serving and validation failures. Coefficients are not causal effects.

**Split.** The training/validation split is stratified and the seed is recorded. Logistic metrics and the dummy baseline use the same validation split. Superiority is not claimed beyond this DGP.

**Schema.** Pydantic rejects extra fields, missing fields, invalid types, out-of-range values, and unseen segments. Schema constants are shared between training and API code rather than copied into separate definitions.

**Tracking.** MLflow records parameters, metrics, and model information to a local file store. The UI is not started in CI. The serving artifact retains a local artifact id, model/schema versions, library versions, and feature order.

**Container boundary.** CI now builds the Docker image, launches it, waits until `/health` reports a loaded model, sends a fixed valid request, and compares the returned probability with an independently trained runner-side bundle within a named tolerance. This is a local contract check, not evidence of production operations.

## Designed failures

- pooled or request-batch scaling in serving while the classifier was trained with training-only preprocessing;
- age and income values swapped under valid field names;
- HTTP 200 treated as if it proved numerical parity;
- covariate shift read as an automatic retraining decision;
- concept drift with delayed labels;
- missing or extra fields;
- model/schema version mismatch;
- rollback represented as “previous artifact id,” not a traffic manager;
- a coefficient vector retained without the feature order needed to reconstruct predictions.

The first container CI attempt added a real correction rather than a cosmetic green badge. The Docker image built successfully, but the first `/health` probe received `ConnectionResetError` during startup. The harness originally treated that reset as terminal. It now retries transient startup resets and only evaluates prediction parity after the service reports `model_loaded=true`; a regression test forces that sequence. Details: [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md).

## Parity and failure tests

Tests check that:

- named-column transforms are invariant to DataFrame column permutation;
- unseen categorical values are rejected rather than silently encoded;
- artifact reload matches in-memory `predict_proba` within a tight numerical tolerance;
- the same seed regenerates the validation probabilities;
- serving helper output equals the fitted pipeline output;
- skew detectors fire on pooled scaling, batch scaling, and feature swapping;
- API responses are 200 for valid records, 422 for invalid records, and 503 when no artifact is available;
- container health polling survives transient connection resets;
- the CI container returns the same probability as the independently instantiated offline pipeline for the fixed fixture;
- PSI is near zero on identical samples and PSI/KS rise under the designed mean shift;
- rollback selection returns the preceding artifact id and fails at the oldest artifact.

`scripts/run_all.py` regenerates figures and summary tables. Those outputs are not the source of truth; the code and tests are.

## Local API and Docker

Local API:

```bash
python scripts/train.py --skip-mlflow
uvicorn mlserv.api.app:app --host 127.0.0.1 --port 8000
```

Local Docker image:

```bash
docker build -t mlserv-lab .
docker run --rm -p 8000:8000 mlserv-lab
```

CI performs the corresponding build and one deterministic parity request. It does not perform load testing or publish an image as a production service.

## What the CI evidence means

A green run establishes that, for the tested environment and fixtures:

- unit/integration tests pass;
- the synthetic training path executes;
- the laboratory reproduction script executes;
- the Docker image builds and starts;
- readiness is distinguishable from transient startup connection errors;
- one schema-valid HTTP prediction matches the offline fitted pipeline within the stated tolerance.

It does **not** establish uptime, latency, concurrency, security hardening, operational SLOs, or general production readiness.

## Known limitations

- Synthetic rows are not credit data; the feature names are labels of columns in a DGP.
- Logistic beating the dummy on this DGP is expected and is not a model-selection result for other tasks.
- PSI, KS, prediction histograms, and delayed accuracy are diagnostics, not a monitoring product.
- One fixed container-parity fixture is not exhaustive coverage of the valid feature space.
- Version strings make mismatches visible but do not prove semantic compatibility.
- Rollback does not move live traffic.
- Canary, shadow traffic, hosted registry, access control, signing, and cloud deployment are not implemented.
- No uptime, latency SLO, production incident history, or Kubernetes experience is claimed.
- No result here is a causal finding about age, income, credit score, or segment.
- sklearn/joblib cross-major-version stability is not guaranteed.

## Repository structure

```text
mlops-reproducible-serving-lab/
├── FLAGSHIP_TRAINING_SERVING_SKEW.md
├── configs/
├── docs/
├── src/mlserv/
├── scripts/
├── tests/
├── Dockerfile
└── .github/workflows/ci.yml
```

Related laboratories: [computational-ml-stem-problem-forge](https://github.com/pavanamthomas/computational-ml-stem-problem-forge), [machine-learning-model-selection-lab](https://github.com/pavanamthomas/machine-learning-model-selection-lab), and [statistical-reasoning-validation](https://github.com/pavanamthomas/statistical-reasoning-validation).

## Citation

See [`CITATION.cff`](CITATION.cff). Licence: MIT, Copyright 2026 Dr. Pavanam Thomas.
