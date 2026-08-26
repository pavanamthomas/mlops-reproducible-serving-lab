# mlops-reproducible-serving-lab

[![CI](https://github.com/pavanamthomas/mlops-reproducible-serving-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/mlops-reproducible-serving-lab/actions)

I trained a synthetic logistic classifier in an sklearn `Pipeline`, served it through FastAPI, and then broke serving on purpose. The number I care about is whether `/predict-proba` matches `pipeline.predict_proba` on the same schema-valid row. HTTP 200 is not that number.

The write-up is [`FLAGSHIP_TRAINING_SERVING_SKEW.md`](FLAGSHIP_TRAINING_SERVING_SKEW.md). A changed serving transform can still return 200. Docker path: [`docs/container_parity.md`](docs/container_parity.md) — build, wait for `/health`, one fixed request, compare to an independently trained runner-side bundle.

The first time I wired that into CI the image built and then the first `/health` probe got `ConnectionResetError` during startup. I had treated the reset as a parity failure. It was a readiness failure. Notes in [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md); the retry is locked in [`tests/test_container_probe.py`](tests/test_container_probe.py).

```bash
python -m pip install -e .
python -m pytest
python scripts/train.py --skip-mlflow
python scripts/run_all.py
```

Python 3.11+. No observational dataset. Tests train small deterministic models; I did not commit a fitted pickle as evidence.

## Serving object

The fitted sklearn `Pipeline` applied to a named DataFrame under the training feature contract. I do not reimplement preprocessing in the API. Schema constants live in one module.

Labels come from a documented synthetic logistic mean in age, log income, credit score, and segment. The model class matches the DGP so the lab can concentrate on serving mistakes. Coefficients are not causal effects. Train/validation split is stratified; seed recorded. Logistic vs dummy is on that split only.

Pydantic rejects extra fields, missing fields, bad types, out-of-range values, and unseen segments. MLflow writes to a local file store; the UI is not started in CI. The artifact keeps a local id, model/schema versions, library versions, and feature order.

CI builds the image, waits until `/health` says `model_loaded=true`, sends one valid request, and compares probability to the offline pipeline within a named tolerance. Local contract check, not operations evidence. What PSI, KS, histograms, and delayed labels do not establish: [`docs/monitoring_limits.md`](docs/monitoring_limits.md).

## Things I break on purpose

- pooled or request-batch scaling while the classifier used training-only preprocessing
- age and income swapped under valid field names
- HTTP 200 treated as numerical parity
- covariate shift read as an automatic retrain
- concept drift with delayed labels
- missing or extra fields
- model/schema version mismatch
- rollback as “previous artifact id,” not a traffic manager
- a coefficient vector without the feature order needed to reconstruct predictions

Parity and skew: [`tests/test_parity.py`](tests/test_parity.py), [`tests/test_skew.py`](tests/test_skew.py), [`tests/test_container_probe.py`](tests/test_container_probe.py), [`tests/test_api.py`](tests/test_api.py). Named-column transforms should not care about DataFrame column order. Unseen categoricals get rejected. Reload matches in-memory `predict_proba`. Same seed regenerates validation probabilities. Container health polling survives a transient reset. PSI is near zero on identical samples and rises under the designed mean shift.

`scripts/run_all.py` regenerates figures and tables. Those files are not the source of truth.

## Local API / Docker

```bash
python scripts/train.py --skip-mlflow
uvicorn mlserv.api.app:app --host 127.0.0.1 --port 8000
```

```bash
docker build -t mlserv-lab .
docker run --rm -p 8000:8000 mlserv-lab
```

A green CI run means tests passed, the synthetic training path ran, the reproduction script ran, the image built and started, readiness is distinguishable from a startup reset, and one schema-valid HTTP prediction matched the offline pipeline. It does not mean uptime, latency, concurrency, security hardening, or production readiness.

## Limits

Synthetic rows are not credit data. Logistic beating the dummy on this DGP is expected. PSI/KS/histograms/delayed accuracy are diagnostics. One container-parity fixture is not coverage of the valid feature space. Version strings make mismatches visible; they do not prove semantic compatibility. Rollback does not move live traffic. No canary, shadow, hosted registry, signing, or Kubernetes. No causal finding about age, income, credit score, or segment. sklearn/joblib cross-major-version stability is not guaranteed.

The classifier is a serving fixture. Validation-design failures live in [machine-learning-model-selection-lab](https://github.com/pavanamthomas/machine-learning-model-selection-lab). MIT; see [`CITATION.cff`](CITATION.cff).

Pavanam Thomas
