# mlops-reproducible-serving-lab

[![CI](https://github.com/pavanamthomas/mlops-reproducible-serving-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/mlops-reproducible-serving-lab/actions)

How do we make a trained ML model reproducible, testable, versioned, deployable, monitorable, and safely replaceable — without Kubernetes theatre?

This repository is a small serving laboratory. A synthetic logistic classifier is trained inside an sklearn `Pipeline`, compared to a dummy baseline, stored with version metadata, served through FastAPI, and then broken on purpose. The flagship failure is training-serving skew: the API can return HTTP 200 while the serving transform is not the fitted Pipeline.

Author: Dr. Pavanam Thomas ([GitHub](https://github.com/pavanamthomas), thomaspavanam@gmail.com).

**Problem → formalization → assumptions → computation → validation → interpretation → limitations.**

## Recruiter 90-Second Audit

Inspect, in this order:

1. [`FLAGSHIP_TRAINING_SERVING_SKEW.md`](FLAGSHIP_TRAINING_SERVING_SKEW.md) — the API “works”; the number is wrong; a parity test catches it.
2. [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md) — failures the tests are required to keep visible.
3. [`src/mlserv/failures/training_serving_skew.py`](src/mlserv/failures/training_serving_skew.py) and [`src/mlserv/api/serving.py`](src/mlserv/api/serving.py) — wrong transforms versus the one helper the handler is allowed to call.
4. [`tests/test_parity.py`](tests/test_parity.py), [`tests/test_skew.py`](tests/test_skew.py), [`tests/test_api.py`](tests/test_api.py) — property checks, not smoke that “a response exists.”
5. [`docs/monitoring_limits.md`](docs/monitoring_limits.md) — PSI/KS and delayed accuracy are named as limited statistics.
6. [`ROADMAP.md`](ROADMAP.md) — bounds and open issues.

Reproduce from a clean clone:

```bash
python -m pip install -e .
python -m pytest
python scripts/run_all.py
```

Python 3.11 or newer. There is no observational dataset. `mlruns/` is gitignored. Tests train a tiny model in a session fixture; no stale pickle is committed.

## Technical Decisions I Can Defend

**Estimand.** Serving correctness is \(\hat g_{\text{serve}}(x)=\hat g_{\text{train}}(x)\) for schema-valid \(x\). \(\hat g_{\text{train}}\) is one fitted sklearn `Pipeline` (StandardScaler + OneHotEncoder with `handle_unknown='error'` + logistic regression). Preprocessing is not a separate script.

**DGP.** Binary labels from a known logistic mean in `age`, log `income`, `credit_score`, and `segment` (`mlserv.data`). The model class matches the DGP on purpose so a gap versus `DummyClassifier(strategy='most_frequent')` is not a misspecification puzzle. Coefficients are not causal effects.

**Split.** Stratified train/val split; seed recorded (`2026` by default). Metrics are val scores plus the dummy on the same split. Superiority is not claimed beyond this DGP.

**Contract.** Pydantic forbids extra fields and rejects missing fields, wrong types, out-of-range values, and unseen segments (HTTP 422). Schema constants live in `mlserv.schema` so the API and the trainer share bounds.

**Tracking.** MLflow logs params, metrics, and the model to a local `file://` store. The UI is not started in CI. The serving artifact is joblib + JSON metadata (`artifact_id`, schema/model versions, library versions, feature order).

**Architecture bound.** One process, one artifact path, a Dockerfile for a local image. No Kubernetes, no cloud lock-in layer, no invented SLOs. Docker is not built on GitHub Actions; pytest already checks the API and the parity estimand (`docs/api_contract.md`).

## Deliberate Failure Cases

- Pooled or batch `StandardScaler` in serving, trained classifier unchanged.
- Age/income values swapped under valid names (CSV mapping).
- Covariate shift on income: PSI and KS move; that is not a retrain policy.
- Concept drift: credit-score slope flipped; labels delayed; accuracy only on arrived labels.
- Missing or extra fields (`schema_drift`).
- `model_version` / `schema_version` mismatch.
- Rollback is “previous artifact id,” not a traffic manager.
- Retention list: a coefficient vector without feature order is not enough.

Details: `docs/failures_and_corrections.md`.

## Independent Validation

Tests check properties:

- Named-column transform is invariant to DataFrame column permutation; unseen `segment` raises.
- Artifact reload matches in-memory `predict_proba` within \(10^{-12}\).
- Same seed ⇒ same val probabilities.
- On this DGP, logistic val accuracy is at least the dummy’s, and Brier is no worse.
- FastAPI: 200 on a valid record, 422 on invalid records, 503 if no artifact.
- Serving helper equals `pipeline.predict_proba`.
- Skew detector fires on pooled scaling, batch scaling, and the feature swap.
- Extreme high/low DGP rows keep predicted probabilities on the correct side of 0.5 with a margin.
- PSI near zero on identical samples; PSI/KS rise under a mean shift.
- `select_previous_artifact` returns the preceding id; oldest id is an error.

`scripts/run_all.py` regenerates figures and `outputs/tables/run_summary.csv`. Those files are not the source of truth.

## Reproduce Everything

```bash
python -m pip install -e .
python -m pytest
python scripts/train.py --skip-mlflow
python scripts/run_all.py
```

Local API (not a production service):

```bash
python scripts/train.py --skip-mlflow
uvicorn mlserv.api.app:app --host 127.0.0.1 --port 8000
```

Local Docker image (not used in production):

```bash
docker build -t mlserv-lab .
docker run --rm -p 8000:8000 mlserv-lab
```

CI: pytest, training smoke (`scripts/train.py --n-samples 120 --skip-mlflow`), `scripts/run_all.py`. MLflow UI is not started. Docker is not built on GHA (see `docs/api_contract.md`).

```text
mlops-reproducible-serving-lab/
├── FLAGSHIP_TRAINING_SERVING_SKEW.md
├── configs/train.yaml
├── docs/
├── src/mlserv/          # DGP, Pipeline, train, API, failures, monitoring
├── scripts/train.py
├── scripts/run_all.py
├── tests/
├── Dockerfile
└── .github/workflows/ci.yml
```

## Limitations and Non-Claims

- Simulated rows are not credit data. Feature names are labels of columns.
- Logistic beating the dummy on this DGP is expected and is not a model-selection result for other tasks.
- PSI, KS, prediction histograms, and delayed accuracy are not a monitoring product (`docs/monitoring_limits.md`).
- Version strings are not a proof of semantic compatibility.
- Rollback does not move traffic.
- The Docker image and uvicorn command are local contract checks. No uptime, latency SLO, or incident history is claimed.
- No result here is a causal finding about age, income, or segment.
- sklearn major-version pickle stability is not guaranteed; the recorded `sklearn_version` makes a mismatch visible.

Related laboratories: [statistical-reasoning-validation](https://github.com/pavanamthomas/statistical-reasoning-validation), [econometrics-causal-inference-lab](https://github.com/pavanamthomas/econometrics-causal-inference-lab).

## Interview Questions This Repository Naturally Raises

1. If `/health` is 200 and `/predict` is 200, what estimand has been checked, and what has not?
2. Why fit `StandardScaler` inside a `Pipeline` rather than on a notebook cell and again in the handler?
3. When would `OneHotEncoder(handle_unknown='ignore')` hide schema drift that this laboratory wants to fail loudly?
4. How is training-serving skew different from covariate shift? Can PSI on raw features miss a feature swap?
5. What does a KS p-value assume, and why is it not an alerting policy?
6. Delayed labels: which accuracy is identified, and which is not?
7. What must be retained besides coefficients for \(\hat g_{\text{serve}}=\hat g_{\text{train}}\) to be reconstructable?
8. Why is `schema_version` equality a weaker statement than preprocessor equality?
9. What would a canary add that `select_previous_artifact` does not, and what false sense of safety could a canary create?
10. How would you test serving correctness if the model consumed embeddings from another team’s service?
11. Why log to an MLflow file store without starting the UI in CI?
12. Under what change to the DGP would beating `DummyClassifier` stop being expected, and what would that do to the interpretation of val accuracy?

## Citation

See [`CITATION.cff`](CITATION.cff). Licence: MIT, Copyright 2026 Dr. Pavanam Thomas.
