# Data policy

No credit files or production logs. Rows come from the logistic DGP in `src/mlserv/data.py` so serving skew can be measured against a known Pipeline.

## What is used

All rows are **simulated** from the logistic DGP in `src/mlserv/data.py`.
Randomness is controlled through `numpy.random.Generator` with documented
seeds (default `2026` unless a test uses a second seed so that two arms
are independent).

No file in `data/` is required. No download script is required.

Features (`age`, `income`, `credit_score`, `segment`) are names of
simulated columns. They are not measurements of people.

## What is not claimed

Val accuracy, Brier score, PSI, KS, and delayed accuracy describe the
behaviour of a procedure under this DGP. They are not estimates for a
lending book, a production model, or a published empirical study.

## Regeneration

Parity dumps under `outputs/` are session artifacts from `scripts/run_all.py`, not a served production log.

## Third-party code

The package depends on NumPy, pandas, scikit-learn, SciPy, matplotlib,
FastAPI, Pydantic, MLflow, PyYAML, joblib, pytest, httpx, and uvicorn
under their respective licences. This repository does not copy vendor
tutorials or copyrighted worked examples into `docs/`.
