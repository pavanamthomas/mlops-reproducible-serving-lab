# Reproducibility

The estimand for serving is equality of functions, not a leaderboard score:

\[
\hat g_{\text{serve}}(x) = \hat g_{\text{train}}(x)
\]

for every record \(x\) that passes schema version `1.0`. A pickle of logistic coefficients does not identify that equality. The scaler means, the one-hot category order, and the feature names have to be the same object.

## What this laboratory retains

Recorded on every artifact (see `mlserv.failures.reproducibility.REQUIRED_RETENTION`):

- seed (`2026` unless a second stream is required)
- Python, sklearn, NumPy, pandas versions
- feature names and order
- `schema_version`, `model_version`, hyperparameters
- train/val counts and the DGP name
- the fitted sklearn `Pipeline` (preprocessor + classifier)
- an `artifact_id`

The stratified split is identified by the seed and `test_size`, not by a committed index file.

## What is not retained

- The MLflow UI process. CI logs to a file store when `scripts/run_all.py` runs and does not start the UI.
- Training rows. The DGP is code. Anyone can regenerate an iid sample; they will not recover the exact same 800 rows unless they use the same seed and generator path.
- A content hash of the Pydantic models. Version strings are the contract used here.

## How to reconstruct a run

```bash
python -m pip install -e .
python scripts/train.py --skip-mlflow
python -m pytest
python scripts/run_all.py
```

`mlruns/` is gitignored. A clean clone does not contain previous file-store experiments. That is intended. The source of truth is the code, the YAML config, and the tests.

MLflow 3.x marks the filesystem backend as maintenance-mode. The laboratory sets `MLFLOW_ALLOW_FILE_STORE=true` when logging so that a local `file://` store still works without standing up SQLite or a server. That is an opt-in to a local directory, not a claim that the file store is MLflow's recommended production backend.

## What reproducibility is not

Identical scores on this DGP are not evidence that a production model is reproducible. Library upgrades can change sklearn's logistic solver path; the recorded `sklearn_version` is there so that mismatch is visible, not so that it is automatically repaired.
