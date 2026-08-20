# Flagship: training-serving skew

## The question

Can a model be “deployed” — health check green, JSON validated, HTTP 200 — and still return the wrong number?

Yes. The estimand is not “the server responded.” It is

\[
\hat g_{\text{serve}}(x)=\hat g_{\text{train}}(x)
\]

for records \(x\) that pass schema `1.0`. \(\hat g_{\text{train}}\) is the fitted sklearn `Pipeline`. Anything else is a different function.

## The DGP

Rows are simulated (`mlserv.data`). Labels are Bernoulli with a logistic mean in `age`, log `income`, `credit_score`, and `segment`. This is not an applicant file. The DGP is known so that a wrong transform can be compared to a right one.

## What “works”

`POST /predict-proba` accepts

```json
{"age": 40.0, "income": 40.0, "credit_score": 0.5, "segment": "A"}
```

Pydantic is satisfied: types are right, ranges are right, `segment` is in {A, B, C}. The handler can return 200. None of that checks the preprocessor.

## What is wrong

Three serving paths are implemented as failures in `src/mlserv/failures/training_serving_skew.py`:

1. Refit `StandardScaler` on train+serve rows, then apply the trained logistic head.
2. Refit `StandardScaler` on the serving batch only (`fit_transform` in the handler).
3. Write income values into the `age` column and conversely. Names stay legal.

In each case the API *could* still answer. The probabilities are not the Pipeline’s probabilities. `scripts/run_all.py` writes `outputs/figures/training_serving_skew.png` for the swap.

A monitoring PSI on raw features can miss (3) when both columns remain in range. Skew is not the same object as covariate shift. See `docs/drift.md` and `docs/monitoring_limits.md`.

## What catches it

An automated parity test: serving probabilities versus `pipeline.predict_proba` on the same named DataFrame. Tolerance \(10^{-12}\) on the correct path. The skew detector requires `max_abs_diff` above a small tolerance on each wrong path.

Locked by `tests/test_parity.py` and `tests/test_skew.py`.

## Correction

Serialise one `Pipeline`. Load that object. Reindex columns to the recorded feature order. Do not fit a scaler in the handler. Do not treat HTTP 200 as \(\hat g_{\text{serve}}=\hat g_{\text{train}}\).

## Non-claims

This flagship is a constructed counterexample on a synthetic DGP. It is not an incident report. It does not measure how often production systems make this mistake. The Docker image is not a production deployment of a fix.
