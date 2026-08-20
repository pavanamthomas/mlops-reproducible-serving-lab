# Training-serving skew

Flagship write-up: [`FLAGSHIP_TRAINING_SERVING_SKEW.md`](../FLAGSHIP_TRAINING_SERVING_SKEW.md).

## Estimand

Let \(\hat g\) be the fitted sklearn `Pipeline` (StandardScaler + OneHotEncoder + logistic regression). Correct serving is \(\hat g(x)\) for a named record \(x\). Skew is any other transform \(T\) such that \(T(x) \neq \hat g(x)\) on a set of positive measure under the serving distribution.

HTTP 200 does not identify \(T = \hat g\). Pydantic acceptance does not identify it either. Both only identify that the JSON parsed and the fields lay inside the recorded ranges.

## Failures constructed here

1. **Pooled scaler.** Numeric means and standard deviations are computed on train+serve, then the *trained* classifier is applied. Training used train-only statistics.
2. **Batch scaler.** A new `StandardScaler` is fit on the serving batch alone (`fit_transform` in a handler).
3. **Named-column value swap.** Age values are written into `income` and conversely. Names and ranges can still be valid, so the API still answers.

Code: `src/mlserv/failures/training_serving_skew.py`.  
Parity helper used by the API: `src/mlserv/api/serving.py`.

## Check

`tests/test_parity.py` requires max absolute probability difference \(< 10^{-12}\) between `pipeline.predict_proba` and the serving helper.

`tests/test_skew.py` requires the detector to fire (`max_abs_diff > tolerance`) on each wrong transform.

## Correction

Serve the fitted `Pipeline` as one object. Do not re-fit preprocessing in the handler. Do not convert to a numpy array and guess column order.

## What remains unknown

Whether a given production stack (feature store, streaming join, CSV dump) implements the pooled-scaler bug or a different one. The laboratory shows that the bug is compatible with a green health check.
