# Container prediction parity

The serving invariant in this laboratory is narrower than “the API works.” For a schema-valid record, the probability returned by the running service must match the probability from the fitted training `Pipeline` for the same record.

The CI path now tests that invariant across a Docker boundary:

1. train a deterministic small artifact on the runner;
2. build the repository Docker image, which trains its own deterministic artifact from the same code and seed;
3. launch the image and poll `/health` until a model is actually loaded;
4. send a fixed valid record to `/predict-proba`;
5. compare the served probability with the runner-side fitted bundle within an absolute tolerance of `1e-10`;
6. compare model and schema versions as separate contract checks.

The fixed record is intentionally ordinary: age 45, income 80000, credit score 0.72, segment B. The point is not to find a dramatic prediction. It is to test whether two independently instantiated copies of the same deterministic training/serving path agree at the interface boundary.

## First CI failure and correction

The first end-to-end run did not reach the probability comparison. Tests, training, reproduction, and Docker build had passed, but the health probe received `ConnectionResetError` during container startup. The probe treated that reset as terminal.

That was a readiness-classification bug in the test harness, not evidence of model skew. The correction was to treat a startup connection reset like other transient connection failures, retry health polling, and only evaluate prediction parity after `/health` reports `model_loaded=true`. A regression test now forces two connection resets before a successful health response and requires the probe to recover.

Recording this failure matters because readiness and model correctness are different properties. Silencing the reset or simply sleeping for an arbitrary fixed number of seconds would make the test less informative.

## What this supports

- the Docker image builds in CI;
- the container can load its fitted artifact;
- a known schema-valid request crosses the HTTP boundary;
- served and offline probabilities agree for the fixed fixture under the deterministic training path;
- startup resets are retried without being confused with prediction mismatch;
- model and schema versions remain visible at the interface.

## What it does not support

- production uptime or latency claims;
- load testing or concurrency guarantees;
- canary, shadow, or traffic-routing behavior;
- Kubernetes or cloud deployment claims;
- exhaustive parity over the feature space;
- bitwise reproducibility across arbitrary sklearn, BLAS, operating-system, or hardware changes.

The check exists to extend one explicit invariant across one additional boundary. It is not a substitute for production operations evidence.
