# What's still open

Docker prediction-parity is in CI as of August 2026. One fixture, one request. That is the check.

- Canary and shadow traffic: not here. Rollback is still “previous artifact id,” not a router.
- Broader sampling across the valid schema: not implemented. I have the one deterministic request.
- Hosted registry, object-store provenance, signed artifacts, access control: outside this local lab.
- Cross-version sklearn/joblib pickle: I record versions so a mismatch is visible. I do not promise they load.
- Calibration under shift as a deployment policy: no. Brier on synthetic splits only.

Not doing Kubernetes, service meshes, or invented SLOs. Logistic beating the dummy is expected on this DGP. PSI/KS are not an observability stack.

See `docs/failures_and_corrections.md` and `docs/container_parity.md`.
