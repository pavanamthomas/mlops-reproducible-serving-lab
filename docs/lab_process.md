# How this lab records work

Serving claims here are claims that HTTP output equals the fitted training `Pipeline` on the same schema-valid input. Write the skew (column swap, dropped step, stale artifact) before changing the API.

If the claim is numerical, add a test that would fail if health were treated as parity. CI on `main` trains an artifact, builds the Docker image, waits for model readiness (retrying `ConnectionResetError` on first probe), and compares `/predict-proba` with the offline Pipeline. Passing CI is not a production SLO. The MLflow UI is not started.

Issues are the public queue. `ROADMAP.md` is the bound. A green badge is not Kubernetes.
