# Working on this laboratory

This is a personal research repository. The useful unit of work is a limitation, a failed specification, or a tighter check.

1. Open or update a GitHub issue. Name the estimand (usually equality of training-time and serving-time \(\hat g(x)\)), the DGP, and the mismatch.
2. If the claim is numerical, add a test that fails on `main` before the change and passes after.
3. Keep commits narrow. Do not bundle formatting with a scientific or serving-contract change.
4. Comment invariants and failure risks, not obvious syntax.
5. Do not commit `mlruns/`, trained pickles, or `.env` files. Tests train a tiny model in a session fixture.

Recorded failures: `docs/failures_and_corrections.md`.
Queue and bounds: `ROADMAP.md` and GitHub Issues.
Checks: `python -m pytest` and `.github/workflows/ci.yml`.
