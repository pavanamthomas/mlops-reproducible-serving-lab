"""Train the synthetic laboratory model and write a local artifact.

Does not start the MLflow UI. Use ``--skip-mlflow`` in CI smoke runs.

Usage, from the repository root::

    python scripts/train.py
    python scripts/train.py --n-samples 120 --skip-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlserv.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mlserv on the synthetic DGP.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Write the joblib bundle only. Do not log to the file store.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Default comes from the YAML config.",
    )
    args = parser.parse_args()

    result = train_from_config(
        args.config,
        artifact_dir=args.artifact_dir,
        n_samples=args.n_samples,
        log_mlflow=not args.skip_mlflow,
        tracking_uri=args.tracking_uri,
    )
    print(f"artifact_id     = {result.metadata['artifact_id']}")
    print(f"model_version   = {result.metadata['model_version']}")
    print(f"schema_version  = {result.metadata['schema_version']}")
    print(f"seed            = {result.metadata['seed']}")
    print(f"n_train         = {result.metadata['n_train']}")
    print(f"n_val           = {result.metadata['n_val']}")
    print(
        f"val accuracy    = {result.metrics.accuracy:.3f}  "
        f"(dummy most-frequent = {result.baseline_metrics.accuracy:.3f})"
    )
    print(
        "These scores are for the synthetic DGP in mlserv.data. "
        "They are not a claim about another task."
    )
    if result.artifact_path is not None:
        print(f"wrote {result.artifact_path}")
    if args.skip_mlflow:
        print("MLflow logging skipped. The MLflow UI was not started.")
    else:
        print("Logged params/metrics/model to the local MLflow file store. UI not started.")


if __name__ == "__main__":
    main()
