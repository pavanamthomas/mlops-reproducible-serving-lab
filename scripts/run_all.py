"""Run the laboratory demonstrations and write figures and a summary table.

All numerical results printed here are simulations under a known DGP.
They are not empirical findings about a real population. This script does
not start the MLflow UI.

Usage, from the repository root::

    python scripts/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlserv.config import load_config
from mlserv.data import generate_dataset, true_probability
from mlserv.failures.concept_drift import (
    delayed_accuracy,
    delayed_label_mask,
    score_on_shifted_concept,
)
from mlserv.failures.data_drift import numeric_shift_report, shift_income
from mlserv.failures.reproducibility import REQUIRED_RETENTION, missing_retention
from mlserv.failures.rollback import ArtifactRecord, select_previous_artifact
from mlserv.failures.schema_drift import add_column, drop_column, schema_drift_report
from mlserv.failures.training_serving_skew import (
    detect_batch_scale_skew,
    detect_feature_swap_skew,
    detect_pooled_scale_skew,
    in_range_swap_frame,
    serve_with_swapped_age_income,
)
from mlserv.failures.version_mismatch import VersionMismatchError, VersionPair, check_versions
from mlserv.monitoring.summaries import (
    distribution_change,
    feature_summaries,
    missingness,
    prediction_distribution,
)
from mlserv.schema import FEATURE_ORDER, require_valid_frame, validate_frame
from mlserv.train import train_and_persist


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    fig_dir = ROOT / "outputs" / "figures"
    tab_dir = ROOT / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = ROOT / "outputs" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlruns = ROOT / "mlruns"

    rows: list[dict[str, object]] = []
    config = load_config(ROOT / "configs" / "train.yaml")
    config = config.replace(
        n_samples=600,
        mlflow_tracking_uri=str(mlruns.resolve()),
    )

    _print_header("A. Synthetic DGP and schema validation")
    batch = generate_dataset(config.n_samples, seed=config.seed)
    violations = validate_frame(batch.X)
    print(f"DGP rows                 = {len(batch.X)}")
    print(f"feature order            = {list(FEATURE_ORDER)}")
    print(f"schema violations        = {len(violations)} (expected 0)")
    print(f"label prevalence         = {batch.y.mean():.3f}")
    print(f"mean true P(y=1)         = {batch.p_true.mean():.3f}")
    print("This is a simulated logistic DGP, not an applicant file.")
    rows.append({"quantity": "n_dgp", "value": len(batch.X)})
    rows.append({"quantity": "label_prevalence", "value": float(batch.y.mean())})

    _print_header("B. Train logistic Pipeline vs DummyClassifier")
    result = train_and_persist(
        config,
        artifact_dir=artifact_dir,
        log_mlflow=True,
    )
    print(f"seed                     = {result.metadata['seed']}")
    print(f"python                   = {result.metadata['python_version']}")
    print(f"sklearn                  = {result.metadata['sklearn_version']}")
    print(f"schema_version           = {result.metadata['schema_version']}")
    print(f"model_version            = {result.metadata['model_version']}")
    print(f"feature_order            = {result.metadata['feature_order']}")
    print(f"n_train / n_val          = {result.metadata['n_train']} / {result.metadata['n_val']}")
    print(
        f"logistic val accuracy    = {result.metrics.accuracy:.3f}  "
        f"log_loss={result.metrics.log_loss:.3f}  "
        f"auc={result.metrics.roc_auc:.3f}  "
        f"brier={result.metrics.brier:.3f}"
    )
    print(
        f"dummy val accuracy       = {result.baseline_metrics.accuracy:.3f}  "
        f"log_loss={result.baseline_metrics.log_loss:.3f}  "
        f"brier={result.baseline_metrics.brier:.3f}"
    )
    print(
        "If logistic accuracy exceeds the dummy here, that is expected: the DGP is "
        "logistic in these features. It is not a claim about other tasks."
    )
    print(f"MLflow file store        = {mlruns} (UI not started)")
    rows.append({"quantity": "val_accuracy", "value": result.metrics.accuracy})
    rows.append({"quantity": "dummy_val_accuracy", "value": result.baseline_metrics.accuracy})
    rows.append({"quantity": "val_brier", "value": result.metrics.brier})

    _print_header("C. Training-serving parity and deliberate skew")
    pipeline = result.pipeline
    X_val = result.X_val
    pooled = detect_pooled_scale_skew(pipeline, result.X_train, X_val)
    batch_skew = detect_batch_scale_skew(pipeline, X_val)
    swap_frame = in_range_swap_frame(len(X_val), seed=2026)
    swapped = detect_feature_swap_skew(pipeline, swap_frame)
    print(
        f"pooled-scaler max |dp|   = {pooled.max_abs_diff:.4f}  "
        f"detector_fired={pooled.detector_fired}"
    )
    print(
        f"batch-scaler max |dp|    = {batch_skew.max_abs_diff:.4f}  "
        f"detector_fired={batch_skew.detector_fired}"
    )
    print(
        f"age/income swap max |dp| = {swapped.max_abs_diff:.4f}  "
        f"detector_fired={swapped.detector_fired}"
    )
    print("Correct serving is pipeline.predict_proba on a named DataFrame.")
    rows.append({"quantity": "skew_pooled_max_abs", "value": pooled.max_abs_diff})
    rows.append({"quantity": "skew_swap_max_abs", "value": swapped.max_abs_diff})

    correct_p = pipeline.predict_proba(swap_frame)[:, 1]
    wrong_p = serve_with_swapped_age_income(pipeline, swap_frame)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(correct_p, wrong_p, s=12, alpha=0.6, c="0.2")
    ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=1)
    ax.set_xlabel("P(y=1) from fitted Pipeline")
    ax.set_ylabel("P(y=1) after age/income value swap")
    ax.set_title("Training-serving skew (synthetic val batch)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_serving_skew.png", dpi=120)
    plt.close(fig)

    _print_header("D. Covariate shift (PSI / KS), not a monitoring platform")
    shifted = shift_income(X_val, log_shift=0.8)
    income_shift = numeric_shift_report(X_val["income"], shifted["income"], feature="income")
    print(
        f"income PSI               = {income_shift.psi:.3f}  "
        f"KS={income_shift.ks_statistic:.3f}  p={income_shift.ks_pvalue:.4f}"
    )
    print("PSI and KS are two-sample descriptive statistics on this batch pair.")
    rows.append({"quantity": "income_psi_shifted", "value": income_shift.psi})

    changes = distribution_change(result.X_train, shifted)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    names = [c.feature for c in changes]
    psis = [c.psi for c in changes]
    ax.bar(names, psis, color="0.35")
    ax.set_ylabel("PSI vs training window")
    ax.set_title("Covariate shift simulation (income multiplied)")
    fig.tight_layout()
    fig.savefig(fig_dir / "psi_shift.png", dpi=120)
    plt.close(fig)

    _print_header("E. Concept drift and delayed labels")
    shifted_eval = score_on_shifted_concept(pipeline, n=400, seed=2027, coef_credit=-2.4)
    original_acc = result.metrics.accuracy
    print(f"original val accuracy    = {original_acc:.3f}")
    print(f"flipped-credit accuracy  = {shifted_eval.accuracy:.3f}  (same Pipeline, new P(Y|X))")
    shifted_batch = generate_dataset(400, seed=2027, coef_credit=-2.4)
    preds = pipeline.predict(shifted_batch.X)
    mask = delayed_label_mask(len(preds), delay_fraction=0.6, seed=2027)
    delayed = delayed_accuracy(shifted_batch.y, preds, mask)
    print(
        f"delayed labels available = {delayed.n_available}/{delayed.n_total}  "
        f"accuracy_on_arrived={delayed.accuracy_available:.3f}"
    )
    print("Accuracy on unarrived labels is not identified.")
    rows.append({"quantity": "concept_shift_accuracy", "value": shifted_eval.accuracy})
    rows.append({"quantity": "delayed_n_available", "value": delayed.n_available})

    pred_orig = prediction_distribution(pipeline, X_val)
    pred_shift = prediction_distribution(pipeline, shifted_batch.X)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(pipeline.predict_proba(X_val)[:, 1], bins=20, alpha=0.6, label="original val", color="0.2")
    ax.hist(
        pipeline.predict_proba(shifted_batch.X)[:, 1],
        bins=20,
        alpha=0.5,
        label="concept-shifted X",
        color="0.6",
    )
    ax.set_xlabel("predicted P(y=1)")
    ax.set_ylabel("count")
    ax.set_title("Prediction histogram (not a concept-drift test by itself)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "prediction_histogram.png", dpi=120)
    plt.close(fig)
    rows.append({"quantity": "mean_p1_val", "value": pred_orig.mean_p1})
    rows.append({"quantity": "mean_p1_concept_shift", "value": pred_shift.mean_p1})

    _print_header("F. Schema drift, version mismatch, rollback")
    missing = schema_drift_report(drop_column(X_val, "credit_score"))
    extra = schema_drift_report(add_column(X_val, "device_id", "x"))
    print(f"missing credit_score     = {missing.missing} ok={missing.ok}")
    print(f"unexpected device_id     = {extra.unexpected} ok={extra.ok}")
    try:
        check_versions(
            VersionPair(model_version="1.0.0", schema_version="1.0"),
            VersionPair(model_version="1.0.0", schema_version="2.0"),
        )
        mismatch = "not-raised"
    except VersionMismatchError as exc:
        mismatch = str(exc)
    print(f"version mismatch         = {mismatch}")
    registry = [
        ArtifactRecord("a0", "0.9.0", "1.0", "2026-01-01T00:00:00Z", "models/a0.joblib"),
        ArtifactRecord("a1", "1.0.0", "1.0", "2026-02-01T00:00:00Z", "models/a1.joblib"),
        ArtifactRecord("a2", "1.1.0", "1.0", "2026-03-01T00:00:00Z", "models/a2.joblib"),
    ]
    previous = select_previous_artifact(registry, "a2")
    print(f"rollback of a2           = {previous.artifact_id} ({previous.model_version})")
    rows.append({"quantity": "rollback_previous_is_a1", "value": int(previous.artifact_id == "a1")})

    _print_header("G. Monitoring summaries and retention list")
    summaries = feature_summaries(X_val)
    miss = missingness(X_val)
    print("numeric summaries (val):")
    for item in summaries:
        print(
            f"  {item.name:14s} mean={item.mean:.3f}  "
            f"p50={item.p50:.3f}  missing={item.missing_rate:.3f}"
        )
    print(f"missingness              = {miss}")
    provided = set(result.metadata) | {"fitted_pipeline"}
    missing_items = missing_retention(provided)
    print(f"required retention       = {len(REQUIRED_RETENTION)} items")
    print(f"missing from this run    = {missing_items if missing_items else '()'}")
    extreme_high = require_valid_frame(
        pd.DataFrame(
            [{"age": 88.0, "income": 250000.0, "credit_score": 0.97, "segment": "C"}]
        )
    )
    extreme_low = require_valid_frame(
        pd.DataFrame(
            [{"age": 22.0, "income": 12000.0, "credit_score": 0.12, "segment": "A"}]
        )
    )
    p_high = float(pipeline.predict_proba(extreme_high)[:, 1][0])
    p_low = float(pipeline.predict_proba(extreme_low)[:, 1][0])
    t_high = float(true_probability(extreme_high)[0])
    t_low = float(true_probability(extreme_low)[0])
    print(f"extreme-high P_hat       = {p_high:.3f}  (true DGP p={t_high:.3f})")
    print(f"extreme-low P_hat        = {p_low:.3f}  (true DGP p={t_low:.3f})")
    rows.append({"quantity": "p_hat_extreme_high", "value": p_high})
    rows.append({"quantity": "p_hat_extreme_low", "value": p_low})

    summary = pd.DataFrame(rows)
    out_csv = tab_dir / "run_summary.csv"
    summary.to_csv(out_csv, index=False)
    print()
    print(f"Wrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote figures under {fig_dir.relative_to(ROOT)}")
    print("Done. MLflow UI was not started.")


if __name__ == "__main__":
    main()
