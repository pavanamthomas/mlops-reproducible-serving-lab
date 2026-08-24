"""PSI and Kolmogorov–Smirnov on features under a known shift.

Cheap two-sample numbers, not a retraining policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class UnivariateShift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    n_ref: int
    n_cur: int


def population_stability_index(
    reference: ArrayLike,
    current: ArrayLike,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI using quantile bins of the reference sample.

    PSI = sum_j (p_cur,j - p_ref,j) * log(p_cur,j / p_ref,j).
    Empty bins are clipped to ``epsilon``. That clip is a numerical
    patch, not a statistical model of rare categories.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cur = np.asarray(current, dtype=float).reshape(-1)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must be non-empty")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = np.clip(ref_counts / ref.size, epsilon, None)
    cur_p = np.clip(cur_counts / cur.size, epsilon, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_feature(reference: ArrayLike, current: ArrayLike) -> tuple[float, float]:
    """Two-sample KS statistic and p-value. Not a sequential detector."""
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cur = np.asarray(current, dtype=float).reshape(-1)
    if ref.size < 2 or cur.size < 2:
        raise ValueError("need at least two observations in each sample")
    result = stats.ks_2samp(ref, cur, method="auto")
    return float(result.statistic), float(result.pvalue)


def numeric_shift_report(
    reference: NDArray[np.floating] | ArrayLike,
    current: NDArray[np.floating] | ArrayLike,
    feature: str,
    n_bins: int = 10,
) -> UnivariateShift:
    psi = population_stability_index(reference, current, n_bins=n_bins)
    ks_stat, ks_p = ks_feature(reference, current)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cur = np.asarray(current, dtype=float).reshape(-1)
    return UnivariateShift(
        feature=feature,
        psi=psi,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        n_ref=int(ref.size),
        n_cur=int(cur.size),
    )


def shift_income(
    frame,
    *,
    log_shift: float,
):
    """Copy a feature frame with a multiplicative income shift (covariate shift)."""
    out = frame.copy()
    out["income"] = out["income"] * float(np.exp(log_shift))
    return out
