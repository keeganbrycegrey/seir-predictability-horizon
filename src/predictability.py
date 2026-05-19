"""
Predictability horizon: first time the ensemble IQR exceeds `threshold.`
× the ensemble median of cumulative incidence R(t).

Using R(t) (recovered = total resolved cases) rather than I(t) because:
  - R(t) is monotone → no spurious re-crossings after epidemic peak
  - IQR/median stays meaningful throughout the full simulation window
  - Early intervention shifts the crossing day reliably

Threshold sensitivity at 15 %, 20 %, 25 % is built in.
"""

import numpy as np


def _iqr_over_median(traj, min_count=500.0):
    q25    = np.percentile(traj, 25, axis=0)
    q75    = np.percentile(traj, 75, axis=0)
    iqr    = q75 - q25
    median = np.median(traj, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(median >= min_count, iqr / median, np.nan)
    return ratio


def compute_horizon(R_traj, t, threshold=0.20, min_count=500.0):
    ratio      = _iqr_over_median(R_traj, min_count)
    candidates = np.where(~np.isnan(ratio) & (ratio > threshold))[0]
    return float(t[candidates[0]]) if len(candidates) else np.nan


def compute_horizon_sensitivity(R_traj, t, min_count=500.0):
    return {
        "h15": compute_horizon(R_traj, t, 0.15, min_count),
        "h20": compute_horizon(R_traj, t, 0.20, min_count),
        "h25": compute_horizon(R_traj, t, 0.25, min_count),
    }


def divergence_curve(R_traj, min_count=500.0):
    return _iqr_over_median(R_traj, min_count)
