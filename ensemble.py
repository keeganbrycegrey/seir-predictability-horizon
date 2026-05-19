"""
Parameter sampling for SEIR ensemble.

Baseline ranges from the specification (generic respiratory / COVID-adjacent):
  β  : 0.25 – 0.40   (center 0.325, half-width 0.075)
  γ  : 0.08 – 0.12   (center 0.100, half-width 0.020)
  σ  : 0.15 – 0.25   (center 0.200, half-width 0.050)

Uncertainty scale s=1.0 reproduces these ranges exactly.
s=0.5 halves the width; s=2.0 doubles it.
"""

import numpy as np

# Parameter bounds (spec-defined)
_CENTERS = {
    "beta":  (0.25 + 0.40) / 2,   # 0.3250
    "gamma": (0.08 + 0.12) / 2,   # 0.1000
    "sigma": (0.15 + 0.25) / 2,   # 0.2000
}
_HALF_WIDTHS = {
    "beta":  (0.40 - 0.25) / 2,   # 0.0750
    "gamma": (0.12 - 0.08) / 2,   # 0.0200
    "sigma": (0.25 - 0.15) / 2,   # 0.0500
}

BASELINE = {k: _CENTERS[k] for k in _CENTERS}


def sample_parameters(n_runs, uncertainty_scale=1.0, rng=None):
    """
    Sample β, σ, γ uniformly from scaled uncertainty envelopes.

    Parameters
    n_runs : int
    uncertainty_scale : float
        Multiplier on nominal half-widths. 1.0 = literature ranges.
    rng : numpy.random.Generator, optional

    Returns
    dict with keys 'beta', 'sigma', 'gamma', each ndarray of shape (n_runs,).
    """
    if rng is None:
        rng = np.random.default_rng()

    params = {}
    for key in ("beta", "gamma", "sigma"):
        c  = _CENTERS[key]
        hw = _HALF_WIDTHS[key] * uncertainty_scale
        lo = max(c - hw, 1e-9)
        hi = c + hw
        params[key] = rng.uniform(lo, hi, n_runs)

    return params


def implied_R0(params):
    """compute R₀ = β / γ for each ensemble member."""
    return params["beta"] / params["gamma"]
