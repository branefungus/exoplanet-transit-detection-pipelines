"""Recovery grading and statistics helpers.

Change from the old code: per-bin proportion intervals now use the Wilson score
interval instead of Wald. Wald collapses near rates of 0 or 1 with small n, which
is exactly the regime of the hardest bins. The paired-delta interval and the
exact McNemar test are unchanged.
"""
from __future__ import annotations

import math
import numpy as np

import config as C


# Pipeline-agnostic detection statistic

def detection_snr(t, y, period, t0, duration_days, depth_found) -> float:
    """A common, pipeline-agnostic measure of how convincing the recovered
    transit is: recovered depth over the out-of-transit scatter of the
    normalized flux. Computed the same way for every pipeline (and for null
    trials), so all five land on one false-positive curve.

    Returns nan if the ephemeris is unusable or there aren't enough points.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    P = float(period)
    if not np.isfinite(P) or P <= 0 or not np.isfinite(t0) or not np.isfinite(depth_found):
        return float("nan")
    med = np.nanmedian(y)
    if not np.isfinite(med) or med == 0:
        return float("nan")
    yn = y / med

    phase = ((t - float(t0) + 0.5 * P) % P) - 0.5 * P
    half = 0.5 * float(duration_days)
    out = np.abs(phase) > (1.5 * half)          # clearly out of transit
    out &= np.isfinite(yn)
    if np.sum(out) < 20:
        return float("nan")

    resid = yn[out] - np.nanmedian(yn[out])
    mad = np.nanmedian(np.abs(resid))
    sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(resid))
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(abs(depth_found) / sigma)


# Grading

def t0_mod_distance(t0_found: float, t0_true: float, period: float) -> float:
    """Timing error, wrapped into the interval [-P/2, P/2) so that being off by a
    whole number of periods doesn't count against the fit."""
    dt = ((t0_found - t0_true + 0.5 * period) % period) - 0.5 * period
    return abs(float(dt))


def evaluate_recovery(period_true, t0_true, depth_true,
                      period_found, t0_found, depth_found) -> tuple[int, int, int, int]:
    """Returns (pass, p_ok, t_ok, d_ok) as ints."""
    p_ok = abs(period_found - period_true) / period_true <= C.PERIOD_FRAC_TOL
    t_ok = t0_mod_distance(t0_found, t0_true, period_true) <= C.T0_ABS_TOL_DAYS
    d_ok = False
    if depth_true > 0 and np.isfinite(depth_found):
        d_ok = (C.DEPTH_FACTOR_LOW * depth_true) <= depth_found <= (C.DEPTH_FACTOR_HIGH * depth_true)
    return int(bool(p_ok and t_ok and d_ok)), int(p_ok), int(t_ok), int(d_ok)


# Confidence intervals

def wilson_ci(k: int, n: int, z: float = C.CI_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def paired_delta_ci(a: np.ndarray, b: np.ndarray, z: float = C.CI_Z):
    """CI for the mean of the paired differences b - a (each in {0, 1}).
    Returns (mean, lo, hi, n)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d[np.isfinite(d)]
    n = int(len(d))
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(d))
    if n == 1:
        return (mean, mean, mean, n)
    se = math.sqrt(float(np.var(d, ddof=1)) / n)
    return (mean, mean - z * se, mean + z * se, n)


# Exact McNemar (log-sum-exp for numerical stability with large counts)

def _log_binom_pmf_half(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            - n * math.log(2.0))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value via binomial(b + c, 0.5)."""
    n = int(b + c)
    if n <= 0:
        return float("nan")
    x = int(min(b, c))
    logs = [_log_binom_pmf_half(n, k) for k in range(0, x + 1)]
    m = max(logs)
    p_lower = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return float(min(1.0, 2.0 * p_lower))


def discordant_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    return a_only, b_only
