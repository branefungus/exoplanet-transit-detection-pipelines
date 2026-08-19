"""Recovery-side limb-darkened transit template and depth fitting.

This is the template the pipelines use to estimate depth (and, in P4, to jointly
fit the transit). It's intentionally a simpler small-planet chord model than the
numerically integrated occultation used for injection (see injection.py), so the
benchmark isn't a perfect matched-filter test. Same logic as the old per-runner
copies.
"""
from __future__ import annotations

import numpy as np

import config as C


def ld_intensity(mu: np.ndarray, u1: float = C.LD_U1, u2: float = C.LD_U2) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), 0.0, 1.0)
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    return np.clip(intensity, 0.0, None)


def phase_days_from_t(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    P = float(period)
    return ((t - float(t0) + 0.5 * P) % P) - 0.5 * P


def ld_shape_unit(t: np.ndarray, period: float, t0: float, duration_days: float,
                  b: float = C.LD_B_DEFAULT, u1: float = C.LD_U1, u2: float = C.LD_U2,
                  k_ref: float = C.LD_K_REF) -> np.ndarray:
    """Unit transit template: 1 at mid-transit, 0 outside, with a rounded floor
    from limb darkening."""
    t = np.asarray(t, dtype=float)
    dur = float(duration_days)
    if not np.isfinite(dur) or dur <= 0:
        return np.zeros_like(t, dtype=float)

    ph = phase_days_from_t(t, period=period, t0=t0)
    half = 0.5 * dur
    shape = np.zeros_like(ph, dtype=float)
    inwin = np.abs(ph) <= half
    if not np.any(inwin):
        return shape

    b = float(np.clip(b, 0.0, 0.95))
    k_ref = float(np.clip(k_ref, 0.001, 0.30))
    x_edge = np.sqrt(max(1e-12, (1.0 + k_ref) ** 2 - b * b))
    s = ph[inwin] / half
    x = x_edge * s
    d = np.minimum(np.sqrt(b * b + x * x), 1.0)
    mu = np.sqrt(np.clip(1.0 - d * d, 0.0, 1.0))
    intensity = ld_intensity(mu, u1=u1, u2=u2)

    # Normalize so the template is exactly 1 at mid-transit.
    mu_mid = np.sqrt(max(1e-12, 1.0 - b * b))
    I_mid = float(ld_intensity(np.array([mu_mid]), u1=u1, u2=u2)[0])
    if not np.isfinite(I_mid) or I_mid <= 0:
        I_mid = 1.0
    shape[inwin] = intensity / I_mid
    return shape


def fit_baseline_and_depth_ld(t: np.ndarray, y: np.ndarray, period: float, t0: float,
                              duration_days: float, b: float = C.LD_B_DEFAULT,
                              window_factor: float = 2.5) -> tuple[float, float]:
    """Least-squares fit of y ~ baseline - depth * shape on a window around the
    transit. Returns (baseline, depth); depth is nan when it can't be solved or
    comes out non-positive."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dur = float(duration_days)
    if not np.isfinite(dur) or dur <= 0:
        return float("nan"), float("nan")

    ph = phase_days_from_t(t, period=period, t0=t0)
    use = np.abs(ph) <= (window_factor * dur)
    if np.sum(use) < 30:
        use = np.isfinite(t) & np.isfinite(y)

    tt, yy = t[use], y[use]
    good = np.isfinite(tt) & np.isfinite(yy)
    tt, yy = tt[good], yy[good]
    if len(tt) < 30:
        return float("nan"), float("nan")

    shape = ld_shape_unit(tt, period=period, t0=t0, duration_days=dur, b=b)
    A = np.column_stack([np.ones_like(shape), -shape])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    baseline, depth = float(beta[0]), float(beta[1])
    if not np.isfinite(depth) or depth <= 0:
        depth = float("nan")
    return baseline, depth
