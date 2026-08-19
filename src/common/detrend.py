"""Shared detrending helpers and the BLS search core.

These were deduplicated out of the five old runner scripts; the logic is
unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

import config as C

DURATIONS_DAYS = np.asarray(C.BLS_DURATIONS_DAYS, dtype=float)


def robust_sigma(y) -> float:
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(y))
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else 1e-6


def detrend_rolling_median_subtract(t, y, window_days=1.5):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Bad cadence estimate in detrend_rolling_median_subtract.")
    win = max(7, int(round(window_days / dt)))
    trend = pd.Series(y).rolling(window=win, center=True,
                                 min_periods=max(3, win // 3)).median().to_numpy()
    med = np.nanmedian(y)
    trend = np.where(np.isfinite(trend), trend, med)
    return y - trend


def transit_mask(t, period, t0, duration, pad_factor=3.0):
    t = np.asarray(t, dtype=float)
    half = 0.5 * float(duration) * pad_factor
    phase = ((t - float(t0) + 0.5 * float(period)) % float(period)) - 0.5 * float(period)
    return np.abs(phase) <= half


def segment_ids_from_gaps(t):
    """Label points by contiguous segment, splitting wherever there's a gap
    much larger than the typical cadence."""
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return np.array([], dtype=int)
    dt = np.diff(t)
    cadence = float(np.nanmedian(dt)) if len(dt) else 0.0
    thr = max(0.5, 20.0 * cadence) if cadence > 0 else 0.5
    seg = np.zeros(len(t), dtype=int)
    current = 0
    for i in range(1, len(t)):
        if dt[i - 1] > thr:
            current += 1
        seg[i] = current
    return seg


def build_design_matrix(t):
    """Trend basis of a cubic polynomial, two long-period sinusoids, and a
    per-segment offset. Used by P1's regression detrend and P4's joint fit."""
    t = np.asarray(t, dtype=float)
    n = len(t)

    t_mid = float(np.nanmedian(t))
    span = float(np.nanmax(t) - np.nanmin(t))
    span = span if np.isfinite(span) and span > 0 else 1.0
    u = (t - t_mid) / span

    cols = [np.ones(n, dtype=float), u, u ** 2, u ** 3]
    intercept_like = [True, False, False, False]

    for per in (5.0, 10.0):
        ang = 2.0 * np.pi * (t / per)
        cols += [np.sin(ang), np.cos(ang)]
        intercept_like += [False, False]

    seg = segment_ids_from_gaps(t)
    nseg = int(np.max(seg) + 1) if len(seg) else 1
    if nseg > 1:
        for s in range(1, nseg):
            cols.append((seg == s).astype(float))
            intercept_like.append(True)

    return np.column_stack(cols), np.array(intercept_like, dtype=bool)


def ridge_solve(X, y, alpha, intercept_like_cols):
    """Ridge regression with the non-intercept columns standardized and the
    intercept-like columns left unregularized. Coefficients come back in the
    original (unscaled) units."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    means = np.zeros(p, dtype=float)
    stds = np.ones(p, dtype=float)
    Xs = X.copy()
    for j in range(p):
        if intercept_like_cols[j]:
            continue
        m = float(np.nanmean(Xs[:, j]))
        s = float(np.nanstd(Xs[:, j]))
        if not np.isfinite(s) or s < 1e-12:
            s = 1.0
        means[j], stds[j] = m, s
        Xs[:, j] = (Xs[:, j] - m) / s

    reg = np.full(p, float(alpha), dtype=float)
    reg[intercept_like_cols] = 0.0
    A = np.vstack([Xs, np.diag(np.sqrt(reg))])
    b = np.concatenate([y, np.zeros(p, dtype=float)])
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Undo the standardization, then fold the shifts back into the intercept.
    beta_un = beta.copy()
    for j in range(p):
        if not intercept_like_cols[j]:
            beta_un[j] = beta_un[j] / stds[j]
    for j in range(p):
        if not intercept_like_cols[j]:
            beta_un[0] -= (means[j] / stds[j]) * beta[j]
    return beta_un


def ridge_fit_predict(X_fit, y_fit, X_all, alpha=1e-2, intercept_like_cols=None):
    """Fit ridge on (X_fit, y_fit) and predict on X_all. This is P1's trend
    regression: fit on the kept points, predict everywhere."""
    X_fit = np.asarray(X_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    X_all = np.asarray(X_all, dtype=float)

    n, p = X_fit.shape
    if intercept_like_cols is None:
        intercept_like_cols = np.zeros(p, dtype=bool)
        intercept_like_cols[0] = True

    Xf, Xa = X_fit.copy(), X_all.copy()
    for j in range(p):
        if intercept_like_cols[j]:
            continue
        m = float(np.nanmean(Xf[:, j]))
        s = float(np.nanstd(Xf[:, j]))
        if not np.isfinite(s) or s < 1e-12:
            s = 1.0
        Xf[:, j] = (Xf[:, j] - m) / s
        Xa[:, j] = (Xa[:, j] - m) / s

    reg = np.full(p, float(alpha), dtype=float)
    reg[intercept_like_cols] = 0.0
    A = np.vstack([Xf, np.diag(np.sqrt(reg))])
    b = np.concatenate([y_fit, np.zeros(p, dtype=float)])
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return Xa @ beta


# BLS search core (shared by P0-P3 and by P4's seeding)

def _best_from_power(power):
    k = int(np.nanargmax(power.power))
    return (float(power.period[k]), float(power.transit_time[k]),
            abs(float(power.depth[k])), float(power.power[k]))


def run_bls_core(t, y_det):
    """Autopower search, then a harmonic check (half/double the best period),
    then a local refinement grid around the winner. Unchanged from the old code."""
    t = np.asarray(t, dtype=float)
    y_det = np.asarray(y_det, dtype=float)
    bls = BoxLeastSquares(t, y_det)

    power_auto = bls.autopower(
        DURATIONS_DAYS,
        minimum_period=C.PERIOD_MIN,
        maximum_period=C.PERIOD_MAX,
        frequency_factor=C.BLS_FREQUENCY_FACTOR,
    )
    P_best, t0_best, depth_best, score_best = _best_from_power(power_auto)

    # Harmonic check: does half or double the period score better?
    cand = np.array(sorted({P_best, P_best / 2.0, P_best * 2.0}), dtype=float)
    cand = cand[(cand >= C.PERIOD_MIN) & (cand <= C.PERIOD_MAX)]
    if len(cand) > 1:
        P2, t02, d2, s2 = _best_from_power(bls.power(cand, DURATIONS_DAYS))
        if s2 > score_best:
            P_best, t0_best, depth_best, score_best = P2, t02, d2, s2

    # Local refinement in a +/-2% window around the current best.
    lo = max(C.PERIOD_MIN, P_best * 0.98)
    hi = min(C.PERIOD_MAX, P_best * 1.02)
    if hi > lo:
        grid = np.linspace(lo, hi, 1500)
        Pr, t0r, dr, sr = _best_from_power(bls.power(grid, DURATIONS_DAYS))
        if sr > score_best:
            P_best, t0_best, depth_best, score_best = Pr, t0r, dr, sr

    return P_best, t0_best, depth_best, score_best


def pick_top_periods_from_autopower(power, k=6):
    """Top-k periods by BLS power, keeping them at least 2% apart so we don't
    return near-duplicates of the same peak."""
    periods = np.asarray(power.period, dtype=float)
    scores = np.asarray(power.power, dtype=float)
    order = np.argsort(scores)[::-1]
    chosen: list[float] = []
    for idx in order:
        p = float(periods[idx])
        if not (C.PERIOD_MIN <= p <= C.PERIOD_MAX):
            continue
        if all(abs(p - q) / q >= 0.02 for q in chosen):
            chosen.append(p)
        if len(chosen) >= k:
            break
    if not chosen:
        chosen = [float(periods[int(order[0])])]
    return chosen


def bls_seed_periods_and_t0(t, y_det, topk=6):
    bls = BoxLeastSquares(np.asarray(t, float), np.asarray(y_det, float))
    power_auto = bls.autopower(
        DURATIONS_DAYS,
        minimum_period=C.PERIOD_MIN,
        maximum_period=C.PERIOD_MAX,
        frequency_factor=C.BLS_FREQUENCY_FACTOR,
    )
    seeds = []
    for p in pick_top_periods_from_autopower(power_auto, k=topk):
        pow_p = bls.power(np.array([p], dtype=float), DURATIONS_DAYS)
        seeds.append((float(p), float(pow_p.transit_time[0])))
    return seeds
