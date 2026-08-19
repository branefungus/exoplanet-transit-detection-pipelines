"""The five recovery pipelines. Each one exposes

    run(t, y, duration_days, b_ld) -> (period, t0, depth, score)

The algorithms are identical to the old 10_..14_ runner scripts; only the shared
helpers were pulled out into src/common. The registry at the bottom is what
scripts/03_run_pipeline.py and the visual diagnostics import.

A note on the inputs: the pipelines receive duration_days (the true injected
duration) and b_ld (the true impact parameter) from the trial index. That's a
deliberate, disclosed benchmark choice: it isolates period/t0/depth recovery and
treats duration as known. This is stated in the paper's Methods.
"""
from __future__ import annotations

import numpy as np

import config as C
from src.common.detrend import (
    detrend_rolling_median_subtract, run_bls_core, transit_mask, robust_sigma,
    build_design_matrix, ridge_fit_predict, ridge_solve, bls_seed_periods_and_t0,
    DURATIONS_DAYS,
)
from src.common.transit_template import ld_shape_unit, fit_baseline_and_depth_ld


def _normalize(y):
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y)
    if not np.isfinite(med) or med == 0:
        raise ValueError("Bad median flux.")
    return y / med


def _ld_depth_or_bls(t, y_norm, P, t0, duration_days, b_ld, depth_bls):
    """Prefer the limb-darkened template depth; fall back to the BLS box depth if
    the template fit didn't produce a finite value."""
    _, depth_ld = fit_baseline_and_depth_ld(
        t, y_norm, period=P, t0=t0, duration_days=duration_days, b=b_ld)
    return depth_ld if np.isfinite(depth_ld) else depth_bls


# P0: rolling-median detrend -> BLS -> LD depth refit

def run_P0(t, y, duration_days: float, b_ld: float):
    t = np.asarray(t, dtype=float)
    y_norm = _normalize(y)
    y_det = detrend_rolling_median_subtract(t, y_norm, window_days=1.5)
    P, t0, depth_bls, score = run_bls_core(t, y_det)
    depth = _ld_depth_or_bls(t, y_norm, P, t0, duration_days, b_ld, depth_bls)
    return P, t0, float(depth), score


# P1: ridge feature-regression detrend -> BLS -> LD depth refit

def run_P1(t, y, duration_days: float, b_ld: float):
    t = np.asarray(t, dtype=float)
    y_norm = _normalize(y)

    # First pass just to locate the transit so we can mask it out of the fit.
    y_det0 = detrend_rolling_median_subtract(t, y_norm, window_days=1.5)
    P0, t00, _, _ = run_bls_core(t, y_det0)

    in_tr = transit_mask(t, P0, t00, duration_days, pad_factor=3.0)
    keep = ~in_tr

    # Also drop obvious outliers among the kept points before fitting the trend.
    y_keep = y_norm[keep]
    mu = float(np.nanmedian(y_keep))
    sig = robust_sigma(y_keep - mu)
    keep &= np.abs(y_norm - mu) <= (6.0 * sig)
    if np.sum(keep) < max(80, int(0.6 * len(t))):
        keep = np.ones_like(t, dtype=bool)

    X_all, intercept_like = build_design_matrix(t)
    trend = ridge_fit_predict(X_fit=X_all[keep], y_fit=y_norm[keep], X_all=X_all,
                              alpha=1e-2, intercept_like_cols=intercept_like)
    y_res = y_norm - trend
    y_res = y_res - np.nanmedian(y_res)

    P, t0, depth_bls, score = run_bls_core(t, y_res)
    depth = _ld_depth_or_bls(t, y_norm, P, t0, duration_days, b_ld, depth_bls)
    return P, t0, float(depth), score


# P2: masked long trend -> optional wavelet denoise -> BLS -> LD depth refit

def _masked_long_trend(t, y, in_tr_mask, window_days_long=1.5):
    """Rolling-median trend with the in-transit points masked out and filled by
    interpolation, so the transit itself doesn't bias the trend."""
    import pandas as pd
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Bad cadence estimate inside masked_long_trend.")
    win = max(9, int(round(window_days_long / dt)))
    s = pd.Series(y.copy())
    s[np.asarray(in_tr_mask, dtype=bool)] = np.nan
    s = s.interpolate(limit_direction="both")
    trend = s.rolling(window=win, center=True, min_periods=max(5, win // 3)).median().to_numpy()
    med = np.nanmedian(y)
    return np.where(np.isfinite(trend), trend, med)


def _wavelet_denoise_optional(y, wavelet="db4", level=3):
    """Soft-threshold wavelet denoise using a universal (VisuShrink) threshold.
    If pywt isn't installed, this is a no-op and returns y unchanged."""
    try:
        import pywt  # type: ignore
    except Exception:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = pywt.wavedec(y, wavelet, mode="periodization", level=level)
    detail = coeffs[-1]
    mad = np.nanmedian(np.abs(detail - np.nanmedian(detail)))
    sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(detail))
    if not np.isfinite(sigma) or sigma <= 0:
        return y
    n = len(y)
    thresh = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(cD, thresh, mode="soft") for cD in coeffs[1:]]
    y_hat = pywt.waverec(new_coeffs, wavelet, mode="periodization")
    return np.asarray(y_hat[:n], dtype=float)


def run_P2(t, y, duration_days: float, b_ld: float):
    t = np.asarray(t, dtype=float)
    y_norm = _normalize(y)

    y_det0 = detrend_rolling_median_subtract(t, y_norm, window_days=1.5)
    P0, t00, _, _ = run_bls_core(t, y_det0)

    in_tr = transit_mask(t, P0, t00, duration_days, pad_factor=3.0)
    trend = _masked_long_trend(t, y_norm, in_tr, window_days_long=1.5)
    y_res = y_norm - trend
    y_res = y_res - np.nanmedian(y_res)
    y_res_dn = _wavelet_denoise_optional(y_res, wavelet="db4", level=3)

    P, t0, depth_bls, score = run_bls_core(t, y_res_dn)
    depth = _ld_depth_or_bls(t, y_norm, P, t0, duration_days, b_ld, depth_bls)
    return P, t0, float(depth), score


# P3: celerite2 Matern-3/2 GP detrend on the out-of-transit data -> BLS -> LD depth refit

def _gp_trend_fit_predict_masked(t_fit, y_fit, t_pred,
                                 min_rho_days=C.P3_MIN_RHO_DAYS,
                                 max_rho_days=C.P3_MAX_RHO_DAYS):
    from celerite2 import GaussianProcess, terms
    from scipy.optimize import minimize

    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    t_pred = np.asarray(t_pred, dtype=float)

    mu = float(np.nanmedian(y_fit))
    y0 = y_fit - mu
    yerr = robust_sigma(y0) * np.ones_like(y0)

    init_log_sigma = float(np.log(max(np.nanstd(y0), 1e-6)))
    init_log_rho = float(np.log(max(min_rho_days, 1.0)))

    def make_kernel(log_sigma, log_rho):
        # celerite2 changed this signature across versions; support both.
        try:
            return terms.Matern32Term(log_sigma=float(log_sigma), log_rho=float(log_rho))
        except TypeError:
            return terms.Matern32Term(sigma=float(np.exp(log_sigma)), rho=float(np.exp(log_rho)))

    def nll(params):
        kernel = make_kernel(params[0], params[1])
        gp = GaussianProcess(kernel, mean=0.0)
        gp.compute(t_fit, yerr=yerr)
        return -gp.log_likelihood(y0)

    x0 = np.array([init_log_sigma, init_log_rho], dtype=float)
    bounds = [(np.log(1e-7), np.log(1e-1)), (np.log(min_rho_days), np.log(max_rho_days))]
    try:
        sol = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
        x_opt = sol.x if sol.success else x0
    except Exception:
        x_opt = x0

    gp = GaussianProcess(make_kernel(x_opt[0], x_opt[1]), mean=0.0)
    gp.compute(t_fit, yerr=yerr)
    try:
        trend = gp.predict(y0, t_pred, return_var=False)
    except TypeError:
        trend = gp.predict(y0, t_pred, return_cov=False)
    return trend + mu


def run_P3(t, y, duration_days: float, b_ld: float):
    t = np.asarray(t, dtype=float)
    y_norm = _normalize(y)

    y_det0 = detrend_rolling_median_subtract(t, y_norm, window_days=1.5)
    P0, t00, _, _ = run_bls_core(t, y_det0)

    in_tr = transit_mask(t, P0, t00, duration_days, pad_factor=3.0)
    keep = ~in_tr
    if np.sum(keep) < max(80, int(0.6 * len(t))):
        keep = np.ones_like(t, dtype=bool)

    trend = _gp_trend_fit_predict_masked(t_fit=t[keep], y_fit=y_norm[keep], t_pred=t)
    y_res = y_norm - trend
    y_res = y_res - np.nanmedian(y_res)

    P, t0, depth_bls, score = run_bls_core(t, y_res)
    depth = _ld_depth_or_bls(t, y_norm, P, t0, duration_days, b_ld, depth_bls)
    return P, t0, float(depth), score


# P4: BLS-seeded joint ridge fit of trend + LD transit template, BIC selection

def _bic_score(rss, n, k):
    rss = float(rss)
    if not np.isfinite(rss) or rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + k * np.log(max(n, 2))


def run_P4(t, y, duration_days: float, b_ld: float):
    from astropy.timeseries import BoxLeastSquares

    t = np.asarray(t, dtype=float)
    y_norm = _normalize(y)

    y_det0 = detrend_rolling_median_subtract(t, y_norm, window_days=1.5)
    seeds = bls_seed_periods_and_t0(t, y_det0, topk=C.P4_TOPK_PERIODS)

    X_trend, intercept_like = build_design_matrix(t)
    best = None  # (bic, P, t0, depth, score)
    n = len(t)
    dur = float(duration_days)

    # For each seed period, scan a small t0 grid and jointly fit trend + transit.
    for period, t0_seed in seeds:
        halfspan = C.P4_T0_GRID_HALFSPAN * dur
        for t0_try in t0_seed + np.linspace(-halfspan, halfspan, C.P4_T0_GRID_STEPS):
            shape = ld_shape_unit(t, period=period, t0=t0_try, duration_days=dur, b=b_ld)
            X = np.column_stack([X_trend, -shape])
            il = np.concatenate([intercept_like, np.array([True])])

            beta = ridge_solve(X, y_norm, alpha=C.P4_RIDGE_ALPHA, intercept_like_cols=il)
            res = y_norm - X @ beta
            bic = _bic_score(float(np.sum(res * res)), n=n, k=X.shape[1])

            depth = float(beta[-1])
            if not np.isfinite(depth) or depth <= 0:
                depth = float("nan")
            if best is None or bic < best[0]:
                best = (bic, float(period), float(t0_try), float(depth), float(-bic))

    # Fallback: if nothing was selected, just report the best plain-BLS peak.
    if best is None:
        bls = BoxLeastSquares(t, y_det0)
        power = bls.autopower(DURATIONS_DAYS, minimum_period=C.PERIOD_MIN,
                              maximum_period=C.PERIOD_MAX,
                              frequency_factor=C.BLS_FREQUENCY_FACTOR)
        k = int(np.argmax(power.power))
        return (float(power.period[k]), float(power.transit_time[k]),
                abs(float(power.depth[k])), float(power.power[k]))

    _, P_best, t0_best, depth_best, score_best = best
    return P_best, t0_best, depth_best, score_best


PIPELINES = {
    "P0": run_P0,
    "P1": run_P1,
    "P2": run_P2,
    "P3": run_P3,
    "P4": run_P4,
}
