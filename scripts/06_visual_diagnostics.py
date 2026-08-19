"""Visual diagnostics for a single trial: run all five pipelines on one chosen
trial and save time-series, phase-fold, zoom, and side-by-side comparison plots,
plus a CSV of what each pipeline recovered.

Pipelines are imported straight from the package registry (the old
importlib-from-file machinery is gone).

    python scripts/06_visual_diagnostics.py --trial-id 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.io_utils import load_trial_index, load_trial
from src.common.scoring import evaluate_recovery
from src.common.transit_template import ld_shape_unit
from src.pipelines import PIPELINES

PHASE_BINS = 120
ZOOM_HALF_WIDTH_IN_DURATIONS = 2.0
ZOOM_POINTS_MIN = 80


def phase_fold_hours(t, y, period, t0):
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    ph_h = phase * 24.0
    o = np.argsort(ph_h)
    return ph_h[o], y[o]


def bin_means(x, y, nbins):
    """Mean of y within nbins equal-width x bins; bins with fewer than 5 points
    are dropped."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 10:
        return np.array([]), np.array([])
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_mean = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.sum(m) >= 5:
            y_mean[i] = float(np.nanmean(y[m]))
    ok = np.isfinite(y_mean)
    return centers[ok], y_mean[ok]


def status_str(f) -> str:
    tag = "PASS" if int(f.get("pass", 0)) == 1 else "FAIL"
    return f"{tag} (p,t,d)=({f.get('p_ok', 0)},{f.get('t_ok', 0)},{f.get('d_ok', 0)})"


def compute_true_transit_centers(tmin, tmax, period, t0):
    """Times of the true transit centres that fall within [tmin, tmax]."""
    k0 = int(np.floor((tmin - t0) / period)) - 2
    k1 = int(np.ceil((tmax - t0) / period)) + 2
    centers = t0 + period * np.arange(k0, k1 + 1)
    return centers[(centers >= tmin) & (centers <= tmax)]


def plot_timeseries(trial_id, t, y, meta, f, out_png):
    y_norm = y / np.nanmedian(y)
    P, t0 = float(meta["period_true"]), float(meta["t0_true"])
    half = 0.5 * float(meta["duration_true_days"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.scatter(t, y_norm, s=6, alpha=0.65)
    for c in compute_true_transit_centers(float(t.min()), float(t.max()), P, t0):
        ax.axvspan(c - half, c + half, alpha=0.06, color="C1")
    ax.set_title(
        f"Trial {trial_id} {meta.get('star_id', '')} "
        f"{meta['noise_level']}/{meta['gap_level']}\n"
        f"True: P={P:.4f} d  t0={t0:.4f}  dur={meta['duration_true_days'] * 24:.2f} h  "
        f"depth={meta['depth_true']:.4g}\n"
        f"{f['name']} {status_str(f)}  P={f['period_found']:.4f}  "
        f"t0={f['t0_found']:.4f}  depth={f['depth_found']:.4g}")
    ax.set_xlabel("time (days, trial frame)")
    ax.set_ylabel("normalized flux")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_phase_fold(trial_id, t, y, meta, f, out_png):
    y_norm = y / np.nanmedian(y)
    P, t0 = float(f["period_found"]), float(f["t0_found"])
    half_h = 0.5 * float(meta["duration_true_days"]) * 24.0
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if not np.isfinite(P) or not np.isfinite(t0) or P <= 0:
        ax.text(0.5, 0.5, "Missing period/t0", ha="center", va="center")
    else:
        ph_h, ys = phase_fold_hours(t, y_norm, P, t0)
        xb, yb = bin_means(ph_h, ys, nbins=PHASE_BINS)
        ax.scatter(ph_h, ys, s=6, alpha=0.25, label="folded points")
        if len(xb):
            ax.plot(xb, yb, lw=2, label="binned mean")
        ax.axvspan(-half_h, half_h, alpha=0.08, label="true duration window")
        ax.axvline(0.0, lw=1, alpha=0.5)
        ax.legend(loc="best")
    ax.set_title(f"Trial {trial_id} {f['name']} {status_str(f)} phase fold")
    ax.set_xlabel("phase (hours)")
    ax.set_ylabel("normalized flux")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_zoom(trial_id, t, y, meta, f, out_png, b_ld):
    """Clean phase-folded transit figure: faint folded points, one smooth
    limb-darkened model at the fitted depth, and a reference line at flux = 1.

    No binned curve here: the points carry the data and the smooth curve carries
    the recovered shape. The recovery template has a hard edge at the contact
    points, so for a continuous-looking line we taper the drawn curve with a
    narrow cosine ramp at each edge. That only rounds the plotted corner; the
    depth and duration are unchanged.
    """
    y_norm = y / np.nanmedian(y)
    P, t0 = float(f["period_found"]), float(f["t0_found"])
    dur = float(meta["duration_true_days"])
    if not np.isfinite(P) or not np.isfinite(t0) or P <= 0:
        return

    phase_days = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    half_width = ZOOM_HALF_WIDTH_IN_DURATIONS * dur
    m = np.abs(phase_days) <= half_width
    if np.sum(m) < ZOOM_POINTS_MIN:
        return
    ph, fl = phase_days[m], y_norm[m]

    depth = float(f["depth_found"]) if np.isfinite(f["depth_found"]) else 0.0
    half = 0.5 * dur
    xx = np.linspace(-half_width, half_width, 4000)
    shape = ld_shape_unit(xx + t0, period=P, t0=t0, duration_days=dur, b=b_ld)
    # Cosine taper over the outer 12% of the half-duration near each contact point.
    ramp = 0.12 * half
    ax_phase = np.abs(xx)
    edge_zone = (ax_phase > (half - ramp)) & (ax_phase <= half)
    w = 0.5 * (1.0 + np.cos(np.pi * (ax_phase[edge_zone] - (half - ramp)) / ramp))
    shape[edge_zone] *= w
    model = 1.0 - depth * shape

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.scatter(ph / P, fl, s=6, alpha=0.18, color="#7aa6c2", label="folded data")
    ax.plot(xx / P, model, lw=2.2, color="#c2541b", label="best-fit transit model")
    ax.axhline(1.0, lw=0.8, color="0.6", alpha=0.7)
    ax.set_title(f"Trial {trial_id} {f['name']} {status_str(f)}\nP={P:.4f} d  t0={t0:.4f}")
    ax.set_xlabel("orbital phase (cycles)")
    ax.set_ylabel("normalized flux")
    ax.legend(loc="lower right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def plot_phase_compare_all(trial_id, t, y, meta, found_all, out_png):
    y_norm = y / np.nanmedian(y)
    half_h = 0.5 * float(meta["duration_true_days"]) * 24.0
    names = list(PIPELINES.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.7 * len(names), 4.2), sharey=True)
    for ax, name in zip(axes, names):
        f = found_all[name]
        P, t0 = float(f["period_found"]), float(f["t0_found"])
        if not np.isfinite(P) or not np.isfinite(t0) or P <= 0:
            ax.text(0.5, 0.5, f"{name}\nmissing", ha="center", va="center")
            ax.set_title(f"{name}\n{status_str(f)}")
            ax.set_xlabel("phase (h)")
            continue
        ph_h, ys = phase_fold_hours(t, y_norm, P, t0)
        xb, yb = bin_means(ph_h, ys, nbins=PHASE_BINS)
        ax.scatter(ph_h, ys, s=6, alpha=0.18)
        if len(xb):
            ax.plot(xb, yb, lw=2)
        ax.axvspan(-half_h, half_h, alpha=0.08)
        ax.axvline(0.0, lw=1, alpha=0.5)
        ax.set_title(f"{name}\n{status_str(f)}\nP={P:.3f}")
        ax.set_xlabel("phase (h)")
    axes[0].set_ylabel("normalized flux")
    fig.suptitle(f"Trial {trial_id} phase-fold comparison\n"
                 f"True: P={meta['period_true']:.4f} d  t0={meta['t0_true']:.4f}  "
                 f"dur={meta['duration_true_days'] * 24:.2f} h", y=1.05)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-id", type=int, required=True)
    args = ap.parse_args()
    trial_id = int(args.trial_id)

    idx = load_trial_index()
    row = idx[idx["trial_id"].astype(int) == trial_id]
    if len(row) == 0:
        raise ValueError(f"trial_id={trial_id} not found in {C.TRIAL_INDEX_CSV}")
    row = row.iloc[0]

    t, y = load_trial(str(row["file"]))
    meta = {
        "star_id": str(row.get("star_id", "")),
        "noise_level": str(row["noise_level"]), "gap_level": str(row["gap_level"]),
        "period_true": float(row["period_true"]), "t0_true": float(row["t0_true"]),
        "duration_true_days": float(row["duration_true_days"]),
        "depth_true": float(row["depth_true"]),
    }
    dur = meta["duration_true_days"]
    b_ld = float(row.get("impact_parameter", C.LD_B_DEFAULT))
    print("Trial:", trial_id, meta)

    out_dir = C.FIGURES_DIR / "visual_diagnostics"
    found_all = {}
    for name, run_fn in PIPELINES.items():
        try:
            P, t0, depth, score = run_fn(t, y, duration_days=dur, b_ld=b_ld)
            passed, p_ok, t_ok, d_ok = evaluate_recovery(
                meta["period_true"], meta["t0_true"], meta["depth_true"],
                float(P), float(t0), float(depth))
            found_all[name] = {"name": name, "period_found": float(P),
                               "t0_found": float(t0), "depth_found": float(depth),
                               "score": float(score), "pass": passed,
                               "p_ok": p_ok, "t_ok": t_ok, "d_ok": d_ok, "error": ""}
        except Exception as e:
            found_all[name] = {"name": name, "period_found": np.nan, "t0_found": np.nan,
                               "depth_found": np.nan, "score": np.nan, "pass": 0,
                               "p_ok": 0, "t_ok": 0, "d_ok": 0, "error": repr(e)}
        f = found_all[name]
        print(name, "->", status_str(f),
              f"P={f['period_found']:.5g} t0={f['t0_found']:.5g} depth={f['depth_found']:.4g}",
              f"ERR: {f['error']}" if f["error"] else "")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"trial_id": trial_id, **meta}
    for k in found_all:
        for fld in ("pass", "p_ok", "t_ok", "d_ok",
                    "period_found", "t0_found", "depth_found", "error"):
            summary[f"{k}_{fld}"] = found_all[k][fld]
    pd.DataFrame([summary]).to_csv(out_dir / f"trial_{trial_id:06d}_outputs.csv", index=False)

    for name in PIPELINES:
        f = found_all[name]
        plot_timeseries(trial_id, t, y, meta, f, out_dir / f"trial_{trial_id:06d}_{name}_timeseries.png")
        plot_phase_fold(trial_id, t, y, meta, f, out_dir / f"trial_{trial_id:06d}_{name}_phase.png")
        plot_zoom(trial_id, t, y, meta, f, out_dir / f"trial_{trial_id:06d}_{name}_zoom.png", b_ld)
    plot_phase_compare_all(trial_id, t, y, meta, found_all,
                           out_dir / f"trial_{trial_id:06d}_PHASE_COMPARE_ALL.png")
    print("Saved outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
