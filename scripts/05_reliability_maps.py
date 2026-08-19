"""Reliability heat maps (per-pipeline rates) and paired delta heat maps, with
Wilson CIs on the rates, paired CIs on the deltas, and a per-bin exact McNemar p.

This replaces reliability_map1.py ... reliability_map4.py (which were the same
script with two filenames edited) with one parameterized script:

    python scripts/05_reliability_maps.py --baseline P0 --challenger P4
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
from src.common.io_utils import load_trial_index, load_results
from src.common.scoring import wilson_ci, paired_delta_ci, mcnemar_exact_p, discordant_counts


def plot_rate_heatmap(p_mat, lo_mat, hi_mat, title, out_png: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    im = ax.imshow(p_mat, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(C.GAP_LEVELS)), C.GAP_LEVELS)
    ax.set_yticks(range(len(C.NOISE_LEVELS)), C.NOISE_LEVELS)
    for i in range(p_mat.shape[0]):
        for j in range(p_mat.shape[1]):
            if np.isfinite(p_mat[i, j]):
                ax.text(j, i, f"{p_mat[i, j]:.3f}\n[{lo_mat[i, j]:.3f},{hi_mat[i, j]:.3f}]",
                        ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("gap_level")
    ax.set_ylabel("noise_level")
    fig.colorbar(im, ax=ax, label="rate")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_delta_heatmap(d_mat, lo_mat, hi_mat, title, out_png: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    vmax = max(float(np.nanmax(np.abs(d_mat))), 0.05)
    im = ax.imshow(d_mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(len(C.GAP_LEVELS)), C.GAP_LEVELS)
    ax.set_yticks(range(len(C.NOISE_LEVELS)), C.NOISE_LEVELS)
    for i in range(d_mat.shape[0]):
        for j in range(d_mat.shape[1]):
            if np.isfinite(d_mat[i, j]):
                ax.text(j, i, f"{d_mat[i, j]:+.3f}\n[{lo_mat[i, j]:+.3f},{hi_mat[i, j]:+.3f}]",
                        ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("gap_level")
    ax.set_ylabel("noise_level")
    fig.colorbar(im, ax=ax, label="delta (paired)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="P0")
    ap.add_argument("--challenger", default="P4")
    args = ap.parse_args()
    A, B = args.baseline, args.challenger

    idx = load_trial_index()[["trial_id", "noise_level", "gap_level"]]
    merged = idx.copy()
    for label in (A, B):
        res = load_results(label)[["trial_id"] + C.STATS_METRICS].copy()
        res = res.rename(columns={m: f"{m}_{label}" for m in C.STATS_METRICS})
        merged = merged.merge(res, on="trial_id", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No matched trials between index and results.")
    print(f"{A} vs {B}: {len(merged)} matched trials")

    out_dir = C.FIGURES_DIR / "reliability_maps"
    rows_out = []
    shape = (len(C.NOISE_LEVELS), len(C.GAP_LEVELS))

    # Per-pipeline rate maps.
    for label in (A, B):
        for metric in C.STATS_METRICS:
            p_mat = np.full(shape, np.nan)
            lo_mat = np.full(shape, np.nan)
            hi_mat = np.full(shape, np.nan)
            for i, nl in enumerate(C.NOISE_LEVELS):
                for j, gl in enumerate(C.GAP_LEVELS):
                    sub = merged[(merged["noise_level"] == nl) & (merged["gap_level"] == gl)]
                    n = len(sub)
                    if n == 0:
                        continue
                    k = int(sub[f"{metric}_{label}"].sum())
                    lo, hi = wilson_ci(k, n)
                    p_mat[i, j], lo_mat[i, j], hi_mat[i, j] = k / n, lo, hi
                    rows_out.append({"kind": "pipeline_rate", "pipeline": label,
                                     "metric": metric, "noise_level": nl, "gap_level": gl,
                                     "k_success": k, "n_trials": n, "rate": k / n,
                                     "ci_lo": lo, "ci_hi": hi})
            plot_rate_heatmap(p_mat, lo_mat, hi_mat,
                              f"{label} {metric} rate (Wilson {int(100 * 0.9)}% CI)",
                              out_dir / f"{label}_{metric}_rate.png")

    # Paired delta maps + per-bin McNemar.
    for metric in C.STATS_METRICS:
        d_mat = np.full(shape, np.nan)
        lo_mat = np.full(shape, np.nan)
        hi_mat = np.full(shape, np.nan)
        for i, nl in enumerate(C.NOISE_LEVELS):
            for j, gl in enumerate(C.GAP_LEVELS):
                sub = merged[(merged["noise_level"] == nl) & (merged["gap_level"] == gl)]
                if len(sub) == 0:
                    continue
                a = sub[f"{metric}_{A}"].to_numpy(dtype=int)
                b = sub[f"{metric}_{B}"].to_numpy(dtype=int)
                mean_d, lo_d, hi_d, n = paired_delta_ci(a, b)
                a_only, b_only = discordant_counts(a, b)
                d_mat[i, j], lo_mat[i, j], hi_mat[i, j] = mean_d, lo_d, hi_d
                rows_out.append({"kind": "paired_delta", "pipeline": f"{B}_minus_{A}",
                                 "metric": metric, "noise_level": nl, "gap_level": gl,
                                 "n_trials": int(n), "delta_mean": mean_d,
                                 "ci_lo": lo_d, "ci_hi": hi_d,
                                 "A_only": a_only, "B_only": b_only,
                                 "mcnemar_exact_p": mcnemar_exact_p(a_only, b_only)})
        plot_delta_heatmap(d_mat, lo_mat, hi_mat,
                           f"Delta ({B} - {A}) for {metric}",
                           out_dir / f"DELTA_{B}_minus_{A}_{metric}.png")

    table_out = C.TABLES_DIR / f"reliability_map_values_{A}_vs_{B}.csv"
    pd.DataFrame(rows_out).to_csv(table_out, index=False)
    print("Saved table:", table_out)
    print("Saved figures in:", out_dir)


if __name__ == "__main__":
    main()
