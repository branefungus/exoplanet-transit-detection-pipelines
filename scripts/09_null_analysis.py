"""False-positive analysis from the null trials (completeness vs false-positive curve).

Every trial, real and null, is scored by a common, pipeline-agnostic detection
statistic: the recovered transit's depth over the out-of-transit scatter at the
recovered ephemeris (the `snr` column written by 03_run_pipeline.py). Using one
statistic for all five pipelines makes their curves directly comparable, unlike
each pipeline's native `score` (BLS power for P0-P3, -BIC for P4), which live on
different scales.

For each pipeline we sweep an SNR threshold and trace:

  completeness(thr)  = fraction of REAL trials that pass recovery AND snr >= thr
  false_pos(thr)     = fraction of NULL trials with snr >= thr

Plotting completeness against false_pos gives a detection-tradeoff (ROC-style)
curve per pipeline. We also report completeness at a few fixed false-positive
rates (1%, 5%, 10%) for a compact table.

Requires (run these first):
    python scripts/02_generate_trials.py            # real trials + results
    python scripts/03_run_pipeline.py --pipeline all
    python scripts/02_generate_trials.py --null     # null trials
    python scripts/03_run_pipeline.py --pipeline all --null

Then:
    python scripts/09_null_analysis.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.io_utils import load_results
from src.pipelines import PIPELINES

OUT_FIG = C.FIGURES_DIR / "null_analysis"
TARGET_FPRS = [0.01, 0.05, 0.10]


def load_scores(pipeline: str):
    real = load_results(pipeline)                      # has pass + snr
    nul = load_results(pipeline, null=True)            # has snr
    real_snr = real["snr"].to_numpy(dtype=float)
    real_pass = real["pass"].to_numpy(dtype=int)
    null_snr = nul["snr"].to_numpy(dtype=float)
    # Drop non-finite SNRs (failed or unusable ephemerides).
    rm = np.isfinite(real_snr)
    nm = np.isfinite(null_snr)
    return real_snr[rm], real_pass[rm], null_snr[nm]


def curve(real_snr, real_pass, null_snr, n_thr=200):
    """Sweep a common SNR threshold; return (threshold, completeness, fpr).
    completeness = fraction of real trials that pass AND clear the threshold;
    fpr = fraction of null trials that clear the threshold."""
    lo = float(min(real_snr.min(), null_snr.min()))
    hi = float(max(real_snr.max(), null_snr.max()))
    thrs = np.linspace(lo, hi, n_thr)
    n_real = len(real_pass)
    n_null = len(null_snr)
    comp = np.array([np.sum((real_snr >= t) & (real_pass == 1)) / n_real for t in thrs])
    fpr = np.array([np.sum(null_snr >= t) / n_null for t in thrs])
    return thrs, comp, fpr


def completeness_at_fpr(comp, fpr, target):
    """Highest completeness reachable at or below a target false-positive rate."""
    ok = fpr <= target
    return float(np.max(comp[ok])) if np.any(ok) else float("nan")


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    names = list(PIPELINES.keys())

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    table_rows = []
    for name in names:
        try:
            rs, rp, ns = load_scores(name)
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")
            continue
        thrs, comp, fpr = curve(rs, rp, ns)
        ax.plot(fpr, comp, lw=2, label=name)
        row = {"pipeline": name,
               "n_real": len(rp), "n_null": len(ns),
               "overall_pass_rate": float(np.mean(rp))}
        for tgt in TARGET_FPRS:
            row[f"completeness_at_fpr_{int(tgt*100)}pct"] = completeness_at_fpr(comp, fpr, tgt)
        table_rows.append(row)

    ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1, label="chance")
    ax.set_xlabel("false-positive rate (null trials)")
    ax.set_ylabel("completeness (passing real trials)")
    ax.set_title("Detection tradeoff: completeness vs false-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_png = OUT_FIG / "completeness_vs_fpr.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    tab = pd.DataFrame(table_rows)
    out_csv = C.TABLES_DIR / "null_analysis_summary.csv"
    tab.to_csv(out_csv, index=False)

    print("\nCompleteness at fixed false-positive rates:")
    with pd.option_context("display.width", 120):
        print(tab.to_string(index=False))
    print("\nSaved table:", out_csv)
    print("Saved figure:", out_png)


if __name__ == "__main__":
    main()
