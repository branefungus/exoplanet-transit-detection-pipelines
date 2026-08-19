"""Figure: the 2x2 grid of observing conditions.

Explanatory (methods) figure. It shows the four condition cells the experiment
crosses (low/high noise by minimal/severe gaps), each drawn as a real
light-curve thumbnail with the same injected transit, so the reader can see what
each corner of the 2x2 grid actually looks like before meeting the delta maps
(which share the same 2x2 layout).

Every cell uses the same functions the real generator uses (src.common.injection
plus the OU and gap helpers in 02_generate_trials), so the thumbnails are
faithful to the actual trials. One fixed star, one fixed transit, and one fixed
seed per cell keep it reproducible and make the cells directly comparable (only
the noise and gap settings change between them).

    python scripts/fig_condition_grid.py
    python scripts/fig_condition_grid.py --tic 283722336 --window 8 --depth 0.0025

Output:
    results/figures/explanatory/condition_grid.png
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.injection import build_ld_table, transit_delta_ld

OUT = C.FIGURES_DIR / "explanatory" / "condition_grid.png"


def _load_generator_helpers():
    gen_path = Path(__file__).resolve().parent / "02_generate_trials.py"
    spec = importlib.util.spec_from_file_location("_gen", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_star_csv, mod.ou_red_noise, mod.apply_gap_blocks


def pick_star() -> Path:
    mdf = pd.read_csv(C.MANIFEST_CSV)
    row = mdf.iloc[0]
    raw = Path(str(row["file"]))
    return raw if raw.is_absolute() else (C.ROOT / raw)


def build_cell(t, y0, delta, noise, gap, ou_red_noise, apply_gap_blocks, seed):
    """Return (time, flux) for one condition cell: inject, then add noise and gaps."""
    rng = np.random.default_rng(seed)
    y = y0 * (1.0 - delta)
    if noise == "high":
        white, red = C.WHITE_SIGMA_HIGH, C.RED_SIGMA_HIGH
    else:
        white, red = C.WHITE_SIGMA_LOW, C.RED_SIGMA_LOW
    y = y + rng.normal(0.0, white, size=len(t))
    y = y + ou_red_noise(t, sigma=red, rho_days=C.RED_RHO_DAYS, rng=rng)
    if gap == "severe":
        n_blocks, blen = C.GAP_BLOCKS_SEVERE, C.GAP_LEN_DAYS_SEVERE
    else:
        n_blocks, blen = C.GAP_BLOCKS_MINIMAL, C.GAP_LEN_DAYS_MINIMAL
    keep = apply_gap_blocks(t, n_blocks=n_blocks, block_len_days=blen, rng=rng)
    return t[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, default=None)
    ap.add_argument("--period", type=float, default=3.2)
    ap.add_argument("--depth", type=float, default=0.0025)
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    load_star_csv, ou_red_noise, apply_gap_blocks = _load_generator_helpers()

    if args.tic is not None:
        matches = list(C.STARS_DIR.glob(f"tic_{args.tic}_*.csv"))
        if not matches:
            raise FileNotFoundError(f"No downloaded file for TIC {args.tic} in {C.STARS_DIR}")
        star_path = matches[0]
    else:
        star_path = pick_star()

    t_full, y_full, _ = load_star_csv(star_path)
    t0 = float(t_full.min())
    m = (t_full >= t0) & (t_full <= t0 + args.window)
    t, y0 = t_full[m], y_full[m]
    if len(t) < 200:
        t, y0 = t_full, y_full

    # One fixed injected transit, shared across all four cells.
    ld_table = build_ld_table(C.DEPTH_MIN, C.DEPTH_MAX)
    mid = t0 + 0.5 * (t.max() - t.min())
    delta = transit_delta_ld(t=t, period=args.period, t0=mid,
                             duration_days=C.DURATION_DAYS, depth_mid=args.depth,
                             b=C.IMPACT_PARAMETER, ld_table=ld_table)
    tref = t.min()

    # Rows = noise (low on top, high on bottom); cols = gap (minimal, severe).
    noises = ["low", "high"]
    gaps = ["minimal", "severe"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, sharey=True)
    for i, noise in enumerate(noises):
        for j, gap in enumerate(gaps):
            ax = axes[i][j]
            seed = args.seed + 100 * i + 10 * j   # per-cell seed: reproducible + comparable
            tt, yy = build_cell(t, y0, delta, noise, gap,
                                ou_red_noise, apply_gap_blocks, seed)
            ax.scatter(tt - tref, yy, s=2.5, color="#3a6ea5", alpha=0.5, linewidths=0)
            ax.tick_params(labelsize=8)
            ax.text(0.03, 0.93, f"{noise} noise / {gap} gaps",
                    transform=ax.transAxes, fontsize=9.5, va="top", weight="bold")

    for i, noise in enumerate(noises):
        axes[i][0].set_ylabel("flux", fontsize=9)
        axes[i][0].annotate(f"{noise}\nnoise", xy=(-0.30, 0.5),
                            xycoords="axes fraction", fontsize=10, weight="bold",
                            ha="center", va="center", rotation=90)
    axes[0][0].set_title("minimal gaps", fontsize=10)
    axes[0][1].set_title("severe gaps", fontsize=10)
    axes[1][0].set_xlabel("time (days within window)", fontsize=9)
    axes[1][1].set_xlabel("time (days within window)", fontsize=9)

    fig.suptitle(f"The four observing conditions   "
                 f"(TIC {star_path.stem.split('_')[1]}, same injected transit in each)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
