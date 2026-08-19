"""Figure: the injection-and-corruption cascade for one real light curve.

Explanatory (methods) figure, not a results figure. It shows what a single
"trial" is by drawing the same real light curve after each stage of trial
construction, so the reader can see concretely what Sections 2.3-2.4 do:

    (a) real PDCSAP flux, normalized
    (b) + limb-darkened transit injected
    (c) + white (uncorrelated) noise
    (d) + correlated (red / OU) noise
    (e) + data gaps removed

Every stage uses the same functions the real generator uses
(src.common.injection plus the OU and gap helpers in 02_generate_trials), so the
figure is faithful to the actual pipeline rather than a redrawn cartoon. A fixed
seed and a fixed star keep it reproducible.

    python scripts/fig_injection_cascade.py
    python scripts/fig_injection_cascade.py --tic 130181866 --noise high --gap severe
    python scripts/fig_injection_cascade.py --window 6.0 --period 3.5 --depth 0.0025

Output:
    results/figures/explanatory/injection_cascade.png
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

OUT = C.FIGURES_DIR / "explanatory" / "injection_cascade.png"


def _load_generator_helpers():
    """Import load_star_csv, ou_red_noise, apply_gap_blocks from
    02_generate_trials.py. That filename starts with a digit, so it can't be a
    normal import; load it by path instead."""
    gen_path = Path(__file__).resolve().parent / "02_generate_trials.py"
    spec = importlib.util.spec_from_file_location("_gen", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_star_csv, mod.ou_red_noise, mod.apply_gap_blocks


def pick_star() -> Path:
    """Use the first star in the manifest unless a specific TIC is given."""
    mdf = pd.read_csv(C.MANIFEST_CSV)
    row = mdf.iloc[0]
    raw = Path(str(row["file"]))
    return raw if raw.is_absolute() else (C.ROOT / raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, default=None,
                    help="TIC id to use; default is the first manifest entry.")
    ap.add_argument("--noise", choices=["low", "high"], default="high")
    ap.add_argument("--gap", choices=["minimal", "severe"], default="severe")
    ap.add_argument("--period", type=float, default=3.2, help="injected period (d)")
    ap.add_argument("--depth", type=float, default=0.0025, help="injected mid-depth")
    ap.add_argument("--window", type=float, default=8.0,
                    help="days of data to display (a short window reads better than a full sector)")
    ap.add_argument("--seed", type=int, default=7)
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

    # Restrict to a readable window near the start that actually contains data.
    t0 = float(t_full.min())
    m = (t_full >= t0) & (t_full <= t0 + args.window)
    t, y0 = t_full[m], y_full[m]
    if len(t) < 200:
        t, y0 = t_full, y_full   # window too sparse; fall back to the full curve

    rng = np.random.default_rng(args.seed)

    # (b) inject a limb-darkened transit using the real occultation table.
    ld_table = build_ld_table(C.DEPTH_MIN, C.DEPTH_MAX)
    mid = t0 + 0.5 * (t.max() - t.min())   # a transit mid-way through the window
    delta = transit_delta_ld(t=t, period=args.period, t0=mid,
                             duration_days=C.DURATION_DAYS, depth_mid=args.depth,
                             b=C.IMPACT_PARAMETER, ld_table=ld_table)
    y_inj = y0 * (1.0 - delta)

    # (c) white noise.
    if args.noise == "high":
        white, red = C.WHITE_SIGMA_HIGH, C.RED_SIGMA_HIGH
    else:
        white, red = C.WHITE_SIGMA_LOW, C.RED_SIGMA_LOW
    y_white = y_inj + rng.normal(0.0, white, size=len(t))

    # (d) correlated (red) noise on top.
    y_red = y_white + ou_red_noise(t, sigma=red, rho_days=C.RED_RHO_DAYS, rng=rng)

    # (e) gaps.
    if args.gap == "severe":
        n_blocks, blen = C.GAP_BLOCKS_SEVERE, C.GAP_LEN_DAYS_SEVERE
    else:
        n_blocks, blen = C.GAP_BLOCKS_MINIMAL, C.GAP_LEN_DAYS_MINIMAL
    keep = apply_gap_blocks(t, n_blocks=n_blocks, block_len_days=blen, rng=rng)

    stages = [
        ("(a) real light curve (normalized)", t, y0),
        ("(b) + injected transit", t, y_inj),
        ("(c) + white noise", t, y_white),
        ("(d) + correlated (red) noise", t, y_red),
        (f"(e) + data gaps ({args.gap})", t[keep], y_red[keep]),
    ]

    # Locate each injected transit event and shade it, so the bands line up with
    # the actual dips instead of spanning everything between first and last transit.
    tref = t.min()
    intransit = delta > (0.02 * delta.max() if delta.max() > 0 else np.inf)
    bands = []
    if intransit.any():
        idx = np.where(intransit)[0]
        splits = np.where(np.diff(idx) > 1)[0] + 1   # one run per transit event
        for run in np.split(idx, splits):
            bands.append((t[run].min() - tref, t[run].max() - tref))

    fig, axes = plt.subplots(len(stages), 1, figsize=(7.5, 9.0), sharex=True)
    for ax, (label, tt, yy) in zip(axes, stages):
        for lo, hi in bands:
            ax.axvspan(lo, hi, color="#e8a13c", alpha=0.18, zorder=0)
        ax.scatter(tt - tref, yy, s=3, color="#3a6ea5", alpha=0.55, linewidths=0)
        ax.set_ylabel("flux", fontsize=8)
        ax.text(0.012, 0.90, label, transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left", weight="bold")
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time (days within window)", fontsize=9)
    fig.suptitle(f"Building one trial   (TIC {star_path.stem.split('_')[1]}, "
                 f"noise={args.noise}, gap={args.gap})", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
