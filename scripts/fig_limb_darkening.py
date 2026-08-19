"""Figure: what limb darkening does to the transit shape.

Explanatory (methods) figure motivating the "rounded floor" language. It shows
the limb-darkened transit light curve next to a plain rectangular ("box") dip of
the same depth and duration, so the reader can see the rounded floor and sloped
edges that the limb-darkened model produces and a box does not.

It uses the same occultation table the generator uses
(src.common.injection.transit_delta_ld), so the drawn transit is exactly the
shape injected into the trials, not an approximation.

    python scripts/fig_limb_darkening.py
    python scripts/fig_limb_darkening.py --depth 0.0025 --period 3.2

Output:
    results/figures/explanatory/limb_darkening.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.injection import build_ld_table, transit_delta_ld

OUT = C.FIGURES_DIR / "explanatory" / "limb_darkening.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.0025)
    ap.add_argument("--period", type=float, default=3.2)
    args = ap.parse_args()

    fig, axR = plt.subplots(1, 1, figsize=(7.0, 4.4))

    depth = args.depth
    dur = C.DURATION_DAYS

    # A phase window a few durations wide, centred on mid-transit. We evaluate the
    # occultation table far from t=0 (offset by 1000 d) to avoid any edge effects
    # at the origin, then plot against the local time axis.
    half_win = 2.2 * dur
    t = np.linspace(-half_win, half_win, 2000)
    t0 = 0.0

    ld_table = build_ld_table(C.DEPTH_MIN, C.DEPTH_MAX)
    delta = transit_delta_ld(t=t + 1000.0, period=args.period, t0=1000.0 + t0,
                             duration_days=dur, depth_mid=depth,
                             b=C.IMPACT_PARAMETER, ld_table=ld_table)
    ld_curve = 1.0 - delta

    # Matched box: same depth and same total (first-to-fourth contact) duration.
    box = np.ones_like(t)
    box[np.abs(t) <= dur / 2.0] = 1.0 - depth

    th = t * 24.0  # hours, for readability
    axR.plot(th, box, color="#888888", lw=1.8, ls="--", label="box (rectangular) dip")
    axR.plot(th, ld_curve, color="#2f6fb0", lw=2.4, label="limb-darkened transit")
    axR.axhline(1.0, color="0.7", lw=0.8)
    axR.set_xlabel("time from mid-transit (hours)")
    axR.set_ylabel("normalized flux")
    axR.set_title("Transit shape: rounded floor vs. flat box", fontsize=11)
    axR.legend(loc="lower right", fontsize=9)
    axR.grid(alpha=0.25)
    axR.annotate("rounded floor and\nsloped edges",
                 xy=(0, ld_curve[np.argmin(np.abs(th))]),
                 xytext=(3.0, 1 - 0.55 * depth),
                 fontsize=8.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color="0.4"))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
