"""Per-star robustness check for the paired pipeline comparisons.

The 2400 trials share only 30 base light curves, so the trials aren't fully
independent. This script confirms that a pooled advantage isn't being driven by
a handful of atypical stars: for each star it computes the baseline vs
challenger pass rate and the paired delta, then reports how many of the 30 stars
the challenger wins, ties, or loses on, along with a sign test on the per-star
deltas.

    python scripts/08_per_star_robustness.py
    python scripts/08_per_star_robustness.py --baseline P0 --challengers P1 P2 P3 P4 --metric pass

Outputs:
  results/tables/per_star_<baseline>_vs_<challenger>_<metric>.csv   (one row per star)
  results/tables/per_star_robustness_report.txt                    (human-readable)
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.io_utils import load_trial_index, load_results
from src.common.scoring import mcnemar_exact_p, discordant_counts

REPORT_PATH = C.TABLES_DIR / "per_star_robustness_report.txt"


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test on per-star wins vs losses (ties dropped),
    which is a binomial(wins + losses, 0.5). We just reuse the McNemar exact
    engine, since it's the same computation."""
    return mcnemar_exact_p(wins, losses)


def per_star_table(baseline: str, challenger: str, metric: str) -> pd.DataFrame:
    idx = load_trial_index()
    if "star_id" not in idx.columns:
        raise ValueError("trial_index.csv has no star_id column; regenerate trials with the current generator.")
    idx = idx[["trial_id", "star_id"]]

    a = load_results(baseline)[["trial_id", metric]].rename(columns={metric: "A"})
    b = load_results(challenger)[["trial_id", metric]].rename(columns={metric: "B"})
    df = idx.merge(a, on="trial_id").merge(b, on="trial_id")

    rows = []
    for star, g in df.groupby("star_id"):
        av = g["A"].to_numpy(dtype=int)
        bv = g["B"].to_numpy(dtype=int)
        n = len(g)
        a_only, b_only = discordant_counts(av, bv)
        rows.append({
            "star_id": star, "n": n,
            "A_rate": av.mean(), "B_rate": bv.mean(),
            "delta(B-A)": bv.mean() - av.mean(),
            "A_only": a_only, "B_only": b_only,
        })
    return pd.DataFrame(rows).sort_values("star_id").reset_index(drop=True)


def summarize_stars(tab: pd.DataFrame, eps: float = 1e-9) -> dict:
    d = tab["delta(B-A)"].to_numpy()
    wins = int(np.sum(d > eps))
    losses = int(np.sum(d < -eps))
    ties = int(np.sum(np.abs(d) <= eps))
    return {
        "n_stars": len(tab), "challenger_wins": wins,
        "challenger_losses": losses, "ties": ties,
        "median_delta": float(np.median(d)), "mean_delta": float(np.mean(d)),
        "sign_test_p": sign_test_p(wins, losses),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="P0")
    ap.add_argument("--challengers", nargs="+", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--metric", default="pass", choices=C.STATS_METRICS)
    args = ap.parse_args()

    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Per-star robustness report",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Baseline: {args.baseline}   Metric: {args.metric}",
        "For each challenger: how many of the N stars does it beat the baseline on,",
        "and an exact two-sided sign test on the per-star deltas (ties dropped).",
        "",
    ]

    for ch in args.challengers:
        try:
            tab = per_star_table(args.baseline, ch, args.metric)
        except FileNotFoundError as e:
            lines.append(f"{args.baseline} vs {ch}: SKIPPED ({e})")
            print(f"Skipped {ch}: {e}")
            continue

        tab.to_csv(C.TABLES_DIR / f"per_star_{args.baseline}_vs_{ch}_{args.metric}.csv", index=False)
        s = summarize_stars(tab)
        lines += [
            "=" * 60,
            f"  {args.baseline} vs {ch}   (metric: {args.metric})",
            "=" * 60,
            f"  stars: {s['n_stars']}    {ch} wins on {s['challenger_wins']}, "
            f"loses on {s['challenger_losses']}, ties {s['ties']}",
            f"  median per-star delta: {s['median_delta']:+.4f}    "
            f"mean: {s['mean_delta']:+.4f}",
            f"  sign test (exact, two-sided): p = {s['sign_test_p']:.3e}",
            "",
        ]
        print(f"{args.baseline} vs {ch} ({args.metric}): "
              f"{ch} wins {s['challenger_wins']}/{s['n_stars']}, sign-test p={s['sign_test_p']:.2e}")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\nReport written to:", REPORT_PATH)


if __name__ == "__main__":
    main()
