"""Holm-Bonferroni correction across the family of secondary paired comparisons.

The paper's primary, pre-specified comparison is P4 vs P0 on the `pass` metric,
and it's reported uncorrected. Every other pipeline/metric comparison is
secondary, and together they form a family of (n_challengers x n_metrics) tests.
Reporting raw p-values for a family that size inflates the family-wise error
rate: with 16 independent tests, the chance of at least one spurious result at
alpha=0.05 is about 56%.

Holm-Bonferroni is a step-down procedure that controls the family-wise error
rate while being uniformly more powerful than plain Bonferroni. Sort the m
p-values ascending, compare the k-th smallest against alpha/(m-k+1), and stop at
the first failure, after which nothing is significant. Equivalently, report
adjusted p-values

    p_adj(k) = max over j <= k of [ (m-j+1) * p(j) ],   capped at 1

and compare those directly to alpha. The running maximum enforces monotonicity,
so a larger raw p can never produce a smaller adjusted p.

Reads the per-comparison summaries written by 04_paired_stats.py.

    python scripts/10_holm_correction.py
    python scripts/10_holm_correction.py --baseline P0 --challengers P1 P2 P3 P4
    python scripts/10_holm_correction.py --alpha 0.05

Outputs:
  results/tables/holm_corrected.csv        (one row per comparison, sorted by p)
  results/tables/holm_correction_report.txt
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

OUT_CSV = C.TABLES_DIR / "holm_corrected.csv"
REPORT_PATH = C.TABLES_DIR / "holm_correction_report.txt"


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, returned in the original input order.

    Implements the step-down procedure with a running maximum to keep the
    adjusted values monotone in the raw values, then clips at 1.0.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()

    order = np.argsort(p, kind="mergesort")   # stable, so ties keep input order
    ranked = p[order]

    multipliers = m - np.arange(m)            # m, m-1, ..., 1
    adj_sorted = np.maximum.accumulate(ranked * multipliers)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def collect_comparisons(baseline: str, challengers: list[str]) -> pd.DataFrame:
    """Gather every (challenger, metric) row from the paired-stats summaries."""
    rows = []
    for ch in challengers:
        fp = C.TABLES_DIR / f"paired_{baseline}_vs_{ch}_summary.csv"
        if not fp.exists():
            print(f"  WARNING: missing {fp.name}, skipping {ch}")
            continue
        df = pd.read_csv(fp)
        for _, r in df.iterrows():
            rows.append({
                "baseline": baseline,
                "challenger": ch,
                "metric": r["metric"],
                "baseline_rate": r["A_rate"],
                "challenger_rate": r["B_rate"],
                "delta": r["delta(B-A)"],
                "p_raw": r["mcnemar_exact_p"],
            })
    if not rows:
        raise FileNotFoundError(
            "No paired summary files found. Run scripts/04_paired_stats.py first."
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="P0")
    ap.add_argument("--challengers", nargs="+", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--primary-challenger", default="P4",
                    help="Challenger of the pre-specified primary comparison.")
    ap.add_argument("--primary-metric", default="pass",
                    help="Metric of the pre-specified primary comparison.")
    args = ap.parse_args()

    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_comparisons(args.baseline, args.challengers)
    df["p_holm"] = holm_adjust(df["p_raw"].to_numpy())
    df["significant"] = df["p_holm"] < args.alpha

    # Sort ascending by raw p so the step-down order is visible in the output.
    df = df.sort_values("p_raw").reset_index(drop=True)
    df["holm_rank"] = np.arange(1, len(df) + 1)
    df["holm_threshold"] = args.alpha / (len(df) - df["holm_rank"] + 1)

    cols = ["holm_rank", "baseline", "challenger", "metric",
            "baseline_rate", "challenger_rate", "delta",
            "p_raw", "holm_threshold", "p_holm", "significant"]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    m = len(df)
    n_sig = int(df["significant"].sum())
    is_primary = ((df["challenger"] == args.primary_challenger)
                  & (df["metric"] == args.primary_metric))

    lines = [
        "Holm-Bonferroni correction report",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Baseline: {args.baseline}   Challengers: {', '.join(args.challengers)}",
        f"Family size m = {m}   alpha = {args.alpha}",
        "",
        "The pre-specified primary comparison "
        f"({args.baseline} vs {args.primary_challenger} on {args.primary_metric}) "
        "is reported uncorrected;",
        "it is listed below for completeness but is not part of the corrected family",
        "in the paper's framing.",
        "",
        f"Comparisons surviving correction at alpha={args.alpha}: {n_sig} of {m}",
        "",
        f"{'rank':>4}  {'compare':<12} {'metric':<6} {'delta':>9} "
        f"{'p_raw':>11} {'p_holm':>11}  sig",
        "-" * 68,
    ]
    for _, r in df.iterrows():
        tag = f"{r['baseline']} vs {r['challenger']}"
        lines.append(
            f"{int(r['holm_rank']):>4}  {tag:<12} {r['metric']:<6} "
            f"{r['delta']:>+9.4f} {r['p_raw']:>11.3e} {r['p_holm']:>11.3e}"
            f"  {'yes' if r['significant'] else 'NO'}"
        )

    failed = df[~df["significant"]]
    lines += ["", "Did NOT survive correction:"]
    if len(failed) == 0:
        lines.append("  (none)")
    else:
        for _, r in failed.iterrows():
            lines.append(
                f"  {r['baseline']} vs {r['challenger']} / {r['metric']}: "
                f"raw p={r['p_raw']:.3e} -> Holm p={r['p_holm']:.3e}"
            )

    if is_primary.any():
        pr = df[is_primary].iloc[0]
        lines += [
            "",
            f"Primary comparison ({args.baseline} vs {args.primary_challenger}, "
            f"{args.primary_metric}): raw p = {pr['p_raw']:.3e}, "
            f"Holm p = {pr['p_holm']:.3e}",
        ]

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report)
    print(report)
    print("Saved table:", OUT_CSV)
    print("Saved report:", REPORT_PATH)


if __name__ == "__main__":
    main()
