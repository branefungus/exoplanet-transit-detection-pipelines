"""Paired comparison of the baseline pipeline against each challenger, on the
exact same trials.

By default it compares P0 (the baseline) against P1, P2, P3, and P4 in one run.
For each comparison and each metric (pass, p_ok, t_ok, d_ok) it reports, both
overall and per (noise, gap) bin: the success rates with Wilson CIs, the paired
rate difference with a CI, the discordant counts, and the exact McNemar p-value.

It writes a human-readable report to results/tables/paired_stats_report.txt and
also saves per-comparison CSVs (which feed the reliability maps and the paper).

    python scripts/04_paired_stats.py
    python scripts/04_paired_stats.py --baseline P0 --challengers P1 P2 P3 P4
    python scripts/04_paired_stats.py --baseline P0 --challengers P4
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
from src.common.scoring import wilson_ci, paired_delta_ci, mcnemar_exact_p, discordant_counts

REPORT_PATH = C.TABLES_DIR / "paired_stats_report.txt"


def merged_frame(baseline: str, challenger: str) -> pd.DataFrame:
    """Inner-join the baseline and challenger metrics onto the trial index, so
    every row is one trial scored by both pipelines."""
    idx = load_trial_index()[["trial_id", "noise_level", "gap_level"]]
    out = idx.copy()
    for label in (baseline, challenger):
        res = load_results(label)[["trial_id"] + C.STATS_METRICS].copy()
        res = res.rename(columns={m: f"{m}_{label}" for m in C.STATS_METRICS})
        out = out.merge(res, on="trial_id", how="inner")
    if len(out) == 0:
        raise RuntimeError(f"No matched trials after merging {baseline} and {challenger}.")
    return out


def summarize(df: pd.DataFrame, A: str, B: str, metric: str) -> dict:
    a = df[f"{metric}_{A}"].to_numpy(dtype=int)
    b = df[f"{metric}_{B}"].to_numpy(dtype=int)
    n = len(df)
    ka, kb = int(a.sum()), int(b.sum())
    a_only, b_only = discordant_counts(a, b)
    mean_d, lo_d, hi_d, _ = paired_delta_ci(a, b)
    cia, cib = wilson_ci(ka, n), wilson_ci(kb, n)
    return {
        "metric": metric, "n": n, "A": A, "B": B,
        "A_rate": ka / n, "A_ci_lo": cia[0], "A_ci_hi": cia[1],
        "B_rate": kb / n, "B_ci_lo": cib[0], "B_ci_hi": cib[1],
        "delta(B-A)": mean_d, "delta_ci_lo": lo_d, "delta_ci_hi": hi_d,
        "both_1": int(np.sum((a == 1) & (b == 1))),
        "A_only": a_only, "B_only": b_only,
        "both_0": int(np.sum((a == 0) & (b == 0))),
        "mcnemar_exact_p": mcnemar_exact_p(a_only, b_only),
    }


def fmt_metric_block(s: dict, A: str, B: str) -> list[str]:
    """Format one metric's overall result as report lines."""
    L = []
    L.append(f"  Metric: {s['metric']}   (n = {s['n']} matched trials)")
    L.append(f"    {A} rate : {s['A_rate']:.4f}  [{s['A_ci_lo']:.4f}, {s['A_ci_hi']:.4f}]")
    L.append(f"    {B} rate : {s['B_rate']:.4f}  [{s['B_ci_lo']:.4f}, {s['B_ci_hi']:.4f}]")
    L.append(f"    delta (B - A) : {s['delta(B-A)']:+.4f}  "
             f"[{s['delta_ci_lo']:+.4f}, {s['delta_ci_hi']:+.4f}]")
    L.append(f"    discordant : {A}_only = {s['A_only']}   {B}_only = {s['B_only']}   "
             f"(both_pass = {s['both_1']}, both_fail = {s['both_0']})")
    L.append(f"    McNemar exact p : {s['mcnemar_exact_p']:.3e}")
    return L


def compare_one(baseline: str, challenger: str, report_lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    A, B = baseline, challenger
    df = merged_frame(A, B)

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append(f"  {A}  vs  {B}      (matched trials: {len(df)})")
    report_lines.append("=" * 70)

    summary_rows, bin_rows = [], []

    # Overall, one block per metric.
    report_lines.append("\n  --- OVERALL ---")
    for metric in C.STATS_METRICS:
        s = summarize(df, A, B, metric)
        summary_rows.append(s)
        report_lines += fmt_metric_block(s, A, B)
        report_lines.append("")

    # Broken out by (noise, gap) bin.
    report_lines.append("  --- BY NOISE / GAP BIN ---")
    for metric in C.STATS_METRICS:
        report_lines.append(f"\n  Metric: {metric}")
        for (nl, gl), g in df.groupby(["noise_level", "gap_level"], dropna=False):
            sb = summarize(g, A, B, metric)
            sb["noise_level"], sb["gap_level"] = nl, gl
            bin_rows.append(sb)
            report_lines.append(
                f"    {str(nl):>4} / {str(gl):<8}  n={sb['n']:>4}  "
                f"{A}={sb['A_rate']:.3f}  {B}={sb['B_rate']:.3f}  "
                f"delta={sb['delta(B-A)']:+.3f} [{sb['delta_ci_lo']:+.3f},{sb['delta_ci_hi']:+.3f}]  "
                f"{A}only={sb['A_only']:>3} {B}only={sb['B_only']:>3}  "
                f"p={sb['mcnemar_exact_p']:.2e}")
    report_lines.append("")

    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="P0")
    ap.add_argument("--challengers", nargs="+", default=["P1", "P2", "P3", "P4"])
    args = ap.parse_args()

    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    header = [
        "Paired pipeline comparison report",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Baseline: {args.baseline}    Challengers: {', '.join(args.challengers)}",
        "Intervals are 90% (Wilson for rates, normal-approx for paired deltas).",
        "McNemar p-values are exact (two-sided).",
    ]
    report_lines = list(header)

    for challenger in args.challengers:
        try:
            summ_df, bin_df = compare_one(args.baseline, challenger, report_lines)
        except FileNotFoundError as e:
            report_lines.append("")
            report_lines.append("=" * 70)
            report_lines.append(f"  {args.baseline} vs {challenger}: SKIPPED ({e})")
            report_lines.append("=" * 70)
            print(f"Skipped {challenger}: {e}")
            continue

        summ_df.to_csv(C.TABLES_DIR / f"paired_{args.baseline}_vs_{challenger}_summary.csv", index=False)
        bin_df.to_csv(C.TABLES_DIR / f"paired_{args.baseline}_vs_{challenger}_by_bin.csv", index=False)
        print(f"Done {args.baseline} vs {challenger}")

    report_text = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report_text)

    print("\n" + report_text)
    print("Report written to:", REPORT_PATH)


if __name__ == "__main__":
    main()
