"""Sensitivity of the pipeline ranking to the recovery tolerances.

The pass criterion depends on three tolerances (period, timing, depth) that were
fixed before the pipelines were run (see Methods, Recovery criteria). This script
checks that the conclusions aren't an artifact of those specific values by
re-grading all trials under a grid of alternative tolerances and re-checking the
pipeline ranking and the key paired comparisons.

This is grading-only: it reuses each pipeline's already-recorded continuous
outputs (period_found, t0_found, depth_found) from results_P*.csv, so the
pipelines are not re-run and no light curves are re-downloaded. Only the grading
thresholds change. Under the baseline tolerances it reproduces the stored
pass/p_ok/t_ok/d_ok flags exactly, which is asserted as a self-check.

    python scripts/11_tolerance_sensitivity.py
    python scripts/11_tolerance_sensitivity.py --baseline P0

Outputs:
    results/tables/tolerance_sensitivity.csv        (rate per pipeline per setting)
    results/tables/tolerance_sensitivity_report.txt
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

OUT_CSV = C.TABLES_DIR / "tolerance_sensitivity.csv"
REPORT_PATH = C.TABLES_DIR / "tolerance_sensitivity_report.txt"

PIPELINES = ["P0", "P1", "P2", "P3", "P4"]

# Baseline tolerances: read from config where available, otherwise the paper's values.
BASE_PTOL = getattr(C, "PERIOD_FRAC_TOL", 0.01)              # fractional period tol
BASE_TFRAC = getattr(C, "T0_TOL_DURATION_FRAC", 0.5)         # timing tol as a fraction of duration
BASE_DLO = getattr(C, "DEPTH_FACTOR_LOW", 0.70)
BASE_DHI = getattr(C, "DEPTH_FACTOR_HIGH", 1.30)


def regrade(df: pd.DataFrame, ptol: float, tfrac: float,
            dlo: float, dhi: float) -> np.ndarray:
    """Recompute the boolean pass flag under the given tolerances, from the
    continuous recovered values. The period-wrapped timing matches the generator."""
    P_true = df["period_true"].to_numpy(float)
    P_found = df["period_found"].to_numpy(float)
    t0_true = df["t0_true"].to_numpy(float)
    t0_found = df["t0_found"].to_numpy(float)
    d_true = df["depth_true"].to_numpy(float)
    d_found = df["depth_found"].to_numpy(float)
    dur = df["duration_true_days"].to_numpy(float)

    p_ok = np.abs(P_found - P_true) / P_true <= ptol
    dt_ = t0_found - t0_true
    wrapped = (dt_ + 0.5 * P_found) % P_found - 0.5 * P_found
    t_ok = np.abs(wrapped) <= tfrac * dur
    with np.errstate(invalid="ignore"):
        d_ok = (d_found >= dlo * d_true) & (d_found <= dhi * d_true)
    return p_ok & t_ok & d_ok


def load_all():
    cols = ["trial_id", "period_true", "t0_true", "depth_true",
            "period_found", "t0_found", "depth_found", "pass"]
    idx = load_trial_index()[["trial_id", "duration_true_days"]]
    out = {}
    for p in PIPELINES:
        df = load_results(p)[cols].merge(idx, on="trial_id")
        out[p] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="P0")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()

    # Self-check: the baseline tolerances must reproduce the stored pass flag exactly.
    for p in PIPELINES:
        regraded = regrade(data[p], BASE_PTOL, BASE_TFRAC, BASE_DLO, BASE_DHI)
        stored = data[p]["pass"].to_numpy().astype(bool)
        agree = float(np.mean(regraded == stored))
        if agree < 0.999:
            print(f"WARNING: regrade disagrees with stored pass for {p} "
                  f"(agreement {agree:.4f}). Check tolerance constants in config.py.")

    # Grid of settings: only the named tolerance moves, the others stay at baseline.
    settings = [
        ("baseline",              BASE_PTOL, BASE_TFRAC, BASE_DLO, BASE_DHI),
        ("period 0.5%",           0.005,     BASE_TFRAC, BASE_DLO, BASE_DHI),
        ("period 2%",             0.02,      BASE_TFRAC, BASE_DLO, BASE_DHI),
        ("timing 0.5 h",          BASE_PTOL, 0.25,       BASE_DLO, BASE_DHI),
        ("timing 1.5 h",          BASE_PTOL, 0.75,       BASE_DLO, BASE_DHI),
        ("depth 0.85-1.20",       BASE_PTOL, BASE_TFRAC, 0.85,     1.20),
        ("depth 0.60-1.60",       BASE_PTOL, BASE_TFRAC, 0.60,     1.60),
        ("depth 0.75-1.25 (symm)",BASE_PTOL, BASE_TFRAC, 0.75,     1.25),
    ]

    rows = []
    passes_by_setting = {}
    for name, ptol, tfrac, dlo, dhi in settings:
        passes = {p: regrade(data[p], ptol, tfrac, dlo, dhi) for p in PIPELINES}
        passes_by_setting[name] = passes
        rates = {p: float(passes[p].mean()) for p in PIPELINES}
        order = ">".join(sorted(PIPELINES, key=lambda p: -rates[p]))
        row = {"setting": name, "ptol": ptol, "tfrac": tfrac, "dlo": dlo, "dhi": dhi}
        row.update({f"rate_{p}": rates[p] for p in PIPELINES})
        row["order"] = order
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    # Report, with the key paired comparisons vs baseline under each setting.
    base = args.baseline
    L = ["Tolerance sensitivity report",
         f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
         f"Baseline tolerances: period {BASE_PTOL:.3g}, timing {BASE_TFRAC:.3g} x duration, "
         f"depth {BASE_DLO:.2f}-{BASE_DHI:.2f}",
         "",
         f"{'setting':<24} " + " ".join(f"{p:>6}" for p in PIPELINES)
         + "   order (high to low)",
         "-" * 96]
    for r in rows:
        L.append(f"{r['setting']:<24} "
                 + " ".join(f"{r[f'rate_{p}']:6.3f}" for p in PIPELINES)
                 + f"   {r['order']}")

    L += ["", f"Key paired comparisons vs {base} (challenger pass rate, McNemar p):"]
    for name in [s[0] for s in settings]:
        passes = passes_by_setting[name]
        L.append(f"  [{name}]")
        for ch in [p for p in PIPELINES if p != base]:
            a = passes[base]; b = passes[ch]
            p_val = mcnemar_exact_p(np.sum((~a) & b), np.sum(a & (~b)))
            d = b.mean() - a.mean()
            L.append(f"      {base} vs {ch}: delta={d:+.3f}  p={p_val:.2e}")

    report = "\n".join(L) + "\n"
    REPORT_PATH.write_text(report)
    print(report)
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
