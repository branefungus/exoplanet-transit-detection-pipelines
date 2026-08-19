#!/usr/bin/env python3
"""
Regrade recovery outcomes under a new depth-tolerance window, without rerunning
any pipeline or regenerating any trial.

Why this is valid: the depth window (DEPTH_FACTOR_LOW / DEPTH_FACTOR_HIGH) is a
grading tolerance. It's applied when a recovered result is scored against truth,
not when trials are generated or when a pipeline searches. So the continuous
outputs (period_found, t0_found, depth_found) don't change; only d_ok and the
overall pass flag can. This script recomputes exactly those two columns from the
frozen continuous values already sitting in each results_P*.csv.

It reads the depth factors from config.py so there's a single source of truth,
then rewrites d_ok and pass in place (a .bak backup is made first).

Run from the repo root, after editing DEPTH_FACTOR_LOW/HIGH in config.py:
    python regrade_depth_window.py
    python regrade_depth_window.py --tables results/tables --pipelines P0 P1 P2 P3 P4
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load_depth_factors(repo_root: Path):
    """Import DEPTH_FACTOR_LOW/HIGH from the repo's config.py, so the regrade
    uses the exact same constants everything else reads."""
    sys.path.insert(0, str(repo_root))
    import config as C
    return float(C.DEPTH_FACTOR_LOW), float(C.DEPTH_FACTOR_HIGH)


def regrade_one(df: pd.DataFrame, dlo: float, dhi: float) -> pd.DataFrame:
    need = {"depth_true", "depth_found", "p_ok", "t_ok"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"results file missing columns {missing}; cannot regrade")
    d_true = df["depth_true"].to_numpy(float)
    d_found = df["depth_found"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        d_ok = (d_found >= dlo * d_true) & (d_found <= dhi * d_true)
    # p_ok and t_ok don't depend on the depth window, so reuse them as stored.
    p_ok = df["p_ok"].to_numpy(bool)
    t_ok = df["t_ok"].to_numpy(bool)
    out = df.copy()
    out["d_ok"] = d_ok.astype(int)
    out["pass"] = (p_ok & t_ok & d_ok).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables",
                    help="directory holding results_P*.csv")
    ap.add_argument("--pipelines", nargs="+", default=["P0", "P1", "P2", "P3", "P4"])
    ap.add_argument("--repo-root", default=".",
                    help="dir containing config.py (default: current dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print before/after rates but don't write files")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tables = Path(args.tables)
    dlo, dhi = load_depth_factors(repo_root)
    print(f"Using depth window from config.py: {dlo:.3f} .. {dhi:.3f}\n")

    print(f"{'pipe':<5} {'old_pass':>9} {'new_pass':>9} {'old_d_ok':>9} {'new_d_ok':>9}")
    for p in args.pipelines:
        fp = tables / f"results_{p}.csv"
        if not fp.exists():
            print(f"  WARNING: {fp} not found, skipping")
            continue
        df = pd.read_csv(fp)
        old_pass = df["pass"].mean()
        old_dok = df["d_ok"].mean()
        new = regrade_one(df, dlo, dhi)
        print(f"{p:<5} {old_pass:9.3f} {new['pass'].mean():9.3f} "
              f"{old_dok:9.3f} {new['d_ok'].mean():9.3f}")
        if not args.dry_run:
            shutil.copyfile(fp, fp.with_suffix(".csv.bak"))
            new.to_csv(fp, index=False)
    if args.dry_run:
        print("\n[dry run] no files written.")
    else:
        print("\nDone. Originals saved as results_P*.csv.bak")
        print("NOTE: null-trial files (results_P*_null.csv) are NOT touched; the")
        print("false-positive analysis uses the snr statistic, not the depth window.")


if __name__ == "__main__":
    main()
