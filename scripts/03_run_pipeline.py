"""Run one pipeline (or all of them) over every trial in the index.

This replaces the old 10_run_P0_master.py ... 14_run_P4_master.py with a single
script:

    python scripts/03_run_pipeline.py --pipeline P0
    python scripts/03_run_pipeline.py --pipeline all
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.io_utils import load_trial_index, load_trial, save_results
from src.common.scoring import evaluate_recovery, detection_snr
from src.pipelines import PIPELINES


def run_one_pipeline(name: str, null: bool = False) -> None:
    run_fn = PIPELINES[name]
    idx = load_trial_index(null=null)
    print(f"\n=== {name}{' (null)' if null else ''}: {len(idx)} trials ===")

    rows = []
    for i, r in idx.iterrows():
        trial_id = int(r["trial_id"])
        dur = float(r.get("duration_true_days", C.DURATION_DAYS))
        b_ld = float(r.get("impact_parameter", C.LD_B_DEFAULT))
        try:
            t, y = load_trial(str(r["file"]))
            P, t0, depth, score = run_fn(t, y, duration_days=dur, b_ld=b_ld)
            snr = detection_snr(t, y, P, t0, dur, depth)
            if null:
                # No injected truth here, so just record the reported detection.
                rows.append({
                    "trial_id": trial_id,
                    "period_found": float(P), "t0_found": float(t0),
                    "depth_found": float(depth), "score": float(score),
                    "snr": float(snr), "error": "",
                })
            else:
                passed, p_ok, t_ok, d_ok = evaluate_recovery(
                    period_true=float(r["period_true"]), t0_true=float(r["t0_true"]),
                    depth_true=float(r["depth_true"]),
                    period_found=float(P), t0_found=float(t0), depth_found=float(depth))
                rows.append({
                    "trial_id": trial_id,
                    "period_true": float(r["period_true"]), "t0_true": float(r["t0_true"]),
                    "depth_true": float(r["depth_true"]),
                    "period_found": float(P), "t0_found": float(t0),
                    "depth_found": float(depth), "score": float(score), "snr": float(snr),
                    "pass": passed, "p_ok": p_ok, "t_ok": t_ok, "d_ok": d_ok,
                    "error": "",
                })
        except Exception as e:
            base = {"trial_id": trial_id, "period_found": np.nan, "t0_found": np.nan,
                    "depth_found": np.nan, "score": np.nan, "snr": np.nan, "error": repr(e)}
            if not null:
                base.update({
                    "period_true": float(r.get("period_true", np.nan)),
                    "t0_true": float(r.get("t0_true", np.nan)),
                    "depth_true": float(r.get("depth_true", np.nan)),
                    "pass": 0, "p_ok": 0, "t_ok": 0, "d_ok": 0})
            rows.append(base)
        if (i + 1) % 200 == 0:
            print(f"{name} progress: {i + 1}/{len(idx)}")

    df = save_results(rows, C.results_csv(name, null=null))
    n_err = int((df["error"].astype(str) != "").sum())
    if n_err:
        print(f"WARNING: {n_err} trials errored in {name}; check the error column.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="all",
                    choices=list(PIPELINES.keys()) + ["all"])
    ap.add_argument("--null", action="store_true",
                    help="Run on the injection-free null trials.")
    args = ap.parse_args()

    names = list(PIPELINES.keys()) if args.pipeline == "all" else [args.pipeline]
    for name in names:
        run_one_pipeline(name, null=args.null)


if __name__ == "__main__":
    main()
