"""Shared file I/O: loading the trial index and per-pipeline results tables, for
both the real and the null trial sets. Replaces the loaders and path resolvers
that were duplicated across the old runners.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import config as C


def resolve_trial_path(file_value: str) -> Path:
    """Turn a path from the index into a real one. The current repo always writes
    repo-root-relative paths, so this is mostly trivial now; the old
    multi-fallback resolver only existed because paths drifted between folder
    layouts."""
    p = Path(str(file_value))
    if p.is_absolute() and p.exists():
        return p
    q = C.ROOT / p
    if q.exists():
        return q
    if p.exists():
        return p
    # Last resort: search by basename under the trials directory.
    hits = sorted(C.TRIALS_DIR.rglob(p.name), key=lambda h: len(str(h)))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Trial CSV not found: {file_value}")


def load_time_flux_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "time" not in df.columns or "flux" not in df.columns:
        raise ValueError(f"File missing time/flux columns: {path}")
    t = df["time"].to_numpy(dtype=float)
    y = df["flux"].to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    order = np.argsort(t)
    return t[order], y[order]


def load_trial(file_value: str) -> tuple[np.ndarray, np.ndarray]:
    return load_time_flux_csv(resolve_trial_path(file_value))


def load_trial_index(null: bool = False) -> pd.DataFrame:
    path = C.TRIAL_INDEX_NULL_CSV if null else C.TRIAL_INDEX_CSV
    if not path.exists():
        which = "--null " if null else ""
        raise FileNotFoundError(
            f"Missing trial index {path}. Run scripts/02_generate_trials.py {which}first."
        )
    idx = pd.read_csv(path)
    needed = {"trial_id", "file", "noise_level", "gap_level", "duration_true_days"}
    if not null:
        needed |= {"period_true", "t0_true", "depth_true"}
    missing = needed - set(idx.columns)
    if missing:
        raise ValueError(f"Trial index missing columns: {sorted(missing)}")
    return idx


def save_results(rows: list[dict], out_csv: Path) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    if "pass" in df.columns and len(df):
        print(f"Pass rate: {float(df['pass'].mean()):.4f}  (n={len(df)})")
    return df


def load_results(pipeline: str, null: bool = False) -> pd.DataFrame:
    path = C.results_csv(pipeline, null=null)
    if not path.exists():
        flag = " --null" if null else ""
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/03_run_pipeline.py --pipeline {pipeline}{flag}"
        )
    df = pd.read_csv(path)
    needed = {"trial_id", "snr"} if null else {"trial_id", "pass", "p_ok", "t_ok", "d_ok"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df
