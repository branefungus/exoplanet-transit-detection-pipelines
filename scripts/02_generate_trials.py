"""Generate the injection-recovery trials for every star in the manifest.

Output:
  data/trials/tic_<id>/trial_XXXXXX.csv   (columns: time, flux)
  results/tables/trial_index.csv

The logic is identical to the old generator. The index now always contains the
canonical columns the rest of the code expects (trial_id, file, tic_id, star_id,
noise/gap levels, truths, rng_seed, impact_parameter, ld_u1, ld_u2), so the old
build_master_index step isn't needed anymore.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common.injection import build_ld_table, transit_delta_ld, LimbDarkOccultationTable


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.std(x))
    return s if np.isfinite(s) and s > 0 else 1.0


def load_star_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    df = pd.read_csv(path)
    if "time" not in df.columns or "flux" not in df.columns:
        raise ValueError(f"Star CSV missing time/flux columns: {path}")
    t = df["time"].to_numpy(dtype=float)
    y = df["flux"].to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    order = np.argsort(t)
    t, y = t[order], y[order]

    med = np.nanmedian(y)
    if not np.isfinite(med) or med == 0:
        raise ValueError(f"Bad median flux for {path}")
    y = y / med

    # Sigma-clip against a robust scatter estimate to drop obvious outliers.
    resid = y - np.nanmedian(y)
    keep = np.abs(resid) <= (C.SIGMA_CLIP * robust_sigma(resid))
    t, y = t[keep], y[keep]

    time_offset = float(np.nanmin(t))
    return t - time_offset, y, time_offset


def ou_red_noise(t: np.ndarray, sigma: float, rho_days: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck (exponentially correlated) noise sampled on the actual,
    unevenly spaced timestamps."""
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n == 0:
        return np.array([], dtype=float)
    if rho_days <= 0:
        return rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.0
        a = np.exp(-dt / rho_days)
        x[i] = a * x[i - 1] + rng.normal(0.0, sigma * np.sqrt(max(0.0, 1.0 - a * a)))
    return x


def apply_gap_blocks(t: np.ndarray, n_blocks: int, block_len_days: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Return a keep-mask with n_blocks randomly placed windows removed."""
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n == 0:
        return np.array([], dtype=bool)
    tmin, tmax = float(np.min(t)), float(np.max(t))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for _ in range(int(max(0, n_blocks))):
        start = rng.uniform(tmin, max(tmin, tmax - block_len_days))
        keep &= ~((t >= start) & (t <= start + block_len_days))
    return keep


def choose_period_depth_t0(t: np.ndarray, rng: np.random.Generator):
    tmin, tmax = float(np.min(t)), float(np.max(t))
    span = tmax - tmin

    period_true = float(rng.uniform(C.PERIOD_MIN, C.PERIOD_MAX))
    depth_true = float(rng.uniform(C.DEPTH_MIN, C.DEPTH_MAX))

    # Draw a t0 that puts at least two transits inside the baseline.
    t0_true, sampled_transits, tries = np.nan, 0, 0
    while tries < C.T0_MAX_TRIES:
        tries += 1
        t0_try = tmin + float(rng.uniform(0.0, period_true))
        centers = np.arange(np.floor((tmin - t0_try) / period_true) - 2,
                            np.ceil((tmax - t0_try) / period_true) + 3, dtype=int)
        tc = t0_try + centers * period_true
        tc = tc[(tc >= tmin) & (tc <= tmax)]
        if tc.size >= 2:
            t0_true = float(t0_try)
            sampled_transits = int(tc.size)
            break
    if not np.isfinite(t0_true):
        t0_true = float(tmin + 0.25 * period_true)
        sampled_transits = int(max(1, np.floor(span / period_true)))
    return period_true, depth_true, t0_true, sampled_transits, tries


def make_one_trial(t_star, y_star, rng, noise_level, gap_level,
                   ld_table: LimbDarkOccultationTable, null: bool = False):
    t = np.asarray(t_star, dtype=float)
    y_base = np.asarray(y_star, dtype=float)

    if null:
        # Null trial: same noise and gaps, but no injected transit. The truths
        # are NaN, since there's nothing to recover.
        period_true = float("nan")
        depth_true = float("nan")
        t0_true = float("nan")
        sampled_transits = 0
        t0_tries = 0
        y = y_base.copy()
    else:
        period_true, depth_true, t0_true, sampled_transits, t0_tries = choose_period_depth_t0(t, rng)
        delta = transit_delta_ld(t=t, period=period_true, t0=t0_true,
                                 duration_days=C.DURATION_DAYS, depth_mid=depth_true,
                                 b=C.IMPACT_PARAMETER, ld_table=ld_table)
        y = y_base * (1.0 - delta)

    if str(noise_level).lower() == "high":
        white_sigma, red_sigma = C.WHITE_SIGMA_HIGH, C.RED_SIGMA_HIGH
    else:
        white_sigma, red_sigma = C.WHITE_SIGMA_LOW, C.RED_SIGMA_LOW

    y = y + ou_red_noise(t, sigma=red_sigma, rho_days=C.RED_RHO_DAYS, rng=rng)
    y = y + rng.normal(0.0, white_sigma, size=len(t))

    if str(gap_level).lower() == "severe":
        n_blocks, block_len = C.GAP_BLOCKS_SEVERE, C.GAP_LEN_DAYS_SEVERE
    else:
        n_blocks, block_len = C.GAP_BLOCKS_MINIMAL, C.GAP_LEN_DAYS_MINIMAL

    keep = apply_gap_blocks(t, n_blocks=n_blocks, block_len_days=block_len, rng=rng)
    if int(np.sum(keep)) < C.MIN_POINTS_AFTER_GAPS:
        keep = np.ones_like(keep, dtype=bool)

    meta = {
        "noise_level": str(noise_level), "gap_level": str(gap_level),
        "period_true": float(period_true), "t0_true": float(t0_true),
        "duration_true_days": float(C.DURATION_DAYS), "depth_true": float(depth_true),
        "n_points": int(np.sum(keep)),
        "sampled_transits": int(sampled_transits), "t0_tries": int(t0_tries),
        "white_sigma": float(white_sigma), "red_sigma": float(red_sigma),
        "red_rho_days": float(C.RED_RHO_DAYS),
        "gap_blocks": int(n_blocks), "gap_len_days": float(block_len),
        "ld_u1": float(C.LD_U1), "ld_u2": float(C.LD_U2),
        "impact_parameter": float(C.IMPACT_PARAMETER),
        "is_null": int(bool(null)),
    }
    return t[keep], y[keep], meta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true",
                    help="Generate injection-free null trials (for false-positive analysis).")
    args = ap.parse_args()
    null = args.null

    if not C.MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing {C.MANIFEST_CSV}. Run scripts/01_download_lightcurves.py first.")
    mdf = pd.read_csv(C.MANIFEST_CSV)
    if "tic_id" not in mdf.columns or "file" not in mdf.columns:
        raise ValueError(f"Manifest needs tic_id and file columns. Got: {list(mdf.columns)}")
    mdf = mdf.copy()
    mdf["tic_id"] = mdf["tic_id"].astype(int)
    mdf = mdf.sort_values("tic_id").reset_index(drop=True)

    trials_dir = C.TRIALS_NULL_DIR if null else C.TRIALS_DIR
    index_csv = C.TRIAL_INDEX_NULL_CSV if null else C.TRIAL_INDEX_CSV
    # Offset the null seed stream so the null trials are statistically
    # independent of the real ones (same machinery, different draws).
    seed_offset = 500_000 if null else 0

    trials_dir.mkdir(parents=True, exist_ok=True)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {'NULL (no injection)' if null else 'REAL (injected transits)'}")
    print("Manifest:", C.MANIFEST_CSV, f"({len(mdf)} stars)")
    print("Trials dir:", trials_dir)
    print("Index out:", index_csv)

    ld_table = build_ld_table(C.DEPTH_MIN, C.DEPTH_MAX)

    all_rows = []
    trial_id = 1
    for si, row in mdf.iterrows():
        tic_id = int(row["tic_id"])
        raw_file = Path(str(row["file"]))
        raw_path = raw_file if raw_file.is_absolute() else (C.ROOT / raw_file)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw star file not found for TIC {tic_id}: {raw_path}")

        print(f"\n[{si + 1}/{len(mdf)}] TIC {tic_id}: {raw_path.name}")
        t_star, y_star, time_offset = load_star_csv(raw_path)

        star_dir = trials_dir / f"tic_{tic_id}"
        star_dir.mkdir(parents=True, exist_ok=True)
        star_seed = (C.BASE_SEED * 1_000_003 + tic_id + seed_offset) & 0xFFFFFFFF

        for noise_level in C.NOISE_LEVELS:
            for gap_level in C.GAP_LEVELS:
                for k in range(C.TRIALS_PER_CELL):
                    seed = (star_seed
                            + 10_000 * (C.NOISE_LEVELS.index(noise_level) + 1)
                            + 1_000 * (C.GAP_LEVELS.index(gap_level) + 1)
                            + k) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed)

                    t_tr, y_tr, meta = make_one_trial(t_star, y_star, rng,
                                                      noise_level, gap_level, ld_table,
                                                      null=null)

                    trial_path = star_dir / f"trial_{trial_id:06d}.csv"
                    if C.OVERWRITE_TRIAL_FILES or not trial_path.exists():
                        pd.DataFrame({"time": t_tr, "flux": y_tr}).to_csv(trial_path, index=False)

                    all_rows.append({
                        "trial_id": int(trial_id),
                        "file": str(trial_path.relative_to(C.ROOT)),
                        "raw_source": str(raw_path.relative_to(C.ROOT)) if not raw_file.is_absolute() else str(raw_path),
                        "tic_id": int(tic_id),
                        "star_id": f"tic_{tic_id}",
                        "rng_seed": int(seed),
                        "time_offset_original": float(time_offset),
                        **meta,
                    })
                    trial_id += 1

        print(f"  Done TIC {tic_id}. Total trials: {trial_id - 1}")

    idx = pd.DataFrame(all_rows)
    front = ["trial_id", "file", "raw_source", "tic_id", "star_id", "rng_seed",
             "noise_level", "gap_level",
             "period_true", "t0_true", "duration_true_days", "depth_true", "n_points",
             "sampled_transits", "t0_tries"]
    idx = idx[front + [c for c in idx.columns if c not in front]]
    idx.to_csv(index_csv, index=False)

    print("\nSaved index:", index_csv)
    print("Total trials:", len(idx),
          f"({C.TRIALS_PER_CELL * len(C.NOISE_LEVELS) * len(C.GAP_LEVELS)} per star)")


if __name__ == "__main__":
    main()
