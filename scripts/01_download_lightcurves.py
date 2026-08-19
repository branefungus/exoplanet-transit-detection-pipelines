"""Download one TESS light curve per TIC (SPOC 2-min preferred), save each as a
time/flux CSV, and write a manifest.

Includes retry/backoff on transient network errors. On PDCSAP: in current
lightkurve a downloaded SPOC LightCurve doesn't expose a pdcsap_flux attribute
(the FLUX column of a SPOC product already is PDCSAP), so we check the column set
explicitly and record which flux column was used in the manifest.
"""
from __future__ import annotations

from pathlib import Path
import http.client
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

import lightkurve as lk
import astropy.units as u
from astropy.utils.data import conf as astropy_conf

# Give MAST/astropy a generous network timeout; both are flaky under load.
try:
    from astroquery.mast import Conf as MastConf
    MastConf.timeout = 120
except Exception:
    MastConf = None
astropy_conf.remote_timeout = 120

MAX_RETRIES = 6
BASE_SLEEP_SEC = 2.0
PAUSE_BETWEEN_TICS_SEC = 1.0


def _to_seconds(exptime):
    """Coerce an exposure time (Quantity, number, or None) to seconds as a float."""
    if exptime is None:
        return np.nan
    if hasattr(exptime, "to_value"):
        try:
            return float(exptime.to_value(u.s))
        except Exception:
            try:
                return float(exptime.value)
            except Exception:
                return np.nan
    try:
        return float(exptime)
    except Exception:
        return np.nan


def _row_npts(row) -> int:
    for attr in ("n_cadences", "size"):
        v = getattr(row, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0


def score_search_row(row):
    """Sort key for product preference: prefer PREFER_AUTHOR, then exposure time
    nearest PREFER_EXPTIME_SECONDS, then more points. Lower sorts better."""
    author = str(getattr(row, "author", "") or "")
    exptime_sec = _to_seconds(getattr(row, "exptime", None))
    author_penalty = 0 if author.upper() == C.PREFER_AUTHOR else 1
    exptime_penalty = 1e9 if not np.isfinite(exptime_sec) else abs(exptime_sec - float(C.PREFER_EXPTIME_SECONDS))
    return (author_penalty, exptime_penalty, -_row_npts(row))


def pick_best_product(search_result):
    """Best product by score, restricted to those with at least MIN_POINTS points
    when any qualify; otherwise the best-scoring row overall."""
    if len(search_result) == 0:
        return None
    rows = sorted(list(search_result), key=score_search_row)
    for r in rows:
        if _row_npts(r) >= C.MIN_POINTS:
            return r
    return rows[0]


def _is_transient(e: Exception) -> bool:
    """Whether an exception looks like a transient network hiccup worth retrying."""
    if isinstance(e, http.client.RemoteDisconnected):
        return True
    msg = str(e).lower()
    return any(m in msg for m in [
        "remote end closed connection", "connection aborted", "connection reset",
        "connection refused", "timed out", "timeout", "temporarily unavailable",
        "503", "502", "504", "429", "server error",
    ])


def _retry_sleep(attempt: int):
    # Exponential backoff with a little jitter.
    time.sleep(BASE_SLEEP_SEC * (2 ** (attempt - 1)) + np.random.uniform(0.0, 0.5))


def extract_time_flux(lc) -> tuple[np.ndarray, np.ndarray, str]:
    """Pull (time, flux, flux_column_name) out of a LightCurve. Prefer an explicit
    pdcsap_flux column; for SPOC products the default flux column already is PDCSAP."""
    cols = set(getattr(lc, "colnames", []) or [])
    if "pdcsap_flux" in cols:
        flux = np.asarray(lc["pdcsap_flux"].value, dtype=float)
        flux_name = "pdcsap_flux"
    else:
        flux = np.asarray(lc.flux.value, dtype=float)
        flux_col = str(getattr(lc.meta, "get", lambda *_: "")("FLUX_ORIGIN") or "")
        flux_name = flux_col if flux_col else "flux_default"
    return np.asarray(lc.time.value, dtype=float), flux, flux_name


def download_one_tic(tic_id: int):
    target = f"TIC {int(tic_id)}"
    print(f"\nSearching: {target}")

    sr = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sr = lk.search_lightcurve(target, mission="TESS")
            break
        except Exception as e:
            if attempt == MAX_RETRIES or not _is_transient(e):
                raise
            print(f"  Search failed ({attempt}/{MAX_RETRIES}): {e}")
            _retry_sleep(attempt)

    if sr is None or len(sr) == 0:
        print("  No TESS light curves found.")
        return None

    best = pick_best_product(sr)
    if best is None:
        print("  Could not choose a product.")
        return None

    print(f"  Chosen: author={getattr(best, 'author', None)} "
          f"exptime={getattr(best, 'exptime', None)} sector={getattr(best, 'sector', None)}")

    lc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            lc = best.download(download_dir=str(C.STARS_DIR))
            break
        except Exception as e:
            if attempt == MAX_RETRIES or not _is_transient(e):
                raise
            print(f"  Download failed ({attempt}/{MAX_RETRIES}): {e}")
            _retry_sleep(attempt)

    if lc is None:
        print("  Download failed (lc is None).")
        return None

    # Stitch multi-sector collections down to a single light curve if needed.
    if hasattr(lc, "stitch"):
        try:
            lc = lc.stitch()
        except Exception:
            pass

    time_days, flux, flux_name = extract_time_flux(lc)
    df = pd.DataFrame({"time": time_days, "flux": flux})
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("time")

    out_csv = C.STARS_DIR / f"tic_{int(tic_id)}_{flux_name}.csv"
    df.to_csv(out_csv, index=False)
    print("  Saved:", out_csv, f"({len(df)} points, flux={flux_name})")
    return out_csv


def main():
    C.STARS_DIR.mkdir(parents=True, exist_ok=True)
    if not C.TIC_LIST_CSV.exists():
        raise FileNotFoundError(f"Missing {C.TIC_LIST_CSV}. Run scripts/00_build_tic_list.py first.")

    tics = pd.read_csv(C.TIC_LIST_CSV)["tic_id"].astype(int).tolist()
    saved, failed = [], []

    for tic in tics:
        # Skip TICs we've already fetched, so reruns are cheap and idempotent.
        existing = list(C.STARS_DIR.glob(f"tic_{int(tic)}_*.csv"))
        if existing:
            print(f"\nSkipping TIC {tic} (already downloaded: {existing[0].name})")
            saved.append({"tic_id": int(tic), "file": str(existing[0].relative_to(C.ROOT)),
                          "status": "skipped_existing"})
            time.sleep(PAUSE_BETWEEN_TICS_SEC)
            continue
        try:
            out = download_one_tic(tic)
            if out is not None:
                saved.append({"tic_id": int(tic), "file": str(out.relative_to(C.ROOT)),
                              "status": "downloaded"})
            else:
                failed.append({"tic_id": int(tic), "error": "no_product"})
        except Exception as e:
            print(f"  FAILED TIC {tic}: {e}")
            failed.append({"tic_id": int(tic), "error": repr(e)})
        time.sleep(PAUSE_BETWEEN_TICS_SEC)

    pd.DataFrame(saved).to_csv(C.MANIFEST_CSV, index=False)
    pd.DataFrame(failed).to_csv(C.STARS_DIR / "download_failures.csv", index=False)
    print("\nSaved manifest:", C.MANIFEST_CSV)
    print("Downloaded/skipped:", len(saved), " Failed:", len(failed), " Total:", len(tics))


if __name__ == "__main__":
    main()
