"""Export direct MAST download links for the products actually used, for the
paper's data-availability section.

Reads the TIC list from data/raw/tic_list.csv (so it always matches the sample
that was actually run), searches TESS products for each, picks the best one with
the same preference order used at download time (author, then exposure time,
then point count), and writes a table of MAST download URLs.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import sys

import numpy as np
import pandas as pd
import lightkurve as lk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

OUT_CSV = C.TABLES_DIR / "mast_links.csv"


def _get_col(tbl, names, default=None):
    """First matching column from the search-result table, or a default."""
    for n in names:
        if n in tbl.colnames:
            return tbl[n]
    return default


def pick_best_row(sr: lk.SearchResult):
    """Choose the preferred product row: prefer PREFER_AUTHOR, then the exposure
    time closest to PREFER_EXPTIME_SECONDS, then the most points. Restrict to
    products with at least MIN_POINTS points when any qualify."""
    tbl = sr.table
    if len(tbl) == 0:
        return None
    author = np.array(_get_col(tbl, ["author"], default=np.array([""] * len(tbl))), dtype=str)
    exptime = np.array(_get_col(tbl, ["exptime", "t_exptime", "tess_exptime"],
                                default=np.full(len(tbl), np.nan)), dtype=float)
    npts = _get_col(tbl, ["n_cadences", "size", "n_points", "nobs"], default=np.zeros(len(tbl)))
    try:
        npts = np.array(npts, dtype=float)
    except Exception:
        npts = np.zeros(len(tbl), dtype=float)

    author_pen = np.where(np.char.upper(author) == C.PREFER_AUTHOR, 0.0, 1.0)
    exp_pen = np.abs(exptime - float(C.PREFER_EXPTIME_SECONDS))
    exp_pen = np.where(np.isfinite(exp_pen), exp_pen, 1e9)

    ok_size = npts >= float(C.MIN_POINTS)
    mask = ok_size if np.any(ok_size) else np.ones(len(tbl), dtype=bool)

    best, best_i = None, None
    for i in np.where(mask)[0]:
        key = (float(author_pen[i]), float(exp_pen[i]), float(-npts[i]))
        if best is None or key < best:
            best, best_i = key, int(i)
    return tbl[best_i] if best_i is not None else None


def mast_download_url(data_uri: str) -> str:
    return "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(data_uri, safe=":/")


def main():
    tics = pd.read_csv(C.TIC_LIST_CSV)["tic_id"].astype(int).tolist()
    rows = []
    for tic in tics:
        print("Searching: TIC", tic)
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        if len(sr) == 0:
            rows.append({"tic_id": tic, "status": "no_results"})
            continue
        row = pick_best_row(sr)
        if row is None:
            rows.append({"tic_id": tic, "status": "no_pick"})
            continue
        data_uri = str(row["dataURI"]) if "dataURI" in row.colnames else ""
        rows.append({
            "tic_id": tic,
            "author": str(row["author"]) if "author" in row.colnames else "",
            "sector_or_seq": str(row["sequence_number"]) if "sequence_number" in row.colnames
            else (str(row["sector"]) if "sector" in row.colnames else ""),
            "exptime_s": str(row["exptime"]) if "exptime" in row.colnames
            else (str(row["t_exptime"]) if "t_exptime" in row.colnames else ""),
            "productFilename": str(row["productFilename"]) if "productFilename" in row.colnames else "",
            "dataURI": data_uri,
            "mast_download_url": mast_download_url(data_uri) if data_uri else "",
            "status": "ok" if data_uri else "missing_dataURI",
        })
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
