"""Build the star sample: the N brightest TICs that host a confirmed or known
planet TOI.

Queries the NASA Exoplanet Archive TOI table over TAP, filters to the requested
dispositions (and optional magnitude cut), keeps one row per TIC, and writes the
sample to data/raw/. Selection follows config.py: brightest-N by default, or a
fixed-seed random sample if RANDOM_SAMPLE is set.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def fetch_toi_table() -> pd.DataFrame:
    """Run the TAP query and return the TOI table as a DataFrame."""
    disp_list = ",".join(f"'{d}'" for d in C.DISPOSITIONS)
    where = [f"tfopwg_disp in ({disp_list})", "tid is not null"]
    if C.MAX_TMAG is not None:
        where.append(f"st_tmag <= {float(C.MAX_TMAG)}")

    query = f"""
    select
        tid, toi, toipfx, tfopwg_disp, st_tmag,
        pl_orbper, pl_tranmid, pl_trandurh, pl_trandep, rowupdate
    from toi
    where {" and ".join(where)}
    order by st_tmag asc
    """.strip()

    r = requests.get(TAP_SYNC, params={"query": query, "format": "csv"}, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def main():
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)

    df = fetch_toi_table()
    # Sort brightest-first, then keep a single (the brightest) row per TIC.
    df = df.sort_values(["st_tmag", "tid", "toi"], ascending=[True, True, True])
    df_star = df.drop_duplicates(subset=["tid"], keep="first").copy()

    if len(df_star) < C.N_STARS:
        raise RuntimeError(
            f"Only {len(df_star)} unique TICs after filters, need {C.N_STARS}. "
            "Increase MAX_TMAG or set it to None in config.py."
        )

    if C.RANDOM_SAMPLE:
        df_star = df_star.sample(n=C.N_STARS, random_state=C.RANDOM_SEED).sort_values("st_tmag")
    else:
        df_star = df_star.head(C.N_STARS)

    df_star.to_csv(C.TOI_TABLE_CSV, index=False)
    pd.DataFrame({"tic_id": df_star["tid"].astype(int)}).to_csv(C.TIC_LIST_CSV, index=False)

    print("Saved:", C.TOI_TABLE_CSV)
    print("Saved:", C.TIC_LIST_CSV)
    print("TIC IDs:", df_star["tid"].astype(int).tolist())


if __name__ == "__main__":
    main()
