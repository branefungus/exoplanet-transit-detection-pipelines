"""
Single configuration file for the whole project.

Everything reads its constants from here, and nothing else defines them. To
reproduce the run, leave these values as they are; to explore variations, edit
them here and re-run. The tolerances came over from the first version of the
project (iteration 1). The noise, gap, and injection settings are the values the
30-star generator actually used.
"""
from pathlib import Path

# Paths (relative to the repo root)
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
STARS_DIR = DATA_RAW / "stars"
TRIALS_DIR = ROOT / "data" / "trials"
TRIALS_NULL_DIR = ROOT / "data" / "trials_null"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

TIC_LIST_CSV = DATA_RAW / "tic_list.csv"
TOI_TABLE_CSV = DATA_RAW / "toi_sample.csv"
MANIFEST_CSV = STARS_DIR / "download_manifest.csv"
TRIAL_INDEX_CSV = TABLES_DIR / "trial_index.csv"
TRIAL_INDEX_NULL_CSV = TABLES_DIR / "trial_index_null.csv"


def results_csv(pipeline: str, null: bool = False) -> Path:
    suffix = "_null" if null else ""
    return TABLES_DIR / f"results_{pipeline}{suffix}.csv"


# Star sample selection (script 00)
N_STARS = 30
DISPOSITIONS = ("CP", "KP")     # confirmed planet, known planet
MAX_TMAG = 12.5                 # set to None to disable the brightness filter
RANDOM_SAMPLE = False           # False means take the brightest N
RANDOM_SEED = 42

# Light curve download (script 01)
PREFER_AUTHOR = "SPOC"
PREFER_EXPTIME_SECONDS = 120
MIN_POINTS = 2000

# Trial generation (script 02)
BASE_SEED = 123
TRIALS_PER_CELL = 20            # per (noise, gap) bin, per star
NOISE_LEVELS = ["low", "high"]
GAP_LEVELS = ["minimal", "severe"]

# Injection
PERIOD_MIN = 2.0                # days; also the pipeline search range
PERIOD_MAX = 8.0
DURATION_DAYS = 2.0 / 24.0      # total duration, first to fourth contact
DEPTH_MIN = 0.8e-3              # mid-transit depth is sampled uniformly in this range
DEPTH_MAX = 3.0e-3
T0_MAX_TRIES = 200

# Limb darkening / geometry of the injected signal
LD_U1 = 0.35
LD_U2 = 0.20
IMPACT_PARAMETER = 0.30

# Noise model. These are the values the previous run actually used (the
# generator's own internal defaults). The old config_trials2.py had mismatched
# variable names, so it was silently ignored.
WHITE_SIGMA_LOW = 2.5e-4
WHITE_SIGMA_HIGH = 7.0e-4
RED_SIGMA_LOW = 2.0e-4
RED_SIGMA_HIGH = 7.0e-4
RED_RHO_DAYS = 0.08             # OU correlation timescale (about 2 h)

# Gaps (random removed blocks). Same as the noise above: previous-run defaults.
GAP_BLOCKS_MINIMAL = 2
GAP_LEN_DAYS_MINIMAL = 0.20
GAP_BLOCKS_SEVERE = 6
GAP_LEN_DAYS_SEVERE = 0.60

# Cleaning / sanity
SIGMA_CLIP = 8.0
MIN_POINTS_AFTER_GAPS = 300
OVERWRITE_TRIAL_FILES = True

# Occultation lookup table resolution
LD_TABLE_NK = 18
LD_TABLE_ND = 1400
LD_TABLE_NR = 700

# Pipeline search settings (shared by P0-P4)
BLS_DURATIONS_DAYS = [1.0 / 24.0, 2.0 / 24.0, 3.0 / 24.0, 4.0 / 24.0]
BLS_FREQUENCY_FACTOR = 10

# Recovery-side LD template. This is deliberately not built the same way as the
# injection model: the pipelines use a small-planet chord template with a fixed
# reference radius ratio, while the generator numerically integrates the
# occultation. Keep K_REF fixed; see the README note on template circularity.
LD_K_REF = 0.05
LD_B_DEFAULT = 0.30

# P4 joint-fit settings
P4_TOPK_PERIODS = 6
P4_T0_GRID_STEPS = 13
P4_T0_GRID_HALFSPAN = 1.0       # in units of duration
P4_RIDGE_ALPHA = 1e-2

# P3 GP settings
P3_MIN_RHO_DAYS = 0.7
P3_MAX_RHO_DAYS = 30.0

# Recovery tolerances (grading). Carried over from the original iteration-1
# config.py, unchanged through the 30-star run.
PERIOD_FRAC_TOL = 0.01                   # |P_found - P_true| / P_true <= 1%
T0_ABS_TOL_DAYS = (2.0 / 24.0) * 0.5     # 1 hour, i.e. half the transit duration
DEPTH_FACTOR_LOW = 0.7                   # d_ok when depth lands in [0.7, 1.3] x true
DEPTH_FACTOR_HIGH = 1.3

# Statistics / plotting
CI_Z = 1.96                     # 95% intervals
STATS_METRICS = ["pass", "p_ok", "t_ok", "d_ok"]
