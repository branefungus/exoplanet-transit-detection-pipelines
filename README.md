# Comparing Exoplanet Transit Detection Pipelines Under Realistic Noise and Data Gaps

An injection-and-recovery testbed for comparing exoplanet transit-detection
pipelines under realistic noise and data gaps. Transits with known parameters
are injected into real TESS light curves, five detrending-and-search pipelines
attempt to recover them, and the recovery outcomes are compared with paired,
per-bin, and per-star statistics.

This repository contains the full code accompanying the paper. It reproduces the
star sample, the injected trials, the pipeline runs, and every table and figure.

## What it does

For each trial, a limb-darkened transit is injected into a real TESS light
curve, then corrupted with white noise, correlated (red / OU) noise, and random
data gaps. The experiment crosses two factors:

- **noise**: `low` / `high`
- **gaps**: `minimal` / `severe`

giving a 2x2 grid of observing conditions, with many trials per cell per star.
Every trial is graded on whether the recovered period, epoch (t0), and depth all
land within fixed tolerances.

## Pipelines

All five share the same BLS search core and limb-darkened depth refit; they
differ in how they remove stellar and instrumental trends before the search.

| id | detrending approach |
|----|---------------------|
| P0 | rolling-median subtraction (baseline) |
| P1 | ridge feature-regression detrend |
| P2 | masked long trend + optional wavelet denoise |
| P3 | Gaussian-process detrend (celerite2 Matern-3/2) on out-of-transit data |
| P4 | BLS-seeded joint ridge fit of trend + limb-darkened template, BIC selection |

P0 is the pre-specified baseline; P1-P4 are the challengers.

## Repository layout

```
config.py                     # single source of truth for all constants
regrade_depth_window.py       # re-grade depth outcomes after changing the depth window
scripts/                      # numbered, run in order (see below)
src/
  common/                     # shared helpers (injection, detrend, scoring, io, template)
  pipelines/                  # the five pipelines + the registry
data/                         # raw light curves + generated trials (gitignored)
results/                      # tables + figures (gitignored)
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the benchmark

Run from the repository root, in order. Steps 00 and 01 require network access
(NASA Exoplanet Archive and MAST).

```bash
# 1. Build the star sample and download one light curve per star
python scripts/00_build_tic_list.py
python scripts/01_download_lightcurves.py

# 2. Generate the injection-recovery trials (and the injection-free null set)
python scripts/02_generate_trials.py
python scripts/02_generate_trials.py --null

# 3. Run the pipelines over every trial (real and null)
python scripts/03_run_pipeline.py --pipeline all
python scripts/03_run_pipeline.py --pipeline all --null

# 4. Statistics and figures
python scripts/04_paired_stats.py
python scripts/05_reliability_maps.py --baseline P0 --challenger P4
python scripts/08_per_star_robustness.py
python scripts/09_null_analysis.py
python scripts/10_holm_correction.py
python scripts/11_tolerance_sensitivity.py
```

Outputs are written under `results/tables/` (CSVs and text reports) and
`results/figures/`.

### Other scripts

- `scripts/06_visual_diagnostics.py --trial-id <id>` — run all five pipelines on
  one trial and save time-series, phase-fold, zoom, and comparison plots.
- `scripts/07_mast_links.py` — export direct MAST download links for the exact
  products used, for a data-availability section.
- `scripts/09_null_analysis_with_inset.py` — the null-analysis figure with a
  zoomed low-false-positive panel.
- `scripts/fig_condition_grid.py`, `scripts/fig_injection_cascade.py`,
  `scripts/fig_limb_darkening.py` — explanatory (methods) figures.
- `regrade_depth_window.py` — after editing `DEPTH_FACTOR_LOW/HIGH` in
  `config.py`, re-grade the depth outcomes from the stored continuous values
  without re-running any pipeline.

## Configuration

Every constant lives in `config.py`: the star sample, noise and gap levels,
injection parameters, pipeline search settings, and the recovery tolerances.
Nothing else defines constants, so a run is fully described by that one file.

## Reproducibility

Trial generation is seeded (`BASE_SEED` in `config.py`), with a per-star,
per-condition, per-trial seed stream, so the trial set and the null set are
regenerated deterministically. The null trials use an offset seed stream so they
are statistically independent of the real trials.

## Notes on the benchmark design

- Pipelines receive the true injected duration and impact parameter from the
  trial index. This is a deliberate, disclosed choice that isolates period / t0
  / depth recovery and treats duration as known.
- The injection model (a numerically integrated limb-darkened occultation) is
  deliberately not identical in construction to the recovery-side template (a
  small-planet chord model with a fixed reference radius ratio), so the benchmark
  is not a perfect matched-filter test.

## Citation

If you use this code, please cite the paper: "Comparing Exoplanet Transit
Detection Pipelines Under Realistic Noise and Data Gaps." (Add full author list,
venue, and year here.)

## License

See `LICENSE`.
