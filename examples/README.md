# Examples

Worked examples for using `mobcalibrate` end-to-end on real ACS, ATUS, and
mobility data.

## Where to start

- **[quickstart_synthetic.ipynb](quickstart_synthetic.ipynb)** — the demo.
  Runs the whole pipeline on the included synthetic mobility data
  ([data/mobility/synthetic_mobility_38060.csv](data/mobility/synthetic_mobility_38060.csv))
  and the included ACS tables, and ends with a time-use comparison plot
  (uncalibrated vs calibrated vs ATUS). Only the ATUS files need downloading.
- **[quickstart.ipynb](quickstart.ipynb)** — the production path for real
  mobility data. A single notebook that takes a CBSA code and runs ACS prep →
  ATUS prep → mobility distance / cluster assignment → calibration → weight
  export. It loads a precomputed mobility-ATUS distance matrix, or computes
  one from your own sequences in Section 3.

For inspecting individual stages with extra diagnostics:

- [walkthrough_01_process_mobility.ipynb](walkthrough_01_process_mobility.ipynb) —
  placeholder showing the expected format of processed mobility sequences
  (rows = user-days, one column per T-minute cell, plus `GEOID`), matching
  the synthetic data file. No processing code is included due to data agreement.
- [walkthrough_02_process_acs.ipynb](walkthrough_02_process_acs.ipynb) —
  ACS B19001 / B01001 table processing. Corresponds to Section 1 of the
  quickstart.
- [walkthrough_03_process_atus.ipynb](walkthrough_03_process_atus.ipynb) —
  ATUS respondent loading, sequence construction, weighted k-medoids
  clustering, tempogram visualization. Corresponds to Section 2 of the
  quickstart.
- [walkthrough_04_distance_matrix.ipynb](walkthrough_04_distance_matrix.ipynb) —
  sequence metrics for ATUS and mobility, normalization, and the cosine
  distance matrix. Corresponds to Section 3 of the quickstart.
- [walkthrough_05_run_calibration.ipynb](walkthrough_05_run_calibration.ipynb)
  — calibration step run directly on saved processed inputs. Corresponds to
  Sections 4–6 of the quickstart.

## Setup before running

You need to populate [data/](data/) with raw files first. See the per-folder
READMEs for download instructions and expected file layout:

- [data/acs/README.md](data/acs/README.md) — ACS B19001 / B01001 from
  data.census.gov.
- [data/atus/README.md](data/atus/README.md) — ATUS respondent / CPS /
  activity files from BLS. **The ATUS files are not included in the
  repository due to their size. Download them following this README.**
- [data/mobility/README.md](data/mobility/README.md) — includes the synthetic
  demo data (`synthetic_mobility_38060.csv`); place your own per-user-day
  mobility sequences here for real runs.
- [data/processed/README.md](data/processed/README.md) — intermediate outputs
  (auto-populated by the notebooks).
- [data/results/README.md](data/results/README.md) — final calibration outputs
  (auto-populated by the notebooks).

In addition to the package's own dependencies, the notebooks need the
`examples` extra (run from the repository root):

```bash
pip install -e ".[examples]"
```

## What's in `helpers/`

The [helpers/](helpers/) directory contains thin wrappers around
`mobcalibrate.preprocessing` plus dataset-specific I/O code that's outside
the scope of the package itself. Treat these as a reference implementation —
fork them when adapting to a different mobility-data schema or Census release.

| Module | Role |
|---|---|
| [helpers/acs.py](helpers/acs.py) | Loads raw ACS tables, derives income quartile mappings from CBSA totals, aggregates the detailed sex-by-age table to coarser groups, and merges per-CBG income × age tables into a single joint distribution. Main entry: `process_cbsa()`. |
| [helpers/atus.py](helpers/atus.py) | Loads ATUS respondent / CPS / activity files, recodes demographic variables, downsamples activity diaries to T-minute sequences, auto-derives ATUS income/age groupings to align with ACS, and clusters sequences into K behavioral types. Main entries: `load()`, `stratify()`, `build_sequences()`, `cluster_sequences()`. |
| [helpers/mobility.py](helpers/mobility.py) | Computes sequence metrics on mobility user-day sequences using the same specification as ATUS, then builds the cosine-distance matrix between mobility users and ATUS respondents. Main entry: `distance_to_atus()`. |
| [helpers/plotting.py](helpers/plotting.py) | Tempogram plots for visualizing weighted time-use distributions per behavioral cluster. |
| [helpers/prep_calibration_inputs.py](helpers/prep_calibration_inputs.py) | The last-mile bridge to `mobcalibrate.Calibrator`: ACS-target formatting, ATUS conditional cluster table, k-NN cluster assignment for mobility users, and GEOID-based filtering. |