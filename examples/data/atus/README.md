# ATUS data

Place raw ATUS (American Time Use Survey) data files here. The example
pipeline uses three file types from the same multi-year release:

- **`atusresp_*.dat`** — respondent-level summary file (one row per
  respondent; includes survey weights, demographics, year).
- **`atuscps_*.dat`** — CPS (Current Population Survey) supplement file
  (additional demographic variables, including geography).
- **`atusact_*.dat`** — activity (diary) file (one row per activity episode per respondent, with start time, duration, activity code, and `TEWHERE` location code).

## Expected layout

The helpers in `examples/helpers/atus.py` expect the following directory layout:

```
atus/
├── atusresp-0324/
│   └── atusresp_0324.dat
├── atuscps-0324/
│   └── atuscps_0324.dat
└── atusact-0324/
    └── atusact_0324.dat
```

The suffix `0324` indicates multiyear dataset (2003-2024); substitute the release you download as necessary. Filenames inside each folder follow
the BLS naming convention (`atus{type}_{release}.dat`).

## How to download

1. Visit the BLS ATUS Data Files page, e.g. 
   [https://www.bls.gov/tus/data/datafiles-0324.htm](https://www.bls.gov/tus/data/datafiles-0324.htm)
   (substitute the release suffix you want).
2. Download the multi-year **respondent**, **CPS**, and **activity** files. They're distributed as zip archives containing `.dat` files.
1. Unzip into the layout above. No API key or registration required.

## Where this data is used

- [../../quickstart.ipynb](../../quickstart.ipynb), Section 2.
- [../../walkthrough_03_process_atus.ipynb](../../walkthrough_03_process_atus.ipynb), step-by-step version with cluster diagnostics and tempograms.

Helper functions to go with:

- `examples.helpers.atus.load(resp_file, cps_file, act_file, tewhere_map, ...)`
  — loads and joins the three files, recodes demographics, maps `tewhere`
  codes to activity states.
- `examples.helpers.atus.stratify(resp, income_margin, age_margin)` —
  re-codes respondents onto ACS-derived income and age groups.
- `examples.helpers.atus.build_sequences(diaries, T=30)` — converts
  activity diaries to T-minute sequences (default 30-minute, i.e. 48
  intervals per day).
- `examples.helpers.atus.cluster_sequences(...)` — runs weighted k-medoids
  with restarts; returns `atus_meta` (metadata + cluster labels) and
  `medoid_info` (medoid indices + per-cluster distance thresholds).

Outputs are written to [../processed/](../processed/) in the case of walkthrough notebooks (or used downstream without saving in the case of `quickstart.ipynb`).