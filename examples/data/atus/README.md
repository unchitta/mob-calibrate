# ATUS data

Place raw ATUS (American Time Use Survey) data files here. The example
pipeline uses three file types from the same multi-year release:

- **`atusresp_*.dat`** — respondent-level summary file (one row per
  respondent; includes survey weights, demographics, year).
- **`atuscps_*.dat`** — CPS (Current Population Survey) supplement file
  (additional demographic variables, including geography).
- **`atusact_*.dat`** — activity (diary) file (one row per activity episode per respondent, with start time, duration, activity code, and `TEWHERE` location code).

**These files are not included in the repository due to their size.** They
are required to run every example notebook that processes ATUS, including
the demo ([../../quickstart_synthetic.ipynb](../../quickstart_synthetic.ipynb)),
so download them first following the steps below.

## How to download

1. Download the three multi-year (2003–2024) zip archives from BLS. No API
   key or registration required:
   - Respondent file: [atusresp-0324.zip](https://www.bls.gov/tus/datafiles/atusresp-0324.zip)
   - ATUS-CPS file: [atuscps-0324.zip](https://www.bls.gov/tus/datafiles/atuscps-0324.zip)
   - Activity file: [atusact-0324.zip](https://www.bls.gov/tus/datafiles/atusact-0324.zip)

   These links are listed on the BLS ATUS multi-year data files page,
   [https://www.bls.gov/tus/data/datafiles-0324.htm](https://www.bls.gov/tus/data/datafiles-0324.htm).
   For a different release, use that release's page and substitute its suffix.
2. Unzip each archive into its own folder here, following the layout below.

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

The suffix `0324` indicates the multi-year dataset (2003–2024); substitute the release you download as necessary. Filenames inside each folder follow
the BLS naming convention (`atus{type}_{release}.dat`). The archives also contain SAS, SPSS, and Stata import programs, which can stay alongside the `.dat` files.

## Where this data is used

- [../../quickstart_synthetic.ipynb](../../quickstart_synthetic.ipynb), Section 2 (demo).
- [../../quickstart.ipynb](../../quickstart.ipynb), Section 2.
- [../../walkthrough_03_process_atus.ipynb](../../walkthrough_03_process_atus.ipynb), step-by-step version with cluster diagnostics and tempograms.

Helper functions to go with:

- `examples.helpers.atus.load(resp_file, cps_file, act_file, tewhere_map, ...)`
  — loads and joins the three files, recodes demographics, and maps `TEWHERE`
  codes to location codes. Returns `resp` (respondents) and `diaries`.
- `examples.helpers.atus.stratify(resp, income_margin, age_margin)` —
  re-codes respondents onto ACS-derived income and age groups. Returns
  `atus_meta` and the group labels.
- `examples.helpers.atus.build_sequences(diaries, T=30)` — converts
  activity diaries to T-minute sequences (default 30-minute, i.e. 48
  intervals per day).
- `examples.helpers.atus.cluster_sequences(atus_seq, weights, K, sequence_metric_specs)`
  — computes sequence metrics and runs weighted k-medoids with restarts.
  Returns the sequence metrics, the clustering results (medoids + cluster
  labels), and per-cluster distance thresholds.

Outputs are written to [../processed/](../processed/) in the case of walkthrough notebooks (or used downstream without saving in the case of the quickstart notebooks).
