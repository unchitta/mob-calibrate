# ACS data

Place raw ACS Detailed Table extracts here. The example pipeline reads two tables per CBSA:

- **B19001** — Household Income in the Past 12 Months (used for the income
  axis of the calibration target)
- **B01001** — Sex by Age (used for the age axis after age grouping)

## Expected layout

Files are organized by ACS release and CBSA code:

```
acs/
└── ACS5Y2020/
    └── {CBSA_CODE}/
        ├── ACSDT5Y2020.B01001-Data.csv
        ├── ACSDT5Y2020.B01001-Column-Metadata.csv
        ├── ACSDT5Y2020.B01001-Table-Notes.txt
        ├── ACSDT5Y2020.B19001-Data.csv
        ├── ACSDT5Y2020.B19001-Column-Metadata.csv
        └── ACSDT5Y2020.B19001-Table-Notes.txt
```

Census table downloads include all-CBG estimates plus a final summary row containing the CBSA-level totals. `examples/helpers/acs.py` uses that final row to derive CBSA marginals automatically.

## How to download from data.census.gov

1. Go to [data.census.gov](https://data.census.gov).
2. Search for the table name (`B19001` or `B01001`).
3. Select the table from the search results.
4. Apply Geography filters:
   - Select **Metropolitan/Micropolitan Statistical Area** and choose your
     target CBSA, **and**
   - Select **Block Group → Within Other Geographies → Metropolitan/
     Micropolitan Statistical Area → All Block Groups within [your CBSA]**.
5. Choose the appropriate ACS release. The bundled examples use **2020 ACS
   5-Year Estimates Detailed Tables**.
6. Download the CSVs and place them under
   `acs/ACS5Y2020/{CBSA_CODE}/`.

## Where this data is used

- [../../quickstart.ipynb](../../quickstart.ipynb) — Section 1.
- [../../walkthrough_02_process_acs.ipynb](../../walkthrough_02_process_acs.ipynb)
  — step-by-step version with diagnostics.

The entry point in code is `examples.helpers.acs.process_cbsa(...)`, which
returns a per-CBG distribution table plus separate income and age
marginal DataFrames. Outputs are written to
[../processed/](../processed/) in the case of walkthrough notebooks (or used downstream without saving in the case of `quickstart.ipynb`).