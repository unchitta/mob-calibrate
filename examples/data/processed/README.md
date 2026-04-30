# Processed data

This directory holds intermediate artifacts produced by the example pipeline.
You don't need to put anything here manually; the notebooks will write to it as they run (assuming the walkthrough notebooks are run and not `quickstart.ipynb`).

Files are namespaced by CBSA code (for US-based data) so you can keep multiple regions.

## Files written here

| File | Produced by | Contents |
|---|---|---|
| `acs_cbg_distr_{CBSA}.csv` | ACS prep | CBG-level income and age probability distributions (one row per CBG) |
| `acs_income_margins_{CBSA}.csv` | ACS prep | CBSA-level income marginal (one row per income group, with population counts) |
| `acs_age_margins_{CBSA}.csv` | ACS prep | CBSA-level age marginal |
| `atus_meta_{CBSA}.csv` | ATUS prep | One row per ATUS respondent: survey weights, demographics aligned to ACS, behavioral cluster label |
| `atus_seq_{CBSA}.csv` | ATUS prep | activity sequences for all ATUS respondent (one row per respondent, one column per time interval) |
| `atus_medoid_info_{CBSA}.json` | ATUS prep | Medoid indices and 99th-percentile distance thresholds per cluster, used downstream for in assigning mobility users to ATUS clusters |
| `distance_matrix_*.parquet` | Mobility-ATUS | Cosine distance matrix between mobility user-days (rows) and ATUS diary-days (columns)|

## Notes

- These files can be large (the distance matrix in particular). Treat the directory as a regenerable cache; `.gitignore` excludes its contents from version control
- The CBSA code in filenames matches the configuration in the notebooks