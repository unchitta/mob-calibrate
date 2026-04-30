# Mobility data

Place your mobility user-day sequence file here if needed.

## Expected schema

A single parquet file with one row per **user-day**. Example columns:

- `user_id` (or equivalent unique identifier per user or user-day) :  string or int.
- `GEOID` : string identifier for CBG code for the user's home location. This is what gets links to the ACS table.
- `interval0`, `interval1`, …, `interval{T-1}` : one column per time interval in the sequence, where `T = (24 * 60) / interval_minutes`. Each cell holds the activity-state label for that interval (e.g. `"home"`, `"work"`, `"other"`, or whatever label set you use). The example pipeline uses 30-minute intervals (`T = 48`).

The activity-state vocabulary must match or be mapped to the labels
used in ATUS sequence construction (see example mapping in the [../../walkthrough_03_process_atus.ipynb](../../walkthrough_03_process_atus.ipynb)).

## Cacheing mobility-ATUS distance matrix

You may also cache the mobility-ATUS distance matrix here or under
[../processed/](../processed/) to avoid recomputing on every run. This is typically the most expensive part of the process. The quickstart notebook loads a precomputed parquet by default.

## Where this data is used

- [../../quickstart.ipynb](../../quickstart.ipynb) : Section 3 (distance
  matrix) and Section 4 (cluster assignment).

Helper functions to use:

- `examples.helpers.mobility.distance_to_atus(mob_seq, atus_metrics, ...)`
  — computes per-user-day metrics, aligns the mobility and ATUS metric
  spaces, and returns a cosine-distance DataFrame.
- `examples.helpers.prep_calibration_inputs.assign_mobility_clusters(...)`
  — assigns each mobility user-day to an ATUS behavioral cluster via k-NN
  voting, with optional medoid-distance unassignment.

## Notes

- This directory is **not** populated by the repository. You must supply
  your own data subject to whatever licensing/privacy terms you have
- We exclude contents of this folder from version control using .gitignore