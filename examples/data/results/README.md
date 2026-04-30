# Results

This directory holds final calibration outputs. The notebooks write to it at the end so you likely won't need to put anything here manually.

## Example files written here

| File | From | Contents |
|---|---|---|
| `*.pkl` | `CalibrationResult.save(path)` | Pickled `CalibrationResult`. Full output (all replicate weights, sampled demographic codes, calibrator metadata). Reload with `CalibrationResult.load(path)` |
| `*_weights.parquet` | `result.all_final_weights(to_df=True).to_parquet(path)` | Wide table of final weights: `unit_id` plus `weight_final_0..R-1` |
| `*_long.parquet` | `result.to_long_df().to_parquet(path)` | All replicates weights and metadata in long form (index the replicate using `replicate_id`). Useful for downstream replicate-based variance estimation. |

## Notes

- Weights are stored as `int64`, scaled to `target_pop_tot`
- We exclude the contents of this folder from version control