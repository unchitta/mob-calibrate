# mobcalibrate

Calibration tool for mobile phone location data

## Usage

First, prep the necessary data (see the pipeline notebooks in `examples/`).

Then, create a Calibrator instance passing in the necessary configs:
```
calibrator = Calibrator(
    home_cbgs = home_cbgs,
    assigned_cluster_labels = assigned_labels,
    acs_cbg_probs_df = acs_cbg_probs,
    acs_row_var_name = row_name,
    acs_col_var_name = col_name,
    acs_row_cats = row_cats,
    acs_col_cats = col_cats,
    acs_row_margin = row_margin,
    acs_col_margin = col_margin,
    atus_target_table = atus_target_P,
    target_pop_tot = target_pop_tot,
    geoid_col = geoid_col,
    num_replicates = num_replicates,
    seed = seed
)
```
Note: By setting a seed number, you can make sure you get the same sampled demographic characteristics and main and replicate weights every time.


Then, run the following the create main weights and replicate weights:
```
calibrator.create_main_weights()
calibrator.create_replicate_weights()
```

To get main weights (from both stages) as well as sampled demographic codes, run:
```
calibrator.get_main_weights(return_df=True)
```
(There's option to specify whether a pandas DataFrame or a numpy array is returned)

Use `get_replicate_weights()` to get a 3D numpy array where the first dimension indexes the replicates.
E.g. `calibrator.get_replicate_weights(return_df=False)[0]` returns a numpy array 
containing weights data from the first replicate after main weights (i.e. replicate_id = 1)


To get all weights (main + replicates), use:
```
calibrator.get_all_weights(return_df=False)
```
If `return_df = True`, returns a list of dataframes.

