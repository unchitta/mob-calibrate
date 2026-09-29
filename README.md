# mobcalibrate

Calibration of mobile phone GPS-derived mobility samples to external demographic and behavioral benchmarks/targets. 
The point is to introduce a framework for mobility data calibration that draws largely from traditional survey weighting methods, but adapt them to suit the format and needs of mobility data.
 `mobcalibrate` produces person weights along with bootstrap replicate weights for variance estimation that correct for certain biases by raking the sample to:

1. **Demographic** marginals from the U.S. Census ACS (e.g. age × income at the
   block-group level), and
2. **Behavioral** targets from the BLS American Time Use Survey (ATUS),
   matching the conditional distribution of behavioral cluster types within
   each demographic stratum.

This method assumes that the mobility data does not come with demographic labels due to privacy restrictions, hence demographics need to be sampled. To capture this sampling uncertainty, the framework is designed with replicate weights in mind. Each individual will have not just a single weights but an array of weights to be used in variance estimation.

The package efficiently utilizes `numpy` and `pandas` and inputs/outputs are plain `numpy` arrays and `pandas` DataFrames. This package provides a main API (`Calibrator`) as well as various processing helper APIs. The example pipeline shows how to turn raw
ACS/ATUS files and mobility sequences into inputs that can be fed into the Calibrator (see [examples/](examples/)).

**Note:** for mobility samples in other, non-U.S. countries, please follow the example notebooks and replace the ACS/ATUS data with appropriate corresponding data sources. Many of the preprocessing helpers are written with ACS/ATUS in mind, but please feel free to adapt them for your particular use cases.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Reference

Please refer to our paper for more details about the method and for citation:

Girardini, N. A., Kan, U., López, E., Lepri, B., Lucchini, L., & Centellegher, S. (2026).
*Behavioral calibration of mobile-phone GPS data for population-representative analyses*.
arXiv:2609.01042. [https://arxiv.org/abs/2609.01042](https://arxiv.org/abs/2609.01042)

```bibtex
@misc{girardini2026behavioral,
  title         = {Behavioral calibration of mobile-phone GPS data for population-representative analyses},
  author        = {Girardini, Nicol{\`o} Alessandro and Kan, Unchitta and L{\'o}pez, Eduardo and Lepri, Bruno and Lucchini, Lorenzo and Centellegher, Simone},
  year          = {2026},
  eprint        = {2609.01042},
  archivePrefix = {arXiv},
  primaryClass  = {physics.soc-ph},
  doi           = {10.48550/arXiv.2609.01042},
  url           = {https://arxiv.org/abs/2609.01042}
}
```

## Questions?

Please email Unchitta Kan at ukanjana@gmu.edu.


## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/unchitta/mob-calibrate.git"

# with the example dependencies (+ ipykernel):
pip install "mobcalibrate[examples] @ git+https://github.com/unchitta/mob-calibrate.git"

# with the example dependencies (+ ipykernel *AND* jupyter):
pip install "mobcalibrate[examples-with-jupyter] @ git+https://github.com/unchitta/mob-calibrate.git"

# or from a clone, for development / latest main
git clone https://github.com/unchitta/mob-calibrate.git
cd mob-calibrate
pip install -e .
```

Requires pip 21.2 or later. Install time < 1 minute

Requires Python ≥ 3.9. The package's runtime dependencies are `numpy`,
`pandas`, `scikit-learn`, and `tqdm` (see [pyproject.toml](pyproject.toml)).

The example pipeline under [examples/](examples/) additionally uses
`scipy`, `matplotlib`, `seaborn`, `fastparquet` (for parquet files; `pyarrow`
also works and is used automatically if installed), and `ipykernel` (to run
the notebooks in Jupyter or VS Code).

**Install them with the `examples` or `examples-with-jupyter` extra when you intend to run the notebooks**:

```bash
pip install -e ".[examples]"
```

```bash
pip install -e ".[examples-with-jupyter]"
```

Tested with Python 3.11, with numpy 1.24–2.4, pandas 2.3–3.0,
scikit-learn 1.2–1.9, tqdm 4.65–4.70, scipy 1.10–1.17, matplotlib 3.7–3.11,
seaborn 0.12–0.13, and pyarrow 22–25 or fastparquet 2026.9 for parquet files.

Tested on: Ubuntu Server 18.04 and MacOs Tahoe 26.6.2.

No non-standard hardware is required.


## Demo run

We've included synthetic and public data necessary to run the pipeline from end to end in `examples/`. **After installation, use `examples/quickstart_synthetic.ipynb` to demo the package.** Optionally, explore other notebooks for a better understanding of the pipeline.

**Note:** the raw ATUS data files are not included in the repository due to their size. Download them before running the demo by following the instructions in [examples/data/atus/README.md](examples/data/atus/README.md).

Runtime for the demo on a standard laptop should take no longer than 3-5 minutes (and may be much faster). Runtime on actual data will depend on the size of your mobility data set.


## Method overview

`mobcalibrate` runs a two-stage calibration:

```
                 ┌──────────────── per CBG joint priors ────────────────┐
home CBG ──────► │  Stage 1: joint IPF on ACS marginals                 │ ──► weight1
cluster label    │  Stage 2: rake within each joint stratum             │ ──► weight_final
                 │           to ATUS-derived P(cluster | stratum)       │
                 └──────────────────────────────────────────────────────┘
```

- **Stage 1 (`stage1_ipf`).** For each replicate, demographic codes are sampled
  per unit from per-CBG joint distributions, then iterative proportional
  fitting rakes initial uniform weights to the two ACS marginals (e.g. an age
  vector and an income vector for the CBSA). Each replicate will have different Stage 1 weights due to random sampling. These weights are propagated to the second stage in each replicate, also resulting in different final weights across replicates.
- **Stage 2 (`stage2_rake`).** Within each demographic stratum, weights from the previous stage are raked again so the weighted distribution of behavioral cluster labels
  matches an ATUS-derived target table (`P(cluster | stratum)`). Units with
  cluster label `-1` are treated as unassigned and excluded from Stage 2. Their weights are then transferred to other individuals within the same cluster and stratum at the end (similar to non-response adjustment in traditional survey weighting).
- **Replicates.** `num_replicates` independent weight sets are produced from
  spawned RNG streams. For example, if `num_replicates` is set to 50, then each individual in the sample will have 50 values of weights to account for the demographic sampling uncertainty. These replicate weights should be used for variance estimation when presenting mobility estimates.


## Inputs you'll need (assuming U.S.-based panels)

The `Calibrator` API takes specific input formats. The simplest way is to start from the raw data use the example helpers to derive these calibrator inputs.

### Raw data

| Source | What | Where it goes |
|---|---|---|
| ACS 5-Year (Census) | `B19001` (household income) and `B01001` (sex by age) tables, downloaded per CBSA at block-group and CBSA resolutions | [examples/data/acs/](examples/data/acs/) |
| ATUS (BLS) | Respondent (`atusresp_*.dat`), CPS (`atuscps_*.dat`), and activity (`atusact_*.dat`) files downloaded from BLS | [examples/data/atus/](examples/data/atus/) |
| Mobility (your data) | Mobility sequences (e.g. as parquet for efficiency), with one column per time interval and a `GEOID` column for home CBG | [examples/data/mobility/](examples/data/mobility/) |

Each subfolder under `examples/data/` has a README with download instructions
and the expected file layout.

### Calibrator inputs

Once preprocessed, `Calibrator` consumes:

- `home_cbgs` — `(N,)` array of GEOID strings, one per unit (individual or
  individual-day).
- `assigned_cluster_labels` — `(N,)` int array, with `-1` for unassigned units
  and `0..K-1` for assigned ones.
- `acs_cbg_probs_df` — `pandas.DataFrame` indexed/keyed by GEOID, with one
  column per row category and one column per column category.
- `acs_row_margin`, `acs_col_margin` — 1-D arrays summing to 1 (target
  marginals at the CBSA level).
- `acs_row_cats`, `acs_col_cats` — category labels matching the columns of
  `acs_cbg_probs_df`.
- `atus_target_table` — `(G, K)` row-stochastic array, where `G = num_row_cats`
  * `num_col_cats` is the number of joint demographic strata and `K` is the
  number of behavioral clusters.
- `target_pop_tot` — total population to scale weights to.

The helpers in `examples/helpers/` module provide help for the preprocessing steps, e.g. 
[`prep_acs_targets`](examples/helpers/prep_calibration_inputs.py),
[`prep_atus_target`](examples/helpers/prep_calibration_inputs.py),
[`assign_mobility_clusters`](examples/helpers/prep_calibration_inputs.py),
[`filter_valid_users`](examples/helpers/prep_calibration_inputs.py).



## Quickstart


### 1. Preprocessing

See example of the full executable version [examples/quickstart.ipynb](examples/quickstart.ipynb). 
The flow looks like this:

```python
from examples.helpers import acs, atus, mobility
from examples.helpers import prep_calibration_inputs as prep

# ACS: per-CBG joint distribution + CBSA-level marginals
cbg_distr, income_margin, age_margin = acs.process_cbsa(...)

# ATUS: respondent metadata + behavioral cluster assignments
resp = atus.load(...)
resp = atus.stratify(resp, income_margin, age_margin)
diaries, sequences = atus.build_sequences(...)
atus_meta, medoid_info = atus.cluster_sequences(sequences, ...)

# Mobility-ATUS distance matrix and cluster assignment
dist_df = mobility.distance_to_atus(mob_seq, atus_meta, ...)
cluster_labels = prep.assign_mobility_clusters(dist_df, atus_meta, ...)

# Filter to users with valid home CBG and build Calibrator inputs
user_ids, home_cbgs, cluster_labels, _ = prep.filter_valid_users(
    users_df, cbg_distr, cluster_labels
)
acs_targets = prep.prep_acs_targets(income_margin, age_margin, "income", "age")
atus_target_P = prep.prep_atus_target(atus_meta, "income", "age",
                                      cluster_label_col="cluster", weight_col="weight",
                                      **{k: acs_targets[k] for k in ("num_row_cats","num_col_cats")},
                                      num_clusters=K)
```

The helpers under `examples/helpers/` are intentionally outside the package —
they encode assumptions about ACS/ATUS file layouts and mobility-vendor schemas
that vary between projects. Treat them as a reference implementation and fork as necessary.

### 2. Calibrate

```python
from mobcalibrate import Calibrator

calibrator = Calibrator(
    unit_ids=user_ids,
    home_cbgs=home_cbgs,
    assigned_cluster_labels=cluster_labels,
    acs_cbg_probs_df=cbg_distr,
    acs_row_var_name="income", 
    acs_col_var_name="age",
    acs_row_cats=acs_targets["row_cats"], 
    acs_col_cats=acs_targets["col_cats"],
    acs_row_margin=acs_targets["row_margin"], 
    acs_col_margin=acs_targets["col_margin"],
    atus_target_table=atus_target_P,
    target_pop_tot=acs_targets["target_pop_tot"],
    num_replicates=51,
    seed=42,
)

result = calibrator.create_weights(mode="demographic_behavioral")
```

Setting `seed` makes the sampled demographic codes and all replicate weights
fully reproducible. `mode="demographic_behavioral"` (default) will run the full two-stage calibration as published. However, you may also select these following modes:

- `uniform` — assigns equal weights `1/N`; useful as a baseline.
- `demographic` — Stage 1 only; use this if you don't have ATUS targets (`weights_final = weight1`)
- `behavioral` — Stage 2 only; rakes against ATUS targets using uniform weights as Stage 1 weight



## Working with the output

`create_weights` returns an immutable `CalibrationResult` which provides the following methods to access the data:

| Method | Returns |
|---|---|
| `result.all_final_weights(to_df=True)` | `(N, R)` np.array or DataFrame of `weight_final` across all replicates; column `weight_final_0` is weight estimate from the first replicate, etc. |
| `result.main_weights()` | `(N, 2)` array — `[weight1, weight_final]` for the main replicate. |
| `result.replicate_weights()` | `(R-1, N, 2)` array — bootstrap replicates only. |
| `result.to_df(replicate_id=0)` | Wide DataFrame with weights from a specific replicate along with sampled demographic codes and metadata. |
| `result.to_long_df()` | All replicate weights and metadata in long-form DataFrame, indexed by `replicate_id`. |
| `result.save(path)` | Save the CalibrationResult object as .pkl |
| `CalibrationResult.load(path)` | load CalibrationResult instance from .pkl with type check |

All weights are scaled to `target_pop_tot` and stored as `int64`.



## Quickstart and walkthrough notebooks

Start-to-finish examples (change paths and parameters in the config cells and run):

- **[examples/quickstart_synthetic.ipynb](examples/quickstart_synthetic.ipynb)** : demo on the
  included synthetic mobility data. Computes the mobility-ATUS distance matrix in the notebook
  and ends with an example time-use comparison (uncalibrated vs calibrated vs ATUS).
- **[examples/quickstart.ipynb](examples/quickstart.ipynb)** : the same pipeline for real
  mobility data. Since real mobility sequences cannot be shared, it loads a precomputed
  mobility-ATUS distance matrix (or computes one from your own sequences).

For inspecting and learning more about the individual stages:

- [examples/walkthrough_01_process_mobility.ipynb](examples/walkthrough_01_process_mobility.ipynb)
  : expected format of processed mobility sequences (placeholder; no processing code due to data agreement).
- [examples/walkthrough_02_process_acs.ipynb](examples/walkthrough_02_process_acs.ipynb)
  : ACS table prep with intermediate diagnostics.
- [examples/walkthrough_03_process_atus.ipynb](examples/walkthrough_03_process_atus.ipynb)
  : ATUS sequence building, k-medoids clustering, tempograms.
- [examples/walkthrough_04_distance_matrix.ipynb](examples/walkthrough_04_distance_matrix.ipynb)
  : sequence metrics, normalization, and the mobility-ATUS cosine distance matrix.
- [examples/walkthrough_05_run_calibration.ipynb](examples/walkthrough_05_run_calibration.ipynb)
  : calibration step with extra diagnostics.

See [examples/README.md](examples/README.md) for an index.

Also, please remember to place your data files into the examples/data/ directories as appropriate.



## API reference


### `mobcalibrate.Calibrator`

```python
Calibrator(
    unit_ids, 
    home_cbgs, 
    assigned_cluster_labels,
    acs_cbg_probs_df, 
    acs_row_var_name, acs_col_var_name,
    acs_row_cats, acs_col_cats, 
    acs_row_margin, acs_col_margin,
    target_pop_tot,
    atus_target_table=None,
    geoid_col="GEOID",
    num_replicates=161,
    seed=None,
)
```

`Calibrator.create_weights() -> CalibrationResult` 

### `mobcalibrate.CalibrationResult`

immutable dataclass returned by `create_weights()`. See
[Working with the output](#working-with-the-output).

### `mobcalibrate.preprocessing`

Helpers used by the example pipeline; available for direct use if you have
your need to write your own data processing.

- **ATUS sequence metrics**
  - `sequence_metrics(seq, all_labels)` : extract behavioral metrics from a
    single daily sequence: number of distinct activities, turnover rate,
    reciprocity, activity entropy (natural log), time share per label in
    `all_labels`, and transition counts for every ordered pair of labels
    (`edge_(a, b)`; all `len(all_labels)²` pairs are always present, zero if
    unobserved). (To be used in distance matrix calculations)
  - `compute_metrics_for_all_sequences(sequences, all_labels)` : apply over
    many sequences; returns one row per sequence.
  - `normalize_sequence_metrics(metrics_df, all_labels)` : optional, separate
    rescaling step so features are on comparable [0, 1] scales before cosine
    distance: `num_activities / len(all_labels)`, `entropy / log(len(all_labels))`,
    and edge counts divided by their row total. Apply to both
    mobility-derived metrics and ATUS-derived metrics or neither.
  - `metrics_cosine_D(metrics_df1, metrics_df2)` : pairwise cosine-distance
    matrix between two metric tables.
- **Weighted k-medoids clustering**
  - `weighted_kmedoids(D, k, w=None, ...)` : weighted PAM on a precomputed
    distance matrix.
  - `fit_weighted_kmedoids_with_restarts(D, wA, K, n_restarts=10, ...)` :
    weighted PAM version that restarts multiple times and keeps the best solution.
- **Cluster assignment for new units**
  - `compute_dist_to_medoid_thresholds(D, medoids, labels, percentile=99.0)`
    : per-cluster distance thresholds.
  - `knn_cluster_label(k, dist_matrix, atus_cluster_labels, threshold, ...)`
    : k-NN voting with a confidence threshold and optional medoid distance
    cap; returns `-1` for unassigned units.
- **ATUS target-table construction**
  - `compute_atus_target_table(atus_df, ...)` : weighted `P(cluster |
    stratum)` table for Stage 2.
- **Validation**
  - `get_valid_mask_by_acs_geoid(users_df, acs_cbg_df, geoid_col="GEOID")` :
    boolean mask of users whose home CBG appears in the ACS table.



## Repository layout

```
mobcalibrate/
├── src/mobcalibrate/
│   ├── __init__.py            # exports Calibrator, CalibrationResult
│   ├── calibration.py         # Calibrator + CalibrationResult
│   ├── preprocessing.py       # public preprocessing helpers
│   ├── core.py                # internal: CDF sampling, IPF, raking
│   └── utils.py               # internal: small array utilities
├── examples/
│   ├── quickstart.ipynb            # start to finish example
│   ├── quickstart_synthetic.ipynb  # demo on synthetic mobility data
│   ├── walkthrough_*.ipynb         # walkthroughs for individual processing stages
│   ├── helpers/                    # ACS/ATUS/mobility data-prep helpers
│   └── data/                       # raw + processed + results dirs
└── pyproject.toml
```
