import numpy as np
import pandas as pd

from mobcalibrate.preprocessing import (
    compute_metrics_for_all_sequences,
    metrics_cosine_D,
)
from .atus import select_feature_subset



def distance_to_atus(mob_seq, atus_metrics, sequence_metric_specs,
                     sequence_cols, geoid_col='GEOID'):
    """
    Build cosine distance matrix between mobility sequences and ATUS respondents.

    Since the metrics for mobility and ATUS need to match, the same
    sequence_metric_specs that was used to process ATUS should be passed
    to this function. (compute_metrics_for_all_sequences is called here)

    This function then streamlines computing mobility metrics, aligns the metric
    columns with atus_metrics, and calculate distance matrix using 
    the same feature subset in both datasets.
    
    Edge columns observed in one dataset but not the other are padded with 0
    so both sides share the same feature space.

    Parameters
    ----------
    mob_seq : pd.DataFrame
        User-day mobility sequences. Must contain `sequence_cols` (one cell per
        T-min interval) and a `geoid_col` column for the user's home CBG.
    atus_metrics : pd.DataFrame
        Full feature table returned by `atus.cluster_sequences`.
    sequence_metric_specs : dict
        dict with keys 'home_label', 'work_label', 'all_labels', 'feature_subset'. 
        Must be the same dict passed to `atus.cluster_sequences`.
    sequence_cols : list[str]
        Column names in mob_seq that contain the activity cells.
    geoid_col : str

    Returns
    -------
    dist_df : pd.DataFrame
        Shape (N_mobility_days, N_atus_days). Index follows mob_seq's index;
        columns follow atus_metrics's index. Carries a `geoid_col` column with
        each mobility user-day's home CBG.
    """


    home_label = sequence_metric_specs['home_label']
    work_label = sequence_metric_specs['work_label']
    all_labels = sequence_metric_specs['all_labels']
    feature_subset = sequence_metric_specs.get('feature_subset', 'all')

    # compute features on the mobility side using the same spec as ATUS
    mob_metrics = compute_metrics_for_all_sequences(
        mob_seq[sequence_cols].values,
        home_label=home_label,
        work_label=work_label,
        all_labels=all_labels,
    )
    mob_metrics.index = mob_seq.index

    # align the two feature spaces: pad missing columns with 0, use atus column
    # order as canonical and append any mobility-only columns at the end
    canonical_order = (
        list(atus_metrics.columns)
        + [c for c in mob_metrics.columns if c not in atus_metrics.columns]
    )
    n_added_to_atus = sum(1 for c in canonical_order if c not in atus_metrics.columns)
    n_added_to_mob = sum(1 for c in canonical_order if c not in mob_metrics.columns)

    atus_aligned = atus_metrics.reindex(columns=canonical_order, fill_value=0)
    mob_aligned = mob_metrics.reindex(columns=canonical_order, fill_value=0)

    print(
        f'aligned feature spaces: '
        f'padded {n_added_to_atus} col(s) into ATUS, {n_added_to_mob} into mobility; '
        f"active feature_subset={feature_subset!r}"
    )

    # apply feature_subset consistently to both sides
    atus_subset = select_feature_subset(atus_aligned, feature_subset, all_labels)
    mob_subset = select_feature_subset(mob_aligned, feature_subset, all_labels)

    # cosine distance matrix
    dist_matrix = metrics_cosine_D(mob_subset, atus_subset)

    dist_df = pd.DataFrame(
        dist_matrix,
        index=mob_seq.index,
        columns=atus_metrics.index,
    )
    dist_df[geoid_col] = mob_seq[geoid_col].values
    return dist_df