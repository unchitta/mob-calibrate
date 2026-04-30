import numpy as np
import pandas as pd

from mobcalibrate.preprocessing import (
    knn_cluster_label,
    compute_atus_target_table,
    get_valid_mask_by_acs_geoid,
)


def prep_acs_targets(row_margin_df, col_margin_df, row_var, col_var, pop_col="pop"):
    """
    Prepare ACS marginal targets and category labels for the Calibrator.

    Parameters
    ----------
    row_margin_df : pd.DataFrame
        Marginal counts for the row stratification variable (e.g. income).
        Expected columns: [row_var, pop_col].
    col_margin_df : pd.DataFrame
        Marginal counts for the column stratification variable (e.g. age group).
        Expected columns: [col_var, pop_col].
    row_var : str
        Column name of the row variable category labels.
    col_var : str
        Column name of the column variable category labels.
    pop_col : str
        Column name of the population counts.

    Returns
    -------
    dict with keys:
        row_cats, col_cats : np.ndarray of category label strings
        row_margin, col_margin : np.ndarray of normalized marginal distributions
        target_pop_tot : int, total population count
        num_row_cats, num_col_cats : int
    """
    target_pop_tot = int(col_margin_df[pop_col].sum())

    row_cats = row_margin_df[row_var].values
    col_cats = col_margin_df[col_var].values

    row_margin = (row_margin_df[pop_col] / row_margin_df[pop_col].sum()).values
    col_margin = (col_margin_df[pop_col] / target_pop_tot).values

    return {
        "row_cats": row_cats,
        "col_cats": col_cats,
        "row_margin": row_margin,
        "col_margin": col_margin,
        "target_pop_tot": target_pop_tot,
        "num_row_cats": len(row_cats),
        "num_col_cats": len(col_cats),
    }


def prep_atus_target(atus_meta_df, row_var, col_var, cluster_label_col, weight_col, num_row_cats, num_col_cats, num_clusters):
    """
    Compute the ATUS conditional cluster distribution table P(cluster | demographic stratum).

    Parameters
    ----------
    atus_meta_df : pd.DataFrame
        ATUS respondent metadata with demographic and cluster columns.
    row_var, col_var : str
        Column names for the stratification variables (must be integer-coded 0..K-1).
    cluster_label_col : str
        Column name for cluster assignments.
    weight_col : str
        Column name for respondent weights.
    num_row_cats, num_col_cats, num_clusters : int

    Returns
    -------
    atus_target_P : np.ndarray, shape (num_row_cats * num_col_cats, num_clusters)
        Row-normalized conditional distribution table.
    """
    atus_target_P, _ = compute_atus_target_table(
        atus_meta_df,
        stratum_var1_col=row_var,
        stratum_var2_col=col_var,
        cluster_label_col=cluster_label_col,
        weight_col=weight_col,
        num_var1_cats=num_row_cats,
        num_var2_cats=num_col_cats,
        num_clusters=num_clusters,
    )
    return atus_target_P


def assign_mobility_clusters(
    dist_df, 
    atus_meta_df, 
    cluster_label_col, 
    k, 
    threshold, 
    medoid_indices=None, 
    medoid_thresholds=None
    ):
    """
    Assign mobility users to ATUS clusters via k-NN voting on a precomputed distance matrix.

    The distance matrix columns are assumed to correspond to ATUS respondents in the
    same order as atus_meta_df rows. Any trailing metadata columns (e.g. GEOID) must
    be excluded before passing dist_df — use the `atus_cols` slice described below,
    or pass a pre-sliced array.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Distance matrix with mobility users as rows. Must include a GEOID column and
        one column per ATUS respondent (in the same order as atus_meta_df).
    atus_meta_df : pd.DataFrame
        ATUS metadata; used to determine the number of ATUS respondent columns and
        to extract cluster labels.
    cluster_label_col : str
        Column in atus_meta_df containing integer cluster labels.
    k : int
        Number of nearest ATUS neighbors to consider for voting.
    threshold : float
        Minimum fraction of k neighbors required to agree on a label.
        Users below threshold are assigned -1 (unassigned).
    medoid_thresholds : dict or None
        Optional dict of {cluster_label: (medoid_col_index, max_distance)}.
        Users assigned to a cluster but farther than max_distance from that
        cluster's medoid column in dist_df are unassigned (-1).

    Returns
    -------
    assigned_labels : np.ndarray, shape (N,)
        Integer cluster labels (-1 = unassigned).
    """
    dist_matrix_arr = dist_df.iloc[:, :len(atus_meta_df)].values
    atus_labels = atus_meta_df[cluster_label_col].values
    assigned_labels = knn_cluster_label(
        k, dist_matrix_arr, atus_labels, threshold,
        medoids=medoid_indices,
        medoid_thresh=medoid_thresholds,
    )
    return assigned_labels



def filter_valid_users(users_df, acs_cbg_df, assigned_labels, geoid_col="GEOID", id_col="user_id"):
    """
    Filter mobility users to those whose home CBG appears in the ACS data,
    and return aligned inputs for the Calibrator.

    Parameters
    ----------
    users_df : pd.DataFrame
        Mobility user records; must contain geoid_col.
    acs_cbg_df : pd.DataFrame
        CBG-level ACS probability table; must contain geoid_col.
    assigned_labels : np.ndarray
        Cluster labels aligned to users_df rows.
    geoid_col : str

    Returns
    -------
    home_cbgs : np.ndarray
        GEOID values for valid users.
    assigned_labels_filtered : np.ndarray
        Cluster labels for valid users.
    n_dropped : int
        Number of users dropped due to missing GEOID.
    """
    mask, _ = get_valid_mask_by_acs_geoid(users_df, acs_cbg_df, geoid_col=geoid_col)
    valid = users_df.loc[mask].reset_index(drop=True)
    ids_filtered = valid[id_col]
    home_cbgs_filtered = valid[geoid_col].values
    assigned_labels_filtered = assigned_labels[mask]
    n_dropped = int((~mask).sum())
    return ids_filtered, home_cbgs_filtered, assigned_labels_filtered, n_dropped


def load_medoid_info(path):
    import json
    with open(path) as f:
        medoid_info = json.load(f)
    return {
        "medoid_indices":    {int(k): v for k, v in medoid_info["medoid_indices"].items()},
        "medoid_thresholds": {int(k): v for k, v in medoid_info["medoid_thresholds"].items()},
    }