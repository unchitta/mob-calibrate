import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import *



def knn_cluster_label(k, dist_matrix, atus_cluster_labels, assignment_threshold, medoids=None, medoid_thresh=None):
    """
    assign cuebiq users to atus clusters using voting
    within the set of neighest atus respondents

    at least `assigment_threshold` of the respondents within 
    the nearest respondent set must belong to the same cluster 
    for the cuebiq user to be assigned to that cluster

    if there are multiple clusters that meet this criterion 
    with ties in the number of nearest respondents,
    choose cluster with the smaller sum of distances

    if tie in both the number of neighbors and sum of distances, 
    choose at random
    """

    def find_nn_idx(dist_matrix, k):
        nn_idx = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        return nn_idx
    
    assigned_cluster_labels = []
    nn_idx = find_nn_idx(dist_matrix, k)
    rng = np.random.default_rng()
    for i in tqdm(range(nn_idx.shape[0])):
        nn = nn_idx[i, :]
        nn_dists = dist_matrix[i, nn]
        nn_clusters = atus_cluster_labels[nn]
        labels, counts = np.unique(nn_clusters, return_counts=True)
        props = counts / counts.sum()
        max_frac = np.max(props)

        # if no label meets threshold, label <- unassigned
        if max_frac < assignment_threshold:
            assigned_cluster_labels.append(-1)
            continue
        
        # check for fractional ties
        mask1 = (props == max_frac)
        maj_labels = labels[mask1]
        if maj_labels.size > 1:
            # if tied in # neighnors, calculate the sum of distances for each candidate
            sum_dists = np.array([nn_dists[nn_clusters == label].sum() for label in maj_labels])
            # check to make sure also not tied by distance
            min_dist = np.min(sum_dists)
            mask2 = (sum_dists == min_dist)
            best_labels = maj_labels[mask2]
            if best_labels.size > 1:
                # if also tied in distance, choose at random
                label = rng.choice(best_labels)
            else:
                # else, choose nearest cluster
                label = best_labels[0]
        else:
            # else (no ties in # neighbors), assign the majority label
            label = maj_labels[0]

        # Add check for threshold for distance from medoid
        if medoid_thresh and medoid_thresh:
            # Compute distance from medoid for Cuebiq obs
            dm = dist_matrix[i, medoids[label]]
            if dm > medoid_thresh[label]:
                assigned_cluster_labels.append(-1)
                continue

        assigned_cluster_labels.append(label)

    return np.array(assigned_cluster_labels)



def compute_atus_target_table(
        atus_df, 
        stratum_var1_col, stratum_var2_col, 
        cluster_label_col, weight_col, 
        num_var1_cats, num_var2_cats, num_clusters):
    
    # compute joint demographic stratum codes
    df = atus_df.copy()
    joint = make_joint_code(df[stratum_var1_col], df[stratum_var2_col], num_var2_cats)
    df['joint'] = joint

    # compute target distribution table (rows are demog stratum and columns are clusters)
    # each row sum to 1 (i.e. gives cluster distribution within a stratum)
    num_strata = num_var1_cats * num_var2_cats
    atus_table, row_tot, atus_target_P = weighted_crosstab(
        row_codes=df['joint'],
        col_codes=df[cluster_label_col],
        weights=df[weight_col],
        n_rows=num_strata,
        n_cols=num_clusters,
        mask=None
    )

    return atus_target_P, joint



def get_valid_mask_by_acs_geoid(users_df, acs_cbg_df, geoid_col="GEOID"):
    user_geoids = pd.array(users_df[geoid_col], dtype="string")
    acs_geoids = pd.array(acs_cbg_df[geoid_col].unique(), dtype="string")
    pos = align_idx(user_geoids, acs_geoids)
    mask = pos >= 0 # mask to keep valid users


    # print out diagnostics of invalid users
    missing = ~mask
    n_missing = int(missing.sum())
    missing_vals = user_geoids[missing]
    missing_unique = missing_vals.dropna().unique().to_numpy()
    print(f"{n_missing} rows have user GEOID missing from ACS GEOID (or null), e.g. {missing_unique[:5]}")
    print(f"Unique missing GEOIDs (excluding null): {len(missing_unique)}")

    return mask, pos
