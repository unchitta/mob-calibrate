import numpy as np
import pandas as pd
import tqdm

from .utils import weighted_crosstab, align_idx, make_joint_code




# =============== SEQUENCE METRICS FOR CLUSTERING TUS DATA ================

def sequence_metrics(seq, home_label, work_label, all_labels):
    
    """
    Compute behavioural sequence metrics for a single daily activity sequence.

    Args:
        seq (list): A list of activity labels (e.g. ["Home","Work","Work","Commute"...])
        home_label (str): Label used for "Home" activity
        work_label (str): Label used for "Work" activity
        all_labels (list): list of all possible activity labels in the alphabet

    Returns:
        dict: A dictionary with metrics
    """

    from collections import Counter

    ################## HELPERS #####################
    # Helper: find runs/spells
    def _compute_runs(sequence):
        run_lengths = []
        current_state = sequence[0]
        current_length = 1
        for i in range(1, len(sequence)):
            if sequence[i] == current_state:
                current_length += 1
            else:
                run_lengths.append((current_state, current_length))
                current_state = sequence[i]
                current_length = 1
        run_lengths.append((current_state, current_length))
        return run_lengths

    # Helper: reciprocity from spells
    def _compute_reciprocity(runs):

        # <<< MOD: handle single-activity sequences explicitly
        if len(runs) == 1:
            state = runs[0][0]
            edge_counts = Counter({(state, state): 1})
            return 0.0, dict(edge_counts)

        transitions = [(runs[i][0], runs[i+1][0]) for i in range(len(runs) - 1)]
        edge_set = set(transitions)

        # Count edges
        edge_counts = Counter(transitions)

        # ----- Weighted reciprocity -----
        weighted_mutual = sum(
            min(w, edge_counts.get((j, i), 0))
            for (i, j), w in edge_counts.items()
            if i != j
        )

        total_weight = sum(
            w for (i, j), w in edge_counts.items()
            if i != j
        )

        weighted_reciprocity = (
            weighted_mutual / total_weight
            if total_weight > 0 else 0.0
        )

        return weighted_reciprocity, dict(edge_counts)

    ###################################
    
    # Length
    n = len(seq)

    # Unique activities
    distinct = set(seq)
    num_activities = len(distinct)

    # Activity durations (simple count)
    counts = Counter(seq)
    durations = {act: counts[act]/n if act in counts else 0 for act in all_labels}

    # Turnover rate = # of transitions / (length - 1)
    transitions = sum(1 for i in range(1, n) if seq[i] != seq[i - 1])
    turnover_rate = transitions / (n - 1) if n > 1 else 0

    # Reciprocity
    runs = _compute_runs(seq)
    reciprocity, edges = _compute_reciprocity(runs)

    # initialize results dict
    res = {
        "num_activities": num_activities,
        "turnover_rate": turnover_rate,
        "reciprocity": reciprocity
    }

    # Durations as Time Use
    for k, v in durations.items():
        res[k] = v

    # Edges
    for k, v in edges.items():
        res[f"edge_{k}"] = v

    return res


def compute_metrics_for_all_sequences(sequences, home_label, work_label, all_labels):
    metrics = []
    for s in sequences:
        m = sequence_metrics(s, home_label, work_label, all_labels)
        metrics.append(m)
    seq_metrics = pd.DataFrame(metrics).fillna(0)

    return seq_metrics



# =============== WEIGHTED K-MEDOIDS FUNCTIONS FOR CLUSTERING TUS DATA =================

def weighted_kmedoids(
    D, 
    k, 
    w=None, 
    max_iter=200, 
    tol=1e-6, 
    init="kmedoids++", 
    random_state=None, 
    verbose=False
):
    """
    Weighted K-medoids (PAM) on a precomputed distance matrix.

    Parameters
    ----------
    D : (n,n) ndarray
        Symmetric distance matrix, zeros on diagonal.
    k : int
        Number of clusters / medoids.
    w : (n,) ndarray or None
        Nonnegative weights per point (survey weights). Defaults to ones.
    init : {'random','kmedoids++', array-like}
        Initialization: random, kmedoids++ (weighted), or explicit medoid indices.
    """

    def _weighted_inertia(D, medoids, w):
        """Weighted sum of distances to nearest medoid."""
        # D: (n,n) distance matrix; medoids: list/array of medoid indices; w: (n,)
        dmin = np.min(D[:, medoids], axis=1)
        return float((w * dmin).sum())

    def _init_kmedoids_plusplus(D, k, w, rng):
        """
        k-medoids++ style initialization using weighted probabilities.
        Start with a random point ~ weight; then sample others ~ w * (dist-to-nearest)^2.
        """
        n = D.shape[0]
        medoids = []
        # first medoid: weighted random
        probs = w / w.sum()
        m0 = rng.choice(n, p=probs)
        medoids.append(m0)
        # subsequent medoids
        for _ in range(1, k):
            dmin = np.min(D[:, medoids], axis=1)
            # probability proportional to weight * distance^2
            p = w * (dmin ** 2)
            p_sum = p.sum()
            if p_sum == 0:
                # all points coincide w.r.t. chosen medoids; pick random by weight
                m = rng.choice(n, p=w/w.sum())
            else:
                p = p / p_sum
                m = rng.choice(n, p=p)
            medoids.append(int(m))
        return np.array(sorted(set(medoids)))  # guard against duplicates


    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if np.any(w < 0):
            raise ValueError("Weights must be nonnegative.")
        if w.sum() == 0:
            w = np.ones(n, dtype=float)

    # --- initialize medoids
    if isinstance(init, (list, np.ndarray)):
        medoids = np.array(init, dtype=int)
    elif init == "random":
        probs = w / w.sum()
        medoids = rng.choice(n, size=k, replace=False, p=probs)
    elif init == "kmedoids++":
        medoids = _init_kmedoids_plusplus(D, k, w, rng)
        # if duplicates happened, fill up randomly
        while len(medoids) < k:
            cand = rng.choice(n, p=w/w.sum())
            if cand not in medoids:
                medoids = np.append(medoids, cand)
    else:
        raise ValueError("Unknown init")

    # ensure size k
    if len(medoids) != k:
        raise ValueError("Initialization did not produce k distinct medoids.")

    # Precompute assignment structures
    d_to_m = np.min(D[:, medoids], axis=1)
    nearest_m = medoids[np.argmin(D[:, medoids], axis=1)]
    inertia = float((w * d_to_m).sum())

    if verbose:
        print(f"init inertia={inertia:.6f}")

    # --- PAM SWAP loop
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for mi_idx, mi in enumerate(list(medoids)):
            # Try swapping out medoid mi with every non-medoid h
            non_m = [h for h in range(n) if h not in medoids]
            best_delta = 0.0
            best_h = None

            for h in non_m:
                # Compute change in weighted inertia if we swap mi -> h
                # For each point x, its new distance is min( old-best (unless mi was its best), D[x,h], other medoids )
                # Efficient computation:
                # current best distance d1; current second-best distance d2 (w.r.t. current medoids)
                d_all = D[:, medoids]
                # obtain second-best distances by masking the column of mi
                d1 = d_to_m
                # column index of mi in current medoids:
                col_mi = mi_idx
                # second-best = min over all medoids except mi
                if k > 1:
                    mask = np.ones(k, dtype=bool)
                    mask[col_mi] = False
                    d2 = np.min(d_all[:, mask], axis=1)
                else:
                    d2 = np.full(n, np.inf)

                # candidate distance via new medoid h
                dh = D[:, h]
                # If mi was not the nearest, new best = min(d1, dh)
                # If mi was the nearest, new best = min(d2, dh)
                is_near_mi = (nearest_m == mi)
                new_best = np.where(is_near_mi, np.minimum(d2, dh), np.minimum(d1, dh))

                delta = (w * (new_best - d1)).sum()   # change in objective
                if delta < best_delta:
                    best_delta = delta
                    best_h = h

            # Apply best swap for this medoid if it improves
            if best_h is not None and best_delta < -tol:
                # perform swap
                medoids[mi_idx] = best_h
                # update nearest distances and assignments
                d_to_m = np.min(D[:, medoids], axis=1)
                nearest_m = medoids[np.argmin(D[:, medoids], axis=1)]
                inertia += best_delta
                improved = True
                if verbose:
                    print(f"iter {it}: swap {mi} -> {best_h}, inertia={inertia:.6f}")

        if not improved and verbose:
            print(f"no improving swap at iter {it}, inertia={inertia:.6f}")

    # Final labels
    labels = np.argmin(D[:, medoids], axis=1)
    
    # return dict for medoids {cluster label: medoid}
    medoids_dict = {j: int(m) for j, m in enumerate(medoids)}

    return medoids_dict, labels, inertia


def fit_weighted_kmedoids_with_restarts(
        D, 
        wA, 
        K, 
        n_restarts=10, 
        random_state=None, 
        **kwargs
    ):
    """
    Run weighted_kmedoids several times and keep the best (lowest inertia).
    kwargs are passed to weighted_kmedoids (e.g., max_iter, tol, init, verbose).
    """
    rng = np.random.default_rng(random_state)
    best = None
    for r in range(n_restarts):
        #seed = int(rng.integers(0, 2**31-1))
        seed = 1097657231
        medoids_dict, labels, inertia= weighted_kmedoids(
            D, K, w=wA, random_state=seed, **kwargs
        )
        if (best is None) or (inertia < best["inertia"]):
            best = {"medoids": medoids_dict, "labels": labels, "inertia": inertia, "seed": seed}
    return best



# =============== FUNCTIONS FOR ASSIGNING MOBILITY USERS TO TUS CLUSTERS =================

def compute_dist_to_medoid_thresholds(
    D: np.ndarray,
    medoids_dict: dict,
    labels: np.ndarray,
    percentile: float = 99.0,
):
    """
    Returns
    -------
    medoid_thresh : dict
        {label: percentile distance threshold}
    """
    medoid_thresh = {}

    for label, medoid_idx in medoids_dict.items():
        members = np.where(labels == label)[0]
        if members.size == 0:
            continue
        d = D[members, medoid_idx]
        medoid_thresh[label] = float(np.percentile(d, percentile))

    return medoid_thresh


def knn_cluster_label(k, dist_matrix, atus_cluster_labels, assignment_threshold, medoids=None, medoid_thresh=None):
    """
    assign mobility users to atus clusters using voting
    within the set of neighest atus respondents

    at least `assigment_threshold` of the respondents within 
    the nearest respondent set must belong to the same cluster 
    for the mobility user to be assigned to that cluster

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



# =================================================


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
