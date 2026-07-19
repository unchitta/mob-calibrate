import numpy as np
import pandas as pd
import warnings

from .utils import align_idx, make_joint_code, weighted_crosstab




def build_cdfs(cbg_probs, row_cats, col_cats, geoid_col='GEOID'):
    # index by GEOID for fast alignment
    probs = cbg_probs.set_index(geoid_col)

    P_row = probs[row_cats].to_numpy(dtype=float)
    P_col = probs[col_cats].to_numpy(dtype=float)

    # make sure probs are normalized
    P_row /= P_row.sum(axis=1, keepdims=True)
    P_col /= P_col.sum(axis=1, keepdims=True)

    C_row = np.cumsum(P_row, axis=1)
    C_col = np.cumsum(P_col, axis=1)

    geoid_all = probs.index.to_numpy()

    return geoid_all, C_row, C_col


def align_cdfs_to_home_cbgs(home_cbgs, acs_cbgs, cdf_row, cdf_col):

    pos = align_idx(home_cbgs, acs_cbgs)

    if (pos < 0).any():
        raise ValueError(f"home_cbgs contain GEOIDs missing from acs_cbgs. Prefilter using get_valid_mask_by_acs_geoid() before!")
    
    cdf_row_aligned = cdf_row[pos]
    cdf_col_aligned = cdf_col[pos]

    return cdf_row_aligned, cdf_col_aligned


def sample_codes_from_cdf(cdf, rng):
    """
    C: (n, K) cumulative probs, last col should be ~1
    returns int codes in [0, K-1], shape (n,)
    """
    u = rng.random(cdf.shape[0])[:, None] # (n, 1)
    return (u > cdf).sum(axis=1).astype(np.int32) # inverse-CDF


def stage1_ipf(sampled_row_codes, sampled_col_codes, row_margin, col_margin, n_iter=10):
    """
    IPF on two categorical margins.

    sampled_row_codes: (N,) sampled integer category codes 0..K_row-1 for the row variable
    sampled_col_codes: (N,) sampled integer category codes 0..K_col-1 for the column variable
    row_margin: (K_row) target row marginals (list-like), sums to 1
    col_margin: (K_col) target col marginals (list-like), sums to 1

    initial weights are set to 1/N

    returns stage1 weights on the same scale as 1/N
    """

    # make sure every input is numpy
    row = np.asarray(sampled_row_codes, dtype=np.int32)
    col = np.asarray(sampled_col_codes, dtype=np.int32)
    row_margin = np.asarray(row_margin, dtype=np.float64)
    col_margin = np.asarray(col_margin, dtype=np.float64)

    # number of categories for each margin
    K_row = row_margin.size
    K_col = col_margin.size

    # sanity checks
    N = row.size
    if N == 0 or col.size != N:
        raise ValueError("row/col code arrays must be same length and N>0")
    if row.min() < 0 or row.max() >= K_row:
        raise ValueError("sampled_row_codes out of bounds")
    if col.min() < 0 or col.max() >= K_col:
        raise ValueError("sampled_col_codes out of bounds")
    
    # initial weights
    weights = np.full(N, 1.0 / N, dtype=np.float64)

    # perform raking
    def rake_margin(sample, target, K):
        # current weighted row/col totals
        tot = np.bincount(sample, weights=weights, minlength=K)
        # adjustment factor (target / row(col) total)
        adj = np.divide(target, tot, out=np.ones(K), where=tot > 0)
        # multiply weights by adjustment factor
        np.multiply(weights, adj[sample], out=weights)

    for _ in range(n_iter):
        rake_margin(row, row_margin, K_row)
        rake_margin(col, col_margin, K_col)

    ### diagnostics
    ### (check convergence within absolute + relative tolerance)
    final_row_tot = np.bincount(row, weights=weights, minlength=K_row)
    final_col_tot = np.bincount(col, weights=weights, minlength=K_col)
    row_abs = np.max(np.abs(final_row_tot - row_margin))
    col_abs = np.max(np.abs(final_col_tot - col_margin))
    
    atol=1e-4
    rtol=1e-2
    row_converged = np.allclose(final_row_tot, row_margin, atol=atol, rtol=rtol)
    col_converged = np.allclose(final_col_tot, col_margin, atol=atol, rtol=rtol)
    converged = row_converged and col_converged

    if not converged:
        warnings.warn(
            "IPF did not converge within tolerance.\n"
            f"Row max abs diff: {row_abs:.6g}\n"
            f"Col max abs diff: {col_abs:.6g}\n"
            f"atol={atol}, rtol={rtol}",
            RuntimeWarning
        )

    diag = {
        "converged": converged,
        "row_max_abs_diff": float(row_abs),
        "col_max_abs_diff": float(col_abs),
        "atol": atol,
        "rtol": rtol,
    }

    return weights, diag


def stage1_rake(sampled_row_codes, sampled_col_codes, num_col_cats, joint_margin):
    """
    Single-margin raking on the joint (row, col) distribution.

    Use when a CBSA-level joint target is available (e.g. ACS B19037, household
    income x age of householder). Strictly stronger than two-margin IPF: matches
    the joint exactly (up to sampling support) and thus both marginals as well.
    A single pass suffices because there is only one constraint.

    sampled_row_codes: (N,) int codes 0..K_row-1
    sampled_col_codes: (N,) int codes 0..K_col-1
    num_col_cats: K_col, used to flatten (row, col) -> joint code in row-major
    joint_margin: target joint distribution, 1D (K_row*K_col,) or 2D (K_row, K_col),
                  summing to 1. Row-major flattening matches make_joint_code.

    Initial weights are set to 1/N. Returns weights on the same scale as 1/N.
    """

    row = np.asarray(sampled_row_codes, dtype=np.int32)
    col = np.asarray(sampled_col_codes, dtype=np.int32)
    joint_margin = np.asarray(joint_margin, dtype=np.float64).ravel()

    K = joint_margin.size
    K_col = int(num_col_cats)
    if K % K_col != 0:
        raise ValueError(
            f"joint_margin size {K} not divisible by num_col_cats {K_col}"
        )

    N = row.size
    if N == 0 or col.size != N:
        raise ValueError("row/col code arrays must be same length and N>0")
    if row.min() < 0 or col.min() < 0 or col.max() >= K_col:
        raise ValueError("sampled codes out of bounds")

    joint = make_joint_code(row, col, K_col)
    if joint.max() >= K:
        raise ValueError("joint codes exceed joint_margin support")

    weights = np.full(N, 1.0 / N, dtype=np.float64)
    tot = np.bincount(joint, weights=weights, minlength=K)
    adj = np.divide(joint_margin, tot, out=np.ones(K), where=tot > 0)
    np.multiply(weights, adj[joint], out=weights)

    # diagnostics (convergence within abs+rel tolerance)
    final_tot = np.bincount(joint, weights=weights, minlength=K)
    joint_abs = float(np.max(np.abs(final_tot - joint_margin)))

    atol = 1e-4
    rtol = 1e-2
    converged = np.allclose(final_tot, joint_margin, atol=atol, rtol=rtol)

    # cells with a positive target but no sampled support cannot be matched
    unsupported = (joint_margin > 0) & (tot == 0)
    if unsupported.any():
        warnings.warn(
            f"stage1_rake: {int(unsupported.sum())} target cell(s) have no sampled "
            f"support; their target mass cannot be matched.",
            RuntimeWarning,
        )

    if not converged:
        warnings.warn(
            "stage1_rake did not converge within tolerance.\n"
            f"Joint max abs diff: {joint_abs:.6g}\n"
            f"atol={atol}, rtol={rtol}",
            RuntimeWarning,
        )

    diag = {
        "converged": converged,
        "joint_max_abs_diff": joint_abs,
        "atol": atol,
        "rtol": rtol,
        "unsupported_cells": int(unsupported.sum()),
    }

    return weights, diag


def stage2_rake(
    sampled_row_codes,           # (N,) int codes for joint demo cell g in 0..G-1
    sampled_col_codes,
    num_row_cats,
    num_col_cats,
    assigned_cluster_labels,         # (N,) int in {-1, 0..K-1}
    stage1_weights,                  # (N,) float64 stage1 weights
    P_target,            # (G, K) float64, rows sum to 1 (target p(cluster|demo))
    downweight_factor=1, # proportion of weights of unassigned users to redistribute
    factor_clip=None,    # e.g. (0.05, 20.0) or None
):
    
    """
    Stage 2: calibrate cluster distribution within each demographic stratum.

    - Excludes rows with cluster_code == -1 from the raking step.
    - Leaves those rows' weights unchanged (w2 = w1 for them).
    - Uses conditional targets P_target[g, k] (rows sum to 1).

    Returns: (w2, diag)
    """

    # make sure arrays are in numpy for fast computing
    sampled_joint_demog =  make_joint_code(sampled_row_codes, sampled_col_codes, num_col_cats)
    s = np.asarray(sampled_joint_demog, dtype=np.int32)
    c = np.asarray(assigned_cluster_labels, dtype=np.int32)
    w1 = np.asarray(stage1_weights, dtype=np.float64)
    P_target = np.asarray(P_target, dtype=np.float64)
    G, K = P_target.shape
    N = s.size

    # sanity checks
    if N == 0 or c.size != N or w1.size != N:
        raise ValueError("arrays must have the same length")
    if s.min() < 0 or s.max() >= G:
        raise ValueError("sampled_joint_demog out of bounds (expected 0..G-1)")
    if (c.max() >= K) or (c.min() < -1):
        raise ValueError("assigned_cluster_labels out of bounds (expected -1 or 0..K-1)")

    # select only assigned users
    assigned = (c >= 0)
    w2 = w1.copy()

    if not assigned.any():
        warnings.warn("No assigned clusters, returning Stage 1 weights", RuntimeWarning)
        return w2, {"n_eligible": 0, "n_infeasible_cells": 0}

    s_a = s[assigned]
    c_a = c[assigned]
    w_a = w1[assigned]

    # weighted sum table among assigned users
    cell = make_joint_code(s_a, c_a, K)
    weighted_counts, stratum_totals, P_contingency = weighted_crosstab(s, c, w2, G, K, mask=assigned)
    """
    cell =  s_a * K + c_a
    cell = make_joint_code(s_a, c_a, K)
    weighted_counts = np.bincount(cell, weights=w_a, minlength=G * K).reshape(G, K)  # (G,K)

    # totals in each joint demographic stratum
    stratum_totals = weighted_counts.sum(axis=1)  # (G,)

    # normalized the weighted sum table among assigned users to get contingency table
    P_contingency = np.zeros_like(weighted_counts)
    np.divide(weighted_counts, stratum_totals[:, None], out=P_contingency, where=(stratum_totals[:, None] > 0))
    print(P_contingency)
    """

    # handle zero-cells in contingency table
    #### set the corresponding cells in the target table to 0 and renormalize within that stratum
    zero_cells = (weighted_counts == 0)                       # (G,K)
    infeasible = (P_target > 0) & (zero_cells) & (stratum_totals[:, None] > 0)
    n_infeasible = int(infeasible.sum())
    P_target_adj = P_target.copy()
    if n_infeasible > 0:
        # zero out unsupported target cells
        P_target_adj = np.where(~zero_cells, P_target_adj, 0.0)
        # renormalize rows to sum to 1
        row_sum = P_target_adj.sum(axis=1)  # (G,)
        scale = np.divide(1.0, row_sum, out=np.ones_like(row_sum), where=row_sum > 0)
        P_target_adj *= scale[:, None]
        warnings.warn(
            f"Stage 2: {n_infeasible} cells had positive target but zero support; "
            "truncated targets to supported clusters and renormalized within stratum.",
            RuntimeWarning
        )

    # compute raking ratio between the target and contigency
    adj_factors = np.ones_like(P_contingency)
    np.divide(P_target_adj, P_contingency, out=adj_factors, where=(P_contingency > 0) & (stratum_totals[:, None] > 0))
    
    if factor_clip is not None:
        lo, hi = factor_clip
        adj_factors = np.clip(adj_factors, lo, hi)
    
    # adjust the weights of the assigned users
    w2[assigned] = w_a * adj_factors.ravel()[cell]

    ### REDISTRIBUTE WEIGHT
    if 0 < downweight_factor <= 1:
                
        # sum_distributable per stratum: downweight_factor * sum(w[~assigned] in stratum)
        sum_unassigned_by_stratum = np.bincount(s[~assigned], weights=w2[~assigned], minlength=G)
        sum_distributable_by_stratum = downweight_factor * sum_unassigned_by_stratum

        # weighted count assigned users of per stratum (so that distributable weights can be divided)
        denom_by_stratum = np.bincount(s[assigned], weights=w2[assigned], minlength=G)

        # ratio per stratum: sum_distributable / denom (0 if denom==0)
        ratio_by_stratum = np.zeros(G, dtype=np.float64)
        np.divide(sum_distributable_by_stratum, denom_by_stratum, out=ratio_by_stratum, where=denom_by_stratum > 0)

        # apply distribution
        w2[assigned] += w2[assigned] * ratio_by_stratum[s[assigned]]
        w2[~assigned] *= (1.0 - downweight_factor)


    weighted_counts, stratum_totals, P_contingency = weighted_crosstab(s, c, w2, G, K, mask=assigned)
    #print(P_contingency)

    diagnostics = {
        "n_assigned": int(assigned.sum()),
        "n_infeasible_cells": n_infeasible,
    }
    return w2, diagnostics
