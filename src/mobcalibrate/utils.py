import numpy as np
import pandas as pd






def make_joint_code(var1_codes, var2_codes, num_var2_cats: int):
    var1_codes = np.asarray(var1_codes, dtype=np.int32)
    var2_codes = np.asarray(var2_codes, dtype=np.int32)
    return var1_codes * num_var2_cats + var2_codes



def weighted_crosstab(
    row_codes, # (N,) int in 0..G-1  (e.g., joint stratum)
    col_codes, # (N,) int in 0..K-1  (e.g., cluster)
    weights,   # (N,) float
    n_rows: int, # G
    n_cols: int, # K
    mask=None,    # (N,) boolean array or None
):
    """
    Returns:
      weighted_table   : (G, K) weighted cell totals
      row_tot : (G,)   weighted row totals
      normalized_table  : (G, K) row-normalized shares (zeros where row_tot==0)
    """
    row_codes = np.asarray(row_codes, dtype=np.int32)
    col_codes = np.asarray(col_codes, dtype=np.int32)
    weights  = np.asarray(weights,  dtype=np.float64)

    if mask is None:
        m = np.ones(row_codes.size, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)

    r = row_codes[m]
    c = col_codes[m]
    w = weights[m]
    
    # cell index and weighted table
    cell = make_joint_code(r, c, n_cols)
    weighted_table = np.bincount(cell, weights=w, minlength=n_rows * n_cols).reshape(n_rows, n_cols)

    # row totals and row-normalized shares
    row_tot = weighted_table.sum(axis=1)
    normalized_table = np.zeros_like(weighted_table)
    np.divide(weighted_table, row_tot[:, None], out=normalized_table, where=row_tot[:, None] > 0)

    return weighted_table, row_tot, normalized_table



## HELPER FUNCTION TO FILTER USERS WITH VALID GEOIDs (identified in acs_cbg_probs)

def align_idx(arr1, arr2):
    idxer = pd.Index(pd.array(arr2))
    pos = idxer.get_indexer(arr1)
    return pos

