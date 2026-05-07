import pandas as pd
import numpy as np
from pathlib import Path



def load_acs_table(path, skiprows=None, **kwargs):
    table = pd.read_csv(path, skiprows=skiprows, **kwargs)
    return table


# Ordered ACS B19001 income bin columns and dollar values of the upper bound.
# This will be used to find income quartiles for each CBSA
# (The last bin (200k+) has no upper bound)
_B19001_BINS = [
    ("B19001_002E", 10_000),
    ("B19001_003E", 15_000),
    ("B19001_004E", 20_000),
    ("B19001_005E", 25_000),
    ("B19001_006E", 30_000),
    ("B19001_007E", 35_000),
    ("B19001_008E", 40_000),
    ("B19001_009E", 45_000),
    ("B19001_010E", 50_000),
    ("B19001_011E", 60_000),
    ("B19001_012E", 75_000),
    ("B19001_013E", 100_000),
    ("B19001_014E", 125_000),
    ("B19001_015E", 150_000),
    ("B19001_016E", 200_000),
    ("B19001_017E", float("inf")),
]
 
def _format_dollar(value):
    """Format a dollar threshold as short readable string (e.g. 50k for $50,000)"""
    if value == float("inf"):
        return None
    if value >= 1_000:
        return f"{int(value // 1_000)}k"
    return str(int(value))
 
 
def compute_income_quartile_mapping(table):
    """
    Derive a quartile-based income group mapping from a raw ACS B19001 table.
 
    Uses the CBSA-level row (last row) as a weighted frequency distribution
    to locate the 25th, 50th, and 75th percentile bin boundaries, then
    assigns ACS columns to four quartile groups accordingly.
 
    Parameters
    ----------
    table : pd.DataFrame or path to table
        Raw ACS B19001 table as loaded from CSV (before any processing)
        or path to the CSV file downloaded from data.census.gov.
        The last row must be the CBSA-level aggregate.
 
    Returns
    -------
    dict
        Mapping of quartile labels to lists of ACS column names, in the
        same format expected by process_income_table's group_mapping argument.
        Labels are auto-generated from bin boundaries, e.g. '<35k', '35k-75k'.
    """

    if isinstance(table, (str, Path)):
        table = load_acs_table(table, skiprows=[1])

    cols = [col for col, _ in _B19001_BINS]
    upper_bounds = [ub for _, ub in _B19001_BINS]
 
    cbsa_row = table.iloc[-1][cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    total = cbsa_row.sum()
    if total == 0:
        raise ValueError("CBSA row sums to zero — check that the last row is the CBSA aggregate.")
 
    cdf = cbsa_row.cumsum() / total
    quartile_thresholds = [0.25, 0.50, 0.75]
 
    # find the bin index whose CDF is closest to each quartile threshold
    split_indices = []
    for q in quartile_thresholds:
        idx = int(np.argmin(np.abs(cdf - q)))
        split_indices.append(idx)
 
    # deduplicate splits (can happen if one bin spans multiple quartiles)
    split_indices = sorted(set(split_indices))
 
    # build groups: each group is bins from previous split+1 to current split (inclusive)
    groups = {}
    boundaries = [0] + [i + 1 for i in split_indices] + [len(cols)]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        group_cols = cols[start:end]
        lower = upper_bounds[start - 1] if start > 0 else 0
        upper = upper_bounds[end - 1]
        lower_str = _format_dollar(lower)
        upper_str = _format_dollar(upper)
        if lower == 0:
            label = f"<{upper_str}"
        elif upper == float("inf"):
            label = f"{lower_str}+"
        else:
            label = f"{lower_str}-{upper_str}"
        groups[label] = group_cols
 
    return groups


def process_income_table(path, group_mapping=None, cbsa_code=None):
    """
    Process ACS B19001 (household income) into a wide table of
    normalized income-group distributions by GEOID.

    Parameters
    ----------
    path : str
        Path to the ACS table.
    group_mapping : dict
        Mapping of income-group labels to ACS columns to aggregate.
        If None, quartile boundaries are derived automatically from the
        CBSA-level row via compute_income_quartile_mapping().
    cbsa_code : str or int, optional
        CBSA code (e.g. 38060) identifying the metro-aggregate row to
        extract as marginal totals. The row is located by exact GEOID
        match. If None, no marginals are returned.

    Returns
    -------
    table : pd.DataFrame
        Wide table with GEOID and normalized income-group columns.
    marginals : pd.Series or None
        Grouped marginal values for the CBSA row if cbsa_code is given,
        else None.
    """

    id_col = "GEOID"

    table = load_acs_table(path, skiprows=[1])
    table[id_col] = table["GEO_ID"].str[9:]

    if group_mapping is None:
        group_mapping = compute_income_quartile_mapping(table)

    group_labels = list(group_mapping.keys())

    # select only estimates columns and discard margin of errors
    TOTAL_COL = "B19001_001E" # to be excluded
    estimate_cols = [col for col in table.columns if col.endswith("E") and col != TOTAL_COL]
    table[estimate_cols] = table[estimate_cols].apply(pd.to_numeric, errors="coerce")
    cols_to_keep = [id_col] + estimate_cols
    table = table[cols_to_keep]

    # remap income into coarse categories by combining columns based on group_mapping
    for group_label, cols in group_mapping.items():
        table[group_label] = table[cols].sum(axis=1)

    # extract CBSA-aggregate row (located by GEOID) as marginals, then drop it
    if cbsa_code is not None:
        cbsa_geoid = str(cbsa_code)
        cbsa_mask = table[id_col] == cbsa_geoid
        if cbsa_mask.sum() != 1:
            raise ValueError(
                f"Expected exactly one row with GEOID '{cbsa_geoid}', "
                f"found {int(cbsa_mask.sum())}."
            )
        marginal_values = table.loc[cbsa_mask, group_labels].iloc[0].astype(float)
        table = table.loc[~cbsa_mask].copy()
    else:
        marginal_values = None

    # normalize into probabilities per each GEOID
    table['total'] = table[group_labels].sum(axis=1)
    table[group_labels] = table[group_labels].div(
        table['total'].replace(0, pd.NA),
        axis=0
    )
    # keep only GEOID and the group proportions per row
    table = table[[id_col] + group_labels]
    
    return table, marginal_values


def process_age_table(path, group_mapping, drop_groups=None, cbsa_code=None):
    """
    Process ACS B01001 (sex by age) into a wide table of
    normalized age-group distributions by GEOID.

    Parameters
    ----------
    path : str
        Path to the ACS table.
    group_mapping : dict
        Mapping of age-group labels to lists of detailed ACS age categories.
    drop_groups : list, optional
        Age groups to exclude from the output.
    cbsa_code : str or int, optional
        CBSA code (e.g. 38060) identifying the metro-aggregate row to
        extract as marginal totals. The row is located by exact GEOID
        match. If None, no marginals are returned.

    Returns
    -------
    table : pd.DataFrame
        Wide table with GEOID and normalized age-group columns.
    marginals : pd.Series or None
        Grouped marginal values for the CBSA row if cbsa_code is given,
        else None.
    """

    id_col = "GEOID"
    group_labels_keep = [
        group for group in group_mapping.keys()
        if drop_groups is None or group not in drop_groups
    ]

    # load and process B01001 table skipping first row
    table = load_acs_table(path, skiprows=[0])
    table[id_col] = table["Geography"].str[9:]

    estimate_cols = [
        col for col in table.columns
        if col.startswith("Estimate!!Total:!!")
        and col not in ["Estimate!!Total:!!Male:", "Estimate!!Total:!!Female:"]
    ]
    cols_to_keep = [id_col] + estimate_cols
    table = table[cols_to_keep]

    # extract age groups from sex by age variables (table now has GEOID as index)
    table = age_from_sex_by_age_table(table, group_mapping, drop_groups)

    # extract CBSA-aggregate row (located by GEOID) as marginals, then drop it.
    # iloc[-1] would be unsafe here: age_from_sex_by_age_table pivots on GEOID,
    # which sorts alphabetically; for multi-state CBSAs (e.g. 35620 spans
    # NY/NJ/PA), the CBSA's own GEOID can sort between block-group GEOIDs
    # rather than at the end.
    if cbsa_code is not None:
        cbsa_geoid = str(cbsa_code)
        if cbsa_geoid not in table.index:
            raise ValueError(f"CBSA row with GEOID '{cbsa_geoid}' not found in table.")
        marginal_values = table.loc[cbsa_geoid, group_labels_keep].astype(float)
        table = table.drop(index=cbsa_geoid).copy()
    else:
        marginal_values = None
    
    # normalize into probabilities per each GEOID
    table['total'] = table[group_labels_keep].sum(axis=1)
    table[group_labels_keep] = table[group_labels_keep].div(
        table['total'].replace(0, pd.NA),
        axis=0
    )
    table = table.drop(columns="total")
    table.columns.name = None
    table = table.reset_index()

    return table, marginal_values


def age_from_sex_by_age_table(table, group_mapping, drop_groups=None):
    """
    Aggregate detailed B01001 sex-by-age columns into coarse age groups
    by reshaping to long form, mapping labels, and summing by GEOID.

    Resulting aggregate age groups are based on the keys in group_mapping.
    If drop_groups specified, drop the groups from the result.
    """

    age = table.copy()

    # melt to long form
    value_vars = [
        col for col in table.columns
        if col.startswith("Estimate!!Total:!!")
        and col not in ["Estimate!!Total:!!Male:", "Estimate!!Total:!!Female:"]
    ]
    age = age.melt(id_vars='GEOID', value_vars=value_vars, var_name='variable', value_name='value')
    age['value'] = pd.to_numeric(age['value'], errors='coerce')

    # format age values to be grouped
    age['age'] = (
        age['variable']
        .str.rsplit('!!', n=1)
        .str[-1]
        .str.replace(r':$', '', regex=True)  # drop a trailing colon
        .str.strip()
    )

    # remap into the specified groupings
    age_map = invert_group_mapping(group_mapping)
    age['age_group'] = age['age'].map(age_map)
    age = age.dropna(subset=["age_group"])

    # group by cbg and age_group and sum all values
    age = (
        age.groupby(["GEOID", "age_group"], as_index=False)
        .agg(estimate=("value", "sum"))
    )

    # pivot back into wide form
    age = age.pivot(columns='age_group', index='GEOID', values='estimate')
    age = age.reindex(columns=list(group_mapping.keys()))
    if drop_groups is not None:
        age = age.drop(columns=drop_groups, errors='ignore')

    return age

def invert_group_mapping(group_mapping):
    """
    Convert a mapping of {group_label: [raw_labels]} into
    {raw_label: group_label} for use with Series.map().
    """
    return {
        raw_label: group_label
        for group_label, raw_labels in group_mapping.items()
        for raw_label in raw_labels
    }

def merge_distr_tables(tables, on="GEOID", how="inner", dropna=True):
    """
    Merge a list of processed ACS tables on shared GEOID

    Parameters
    ----------
    tables : list[pd.DataFrame]
        List of wide ACS tables, each with one row per geography.
    on : str, default "GEOID"
        Column to merge on.
    how : str, default "inner"
        Merge type passed to pd.merge.

    Returns
    -------
    pd.DataFrame
        Merged wide table.
    """
    
    from functools import reduce

    if not tables:
        raise ValueError("tables must contain at least one DataFrame")

    if len(tables) == 1:
        return tables[0].copy()

    res = reduce(
        lambda left, right: left.merge(right, on=on, how=how),
        tables
    )

    if dropna:
        return res.dropna()
    return res


def format_cbsa_marginals(series, var_name, value_name = 'pop'):
    """Renames a marginal Series with variable and value labels and returns a DataFrame"""
    m = series.copy()
    m.name = value_name
    m.index.name = var_name
    return m.reset_index()


# Default age grouping, matching walkthrough_02_process_acs.ipynb.
# Users can override by passing their own age_groups dict to process_cbsa.
DEFAULT_AGE_GROUPS = {
    '<18': [
        'Under 5 years',
        '5 to 9 years',
        '10 to 14 years',
        '15 to 17 years',
    ],
    '18-24': [
        '18 and 19 years',
        '20 years',
        '21 years',
        '22 to 24 years',
    ],
    '25-44': [
        '25 to 29 years',
        '30 to 34 years',
        '35 to 39 years',
        '40 to 44 years',
    ],
    '45-66': [
        '45 to 49 years',
        '50 to 54 years',
        '55 to 59 years',
        '60 and 61 years',
        '62 to 64 years',
        '65 and 66 years',
    ],
    '67+': [
        '67 to 69 years',
        '70 to 74 years',
        '75 to 79 years',
        '80 to 84 years',
        '85 years and over',
    ],
}


def process_cbsa(
    income_file,
    age_file,
    cbsa_code,
    income_group_mapping=None,
    age_groups=None,
    drop_age_groups=None,
    row_var='income',
    col_var='age',
):
    """
    One-call ACS pipeline: produces CBG-level joint distribution and CBSA-level margins.

    Thin facade over compute_income_quartile_mapping + process_income_table
    + process_age_table + format_cbsa_marginals + merge_distr_tables.

    Parameters
    ----------
    income_file, age_file : str or Path
        Raw ACS B19001 (income) and B01001 (age) CSVs downloaded from data.census.gov.
    cbsa_code : str or int
        CBSA code (e.g. 38060) used to locate the metro-aggregate row in
        each input table, which is extracted as the marginal totals.
    income_group_mapping : dict or None, default None
        {label: [ACS columns]}. If None, auto-derive as quartiles from the CBSA-level
        row via compute_income_quartile_mapping; the derived labels are printed.
    age_groups : dict or None, default None
        {label: [raw ACS age category strings]}. If None, uses DEFAULT_AGE_GROUPS
        (the grouping used by walkthrough_02_process_acs.ipynb).
    drop_age_groups : list or None
        Age group labels to drop from the output (e.g. ['<18']).
    row_var, col_var : str
        Column names used when formatting the CBSA margin DataFrames
        (must match the ROW_VAR / COL_VAR used downstream).

    Returns
    -------
    acs_cbg_distr : pd.DataFrame
        CBG × (income-group + age-group) joint distribution (row-normalized within each CBG).
    income_margin : pd.DataFrame
        CBSA-level income margin; columns [row_var, 'pop'].
    age_margin : pd.DataFrame
        CBSA-level age margin; columns [col_var, 'pop'].
    """
    if income_group_mapping is None:
        income_group_mapping = compute_income_quartile_mapping(income_file)
        print(f'acs.process_cbsa: auto-derived income quartile mapping: '
              f'{list(income_group_mapping.keys())}')

    if age_groups is None:
        age_groups = DEFAULT_AGE_GROUPS
        print(f'acs.process_cbsa: using DEFAULT_AGE_GROUPS: {list(age_groups.keys())}')

    income_cbg, income_cbsa = process_income_table(
        income_file, group_mapping=income_group_mapping, cbsa_code=cbsa_code
    )
    age_cbg, age_cbsa = process_age_table(
        age_file, group_mapping=age_groups, drop_groups=drop_age_groups, cbsa_code=cbsa_code
    )

    income_margin = format_cbsa_marginals(income_cbsa, var_name=row_var)
    age_margin = format_cbsa_marginals(age_cbsa, var_name=col_var)

    acs_cbg_distr = merge_distr_tables([age_cbg, income_cbg])

    return acs_cbg_distr, income_margin, age_margin