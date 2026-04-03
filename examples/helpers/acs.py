import pandas as pd


def load_acs_table(path, skiprows=None, **kwargs):
    table = pd.read_csv(path, skiprows=skiprows, **kwargs)
    return table


def process_income_table(path, group_mapping, last_row_margins=True):
    """
    Processes ACS B19001 (household income) table and returns a wide DataFrame 
    containing normalized income group distribution for each GEOID (id_col).

    Columns contain the id_col (e.g. GEOID) and each income group in group_mapping.

    If last_row_margins (i.e. last row in the supplied ACS file) corresponds to
    marginal estimates e.g. at the CBSA level, the function will additionally 
    return a Series of those estimates grouped accordingly. This is the intended behavior.
    Otherwise, returns (processed_table, None).

    TODO: 
    - refactor into another function to allow separate processing of margins in a different ACS table
    """

    id_col = "GEOID"
    group_labels = list(group_mapping.keys())

    table = load_acs_table(path, skiprows=[1])
    table[id_col] = table["GEO_ID"].str[9:]

    # select only estimates columns and discard margin of errors
    TOTAL_COL = "B19001_001E" # to be excluded
    estimate_cols = [col for col in table.columns if col.endswith("E") and col != TOTAL_COL]
    table[estimate_cols] = table[estimate_cols].apply(pd.to_numeric, errors="coerce")
    cols_to_keep = [id_col] + estimate_cols
    table = table[cols_to_keep]

    # remap income into coarse categories by combining columns based on group_mapping 
    for group_label, cols in group_mapping.items():
        table[group_label] = table[cols].sum(axis=1)

    # drop CBSA row if last_row_margins and return that as a separate Series
    if last_row_margins:
        marginal_values = table.iloc[-1][group_labels].astype(float)
        table = table.iloc[:-1].copy()
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


def process_age_table(path, group_mapping, drop_groups=None, last_row_margins=True):
    """
    Processes ACS B01001 (sex by age) table and returns a wide DataFrame 
    containing normalized age group distribution for each GEOID (id_col).
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
    
    # extract age groups from sex by age variables
    table = age_from_sex_by_age_table(table, group_mapping, drop_groups)
    
    # extract marginal values if last_row_margins
    if last_row_margins:
        marginal_values = table.iloc[-1][group_labels_keep].astype(float)
        table = table.iloc[:-1].copy()
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
    """ Remap age groups into coarse categories
    The B01001 table is sex by age and there are many joint age categories, 
    (e.g. "Estimate!!Total:!!Male:!!Under 5 years").
    Since there are be many columns to combine, it's easier to pivot to long form
    then split the column names by the delimiter, drop the sex info and remap by age group names
    then finally do a group-by to sum up the estimates
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

def merge_acs_tables(tables):
    return



def cbsa_marginals(series, var_name):
    return