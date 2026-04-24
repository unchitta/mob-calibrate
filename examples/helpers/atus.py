import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
import datetime
from scipy.stats import mode
from collections import defaultdict



# ==================== FUNCTIONS FOR MATCHING INCOME AND AGE GROUPS TO ACS MARGINS ====================

# Ordered ATUS HEFAMINC/HUFAMINC income codes and upper-bound dollar values
# The last code (16) has no upper bound hence set to inf.
_HUFAMINC_BINS = [
    (1,  5_000),
    (2,  7_500),
    (3,  10_000),
    (4,  12_500),
    (5,  15_000),
    (6,  20_000),
    (7,  25_000),
    (8,  30_000),
    (9,  35_000),
    (10, 40_000),
    (11, 50_000),
    (12, 60_000),
    (13, 75_000),
    (14, 100_000),
    (15, 150_000),
    (16, float("inf")),
]
 

def _parse_income_label(label):
    """
    Parse the upper-bound dollar cutoff from ACS-derived income group label
    Returns inf for the last group (e.g. '125k+').
 
    Expected formats: '<35k', '35k-75k', '75k-125k', '125k+'
    """
    label = label.strip()
    if label.startswith("<"):
        return int(label[1:].replace("k", "")) * 1_000
    elif label.endswith("+"):
        return float("inf")
    elif "-" in label:
        upper = label.split("-")[1]
        return int(upper.replace("k", "")) * 1_000
    else:
        raise ValueError(f"Unrecognized income label format: {label!r}")
 
 
def compute_atus_income_group_mapping(income_margins):
    """
    Derive ATUS HUFAMINC income group mapping from ACS-derived quartile labels.
 
    Reads group labels from the income margins file produced by the ACS processing
    step, parses the dollar cutoffs from those labels, and maps each cutoff to the
    closest ATUS HUFAMINC bin boundary.
 
    Parameters
    ----------
    income_margins : pd.DataFrame or str or Path
        Output of format_cbsa_marginals() as loaded from its saved CSV,
        or a path to that CSV file.
 
    Returns
    -------
    labels : list of str
        Income group labels inherited from the ACS margins file.
    mapping : dict
        {group_idx: lambda} object compatible with
        build_grouped_meta's group_specs mapping format.
    """

    if isinstance(income_margins, (str, Path)):
        income_margins = pd.read_csv(income_margins)
    

    label_col = [c for c in income_margins.columns if c != "pop"][0]
    labels = income_margins[label_col].tolist()
 
    # parse upper bound cutoffs; exclude inf (last group has no split point)
    cutoffs = [_parse_income_label(l) for l in labels]
    split_cutoffs = [c for c in cutoffs if c != float("inf")]
 
    # find closest ATUS bin code for each ACS cutoff
    atus_upper_bounds = [ub for _, ub in _HUFAMINC_BINS]
    split_codes = []
    for cutoff in split_cutoffs:
        diffs = [abs(ub - cutoff) for ub in atus_upper_bounds]
        idx = int(np.argmin(diffs))
        split_codes.append(_HUFAMINC_BINS[idx][0])
 
    # build lambda mapping: boundaries define inclusive upper code per group
    boundaries = [0] + split_codes + [_HUFAMINC_BINS[-1][0]]
    mapping = {}
    for i in range(len(labels)):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i == 0:
            mapping[i] = lambda x, hi=hi: x <= hi
        elif i == len(labels) - 1:
            mapping[i] = lambda x, lo=lo: x > lo
        else:
            mapping[i] = lambda x, lo=lo, hi=hi: lo < x <= hi
 
    return labels, mapping
 
 
def compute_atus_age_group_mapping(age_margins):
    """
    Derive ATUS age group mapping from ACS-derived age group labels.
 
    Reads group labels from the age margins file produced by the ACS processing
    step and constructs integer age range conditions for each group.
 
    Parameters
    ----------
    age_margins : pd.DataFrame or str or Path
        Output of format_cbsa_marginals() as loaded from its saved CSV,
        or a path to that CSV file.
 
    Returns
    -------
    labels : list of str
        Age group labels inherited from the ACS margins file.
    mapping : dict
        Integer-keyed dict {group_idx: lambda} compatible with
        build_grouped_meta's STRATIFICATION_SPEC mapping format.
    """

    if isinstance(age_margins, (str, Path)):
        age_margins = pd.read_csv(age_margins)
    
    label_col = [c for c in age_margins.columns if c != "pop"][0]
    labels = age_margins[label_col].tolist()
 
    # derive split points from lower bounds of each non-first label
    # e.g. '18-24' -> 18, '67+' -> 67
    split_ages = []
    for label in labels[1:]:
        label = label.strip()
        if label.endswith("+"):
            split_ages.append(int(label[:-1]))
        elif "-" in label:
            split_ages.append(int(label.split("-")[0]))
        elif label.startswith("<"):
            split_ages.append(int(label[1:]))
 
    boundaries = [0] + split_ages + [float("inf")]
    mapping = {}
    for i in range(len(labels)):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i == 0:
            mapping[i] = lambda x, hi=hi: x < hi
        elif i == len(labels) - 1:
            mapping[i] = lambda x, lo=lo: x >= lo
        else:
            mapping[i] = lambda x, lo=lo, hi=hi: lo <= x < hi
 
    return labels, mapping



# ==================== FUNCTIONS FOR PROCESSING RAW ATUS DATA ====================


def get_resp(resp_file, cps_file, years, cbsa_filter=None, age_min=None):
    """
    Returns a cleaned df of respondents from respondent file with characteristics from CPS file.

    - resp_file : dir to respondent file
    - cps_file : dir to CPS file
    - years : a tuple of year limits, e.g. (2004, 2019)
    - cbsa_filter : (optional) a list of CBSA codes to filter respondents by
    - age_min : (optional) an integer of minimum respondent age
    """

    resp_cols = [
        'TUCASEID', 
        'TUFNWGTP', 
        'TUYEAR', 
        'TUMONTH',
        'TUDIARYDATE', 
        'TUDIARYDAY', 
        'TRDPFTPT'
    ]
    df = pd.read_csv(resp_file, usecols=resp_cols)

    cps_cols = [
        'TUCASEID', 
        'TULINENO',
        'HRYEAR4',
        'GTMETSTA', 
        'GTCBSA', 
        'GESTFIPS', 
        'GTCO',
        'PRTAGE',
        'HEFAMINC',
        'HUFAMINC',
        'PEHSPNON', 
        'PTDTRACE'
    ]
    cps = pd.read_csv(cps_file, usecols=cps_cols, dtype={'GESTFIPS': str, 'GTCO': str})
    cps = cps.query('TULINENO == 1 and GTMETSTA == 1')

    df = df.merge(cps, on='TUCASEID', how='left')
    df = df.query(f'TUYEAR in {years} and GTMETSTA == 1')

    if cbsa_filter is not None:
        df = df.query(f'GTCBSA in {cbsa_filter}').reset_index(drop=True)

    if age_min is not None:
        df = df.query(f'PRTAGE >= {age_min}').reset_index(drop=True)

    df = clean_resp_df(df)
    return df


def clean_resp_df(df):
    """
    Internal func to recode variables in respondent file.
    """

    def classify_race(row):
        if row['PTDTRACE'] == 1 and row['PEHSPNON'] == 2:
            return 1
        if row['PTDTRACE'] == 2 and row['PEHSPNON'] == 2:
            return 2
        if row['PTDTRACE'] not in [1, 2] and row['PEHSPNON'] == 2:
            return 3
        if row['PEHSPNON'] == 1:
            return 4
        return pd.NA
    
    def classify_income_group(row):
        # NOTE: confirm cutoff year — your original comment/code conflicted
        if row['HRYEAR4'] < 2010:
            return row['HUFAMINC']
        else:
            return row['HEFAMINC']

    df = df.copy()

    df['TUDIARYDAY'] = df['TUDIARYDAY'].apply(
        lambda x: 'weekday' if x in [2, 3, 4, 5, 6] else 'weekend'
    )
    df['TUFINLWGT'] = df['TUFNWGTP'] / 365
    df['GTCBSA'] = df['GTCBSA'].astype(int)

    mask = df['GTCO'] != "000"
    df.loc[mask, 'GTCO'] = df.loc[mask, 'GESTFIPS'] + df.loc[mask, 'GTCO']

    df['INCOME_GROUP'] = df.apply(classify_income_group, axis=1)
    df['AGE'] = df['PRTAGE']
    df['EMPLOY_TYPE'] = df['TRDPFTPT'].apply(lambda x: x if x != -1 else 3)
    df['RACE'] = df.apply(classify_race, axis=1)

    cols_to_keep = [
        'TUCASEID',
        'TUYEAR',
        'TUMONTH',
        'TUDIARYDATE',
        'TUDIARYDAY',
        'TUFINLWGT',
        'GTCBSA',
        'GESTFIPS',
        'GTCO',
        'INCOME_GROUP',
        'AGE',
        'RACE',
        'EMPLOY_TYPE',
    ]
    return df[cols_to_keep]


def get_diaries(act_file, resp_df):

    # load data and merge with resp_df and who_df
    act_cols = ['TUCASEID','TUACTIVITY_N','TUACTDUR24','TUSTARTTIM','TRTIER1P','TRTIER2P','TEWHERE']
    diaries = pd.read_csv(act_file, usecols=act_cols).merge(resp_df[['TUCASEID','TUYEAR','TUMONTH','TUDIARYDATE','TUDIARYDAY','TUFINLWGT']], on='TUCASEID')
    diaries = diaries.drop_duplicates(subset=['TUCASEID','TUACTIVITY_N'], keep='first')
    diaries.rename(columns={'TRTIER1P':'TUTIER1CODE','TRTIER2P':'TUTIER2CODE'}, inplace=True)
    
    # extract only day of month from TUDIARYDATE
    diaries['TUDIARYDATE'] = diaries['TUDIARYDATE'].apply(lambda x: int(str(x)[-2:]))

    # extract activity duration in whole hours and minutes
    diaries['dur_h'] = (diaries['TUACTDUR24'] / 60).astype(int)
    diaries['dur_m'] = (diaries['TUACTDUR24'] - (diaries['dur_h'] * 60)).astype(int)

    # calculate activity start time in datetime format (date default to 1900-01-01)
    diaries['start_time'] = pd.to_datetime(diaries[['TUSTARTTIM']] #TUYEAR','TUMONTH','TUDIARYDATE', 
                .astype(str).apply(' '.join, 1), format='%H:%M:%S') #'%Y %m %d %H:%M:%S'

    # if activity start time is before 4AM, means that activity began the next day
    diaries['next_day'] = (diaries['start_time'].dt.time < datetime.datetime.strptime('04:00','%H:%M').time())
    # if this is the case, offset date by 1 day
    diaries['start_time'] = diaries.apply(lambda x: x['start_time'] + pd.DateOffset(days=1) if x['next_day'] else x['start_time'], axis=1)
    # calculate stop time in datetime format by offsetting start time by activity duration
    diaries['stop_time'] = diaries.apply(lambda row: row['start_time'] + pd.DateOffset(hours=row['dur_h'], minutes=row['dur_m']), axis=1)
    
    # clean columns
    cols_to_keep = ['TUCASEID','TUFINLWGT','TUYEAR','TUDIARYDAY','start_time','stop_time','TUACTDUR24','TUACTIVITY_N','TUTIER1CODE','TUTIER2CODE','TEWHERE']
    diaries = diaries[cols_to_keep]
    
    return diaries


def map_tewhere(mapping, diaries):
    diaries = diaries.copy()

    mask = (diaries['TEWHERE'] == -1) & (diaries['TUTIER1CODE'] == 1)
    diaries.loc[mask, 'TEWHERE'] = 1

    diaries['TEWHERE'] = (
        diaries['TEWHERE']
        .replace(-1, pd.NA)
        .groupby(diaries['TUCASEID'])
        .bfill()
    )
    diaries['WHERE'] = diaries['TEWHERE'].map(mapping)

    return diaries



# ==================== FUNCTIONS TO CLEAN UP METADATA IN THE END ====================


def group_codes(value, mapping, default=None):
    for group_code, condition in mapping.items():
        if callable(condition):
            if condition(value):
                return group_code
        else:
            if value in condition:
                return group_code
    return default


def make_joint_code_from_cols(df, vars_, group_specs):
    code = df[vars_[0]].copy()
    for var in vars_[1:]:
        code = code * len(group_specs[var]["labels"]) + df[var]
    return code


def build_grouped_meta(
    df,
    base_cols,
    group_specs,
    cluster_labels=None,
    joint_vars=None,
):
    out = df[base_cols].copy()

    for new_var, spec in group_specs.items():
        out[new_var] = df[spec["source_col"]].apply(
            lambda x: group_codes(x, spec["mapping"])
        )

        if "labels" in spec and spec["labels"] is not None:
            out[f"{new_var}_cat"] = out[new_var].apply(
                lambda x: spec["labels"][x] if pd.notna(x) else pd.NA
            )

    if joint_vars is not None:
        out["joint"] = make_joint_code_from_cols(out, joint_vars, group_specs)

    if cluster_labels is not None:
        out["cluster_label"] = cluster_labels

    return out



# ==================== FUNCTIONS FOR CREATING ATUS SEQUENCES ====================

def stage_diaries(diaries):
    # TODO:
    # - chunk diaries into multiple tasks to minimize memory requirement for a large sample
    # - consider parallelizing these chunks
    # get start datetime using group['start_time'][0]
    start = diaries.query('TUACTIVITY_N == 1')['start_time'].values[0]

    # get end datetime using pd.DateOffset(days=1)
    end = start + pd.DateOffset(hours=23, minutes=59)

    # create a daterange interval dataframe (1-minute increment)
    seq = pd.DataFrame(pd.date_range(start=start, end=end, freq='1min'), columns=['start_time'])
    seq['stop_time'] = seq['start_time'] + pd.DateOffset(minutes=1)
    #seq['TUCASEID'] = caseid

    # merge with group and filter for overlaps
    seq = diaries.merge(seq, how='cross', suffixes=('_tu','_seq'))
    filter = seq['start_time_seq'].ge(seq['start_time_tu']) & seq['stop_time_seq'].le(seq['stop_time_tu'])
    seq = seq[filter].sort_values(by=['TUCASEID','TUACTIVITY_N']).reset_index(drop=True)

    return seq


def stage_sequence(staged_diaries, seq_col=None, fill_func=None):

    # TODO:
    # - consider parallelizing staging

    """
    fill_func takes in staged diaries dataframe at minute t and a series of containing zeros of length diaries
    and returns a series containing values (e.g. 0 or 1, or other states like "A", "B")
    """

    def make_zeros_df(TUCASEIDs, fill_value=None):
        return pd.DataFrame(fill_value, index=TUCASEIDs, columns=range(1440), dtype=object)

    # get start datetime using group['start_time'][0]
    start = staged_diaries.query('TUACTIVITY_N == 1')['start_time_tu'].values[0]
    # get end datetime using pd.DateOffset(days=1)
    end = start + pd.DateOffset(hours=23, minutes=59)
    # increment of 1min
    ti = pd.date_range(start=start, end=end, freq='1min')

    caseid = staged_diaries['TUCASEID'].drop_duplicates()
    seq = make_zeros_df(caseid)
    for i,t in enumerate(ti):
        st = staged_diaries[(staged_diaries['start_time_seq'] == t)]
        if fill_func is not None:
            ser = fill_func(st, seq.loc[:,i])
            seq.loc[:,i] = ser
        else:
            val = st[seq_col].values
            seq.loc[:,i] = val

    return seq


def sequence(staged, T, nan_policy='omit', state_order=None):
    """
    Downsample a 1-minute staged sequence to T-minute cells via modal state.

    state_order : sequence of states, optional
        For string-valued input, explicit priority used to build the
        string→int mapping before scipy.stats.mode. The first entry becomes
        int 0 and wins ties over later entries. Observed states not in
        state_order are appended in sorted order after it.
        Defaults to alphabetical.

    Note (2026-04-23): prior to adding `state_order`, string-valued input
    was always mapped via sorted(unique(states)), meaning mode()'s tie-break
    priority followed the alphabetical order of label names. For the 3-cat
    alphabet ['Home', 'Work', 'Other'] that put Work *after* Other in int
    space, so Other won ties over Work — giving different downsampled
    sequences from the int-coded walkthrough_03 pipeline on the same data.
    """

    if T == 1:
        return

    # convert staged diaries to np array
    staged_arr = staged.to_numpy()

    # if sequence states are of type strings,
    # temporarily convert it to integer states
    # so that downsampling will be simpler
    str_states = (staged_arr.dtype == object)
    if str_states:

        # get unique states in array
        if state_order is not None:
            explicit = list(state_order)
            extras = sorted(set(np.unique(staged_arr)) - set(explicit))
            uniq = explicit + extras
        else:
            uniq = sorted(np.unique(staged_arr))
        uniq_lookup1 = {}
        uniq_lookup2 = {}

        # create 2-way mappings between string and corresponding int states
        for i, u in enumerate(uniq):
            uniq_lookup1[u] = i
            uniq_lookup2[i] = u
        map1 = np.vectorize(uniq_lookup1.get)
        map2 = np.vectorize(uniq_lookup2.get)

        # map string states to integer states
        staged_arr = map1(staged_arr)
    
    # initialize an empty array of size (N, 1440/T)
    n, t = np.shape(staged_arr)
    w = np.arange(0, t+1, T)
    w = list(zip(w[0:], w[1:]))
    seq = np.empty((n,int(t/T)), dtype=int)
    # for each sequence, downsample from 1 minute intervals to T minutes
    # by taking the modal state in each T-minute increment
    for i, (w1,w2) in enumerate(w):
        seq[:, i] = mode(staged_arr[:, w1:w2], axis=1, keepdims=False, nan_policy=nan_policy)[0]
    
    if str_states:
        seq = map2(seq)

    return pd.DataFrame(seq, index=staged.index)



# ==================== WRAPPERS FOR QUICKSTART NOTEBOOK ====================

def load(resp_file, cps_file, act_file, tewhere_map,
         years, age_min=18, cbsa_filter=None):
    """
    Streamlines loading and cleaning ATUS respondent and activity data
    then applies the TEWHERE mapping.

    This function calls get_resp + get_diaries + map_tewhere

    Parameters
    ----------
    resp_file, cps_file, act_file : path-like
        ATUS respondent, CPS, and activity data files.
    tewhere_map : dict (or defaultdict)
        Maps TEWHERE codes to activity alphabet (ints or labels).
        Pass a defaultdict to include fallback for unmapped codes.
    years : list[int]
    age_min : int
    cbsa_filter : list[int] or None

    Returns
    -------
    resp : pd.DataFrame
    diaries : pd.DataFrame
        Includes a 'WHERE' column with mapped activity states.
    """
    resp = get_resp(resp_file, cps_file, years, cbsa_filter=cbsa_filter, age_min=age_min)
    diaries = get_diaries(act_file, resp)
    diaries = map_tewhere(tewhere_map, diaries)
    return resp, diaries


def stratify(resp, income_margin, age_margin, row_var='income', col_var='age'):
    """
    Stratify ATUS respondents onto the income and age groups as defined by ACS margins.

    Auto-derives ATUS→ACS income and age group mappings if no override is passed;
    the derived labels are printed.

    Parameters
    ----------
    resp : pd.DataFrame
        Cleaned respondent DataFrame from `atus.load`.
    income_margin, age_margin : pd.DataFrame
        CBSA-level margin DataFrames produced by `acs.process_cbsa`.
    row_var, col_var : str
        Names to use for the stratified columns in the output (default 'income', 'age').

    Returns
    -------
    atus_meta : pd.DataFrame
        Stratified respondent metadata with `row_var`, `col_var`, joint code, plus base_cols.
    group_maps : dict
        {row_var: {'labels': [...]}, col_var: {'labels': [...]}}
    """

    base_cols = ['TUCASEID', 'TUFINLWGT', 'GTCBSA']

    income_labels, income_mapping = compute_atus_income_group_mapping(income_margin)
    print(f'auto-derived ATUS income groups: {income_labels}')

    age_labels, age_mapping = compute_atus_age_group_mapping(age_margin)
    print(f'auto-derived ATUS age groups: {age_labels}')

    group_specs = {
        row_var: {
            'source_col': 'INCOME_GROUP',
            'mapping': income_mapping,
            'labels': income_labels,
        },
        col_var: {
            'source_col': 'AGE',
            'mapping': age_mapping,
            'labels': age_labels,
        },
    }

    atus_meta = build_grouped_meta(
        df=resp,
        base_cols=base_cols,
        group_specs=group_specs,
        joint_vars=[row_var, col_var],
    )

    group_maps = {
        row_var: {'labels': income_labels},
        col_var: {'labels': age_labels},
    }
    return atus_meta, group_maps


def select_feature_subset(metrics_df, feature_subset, all_labels):
    """
    Select a subset of feature columns from a compute_metrics_for_all_sequences output.

    feature_subset : one of
        'all'          → return every column unchanged
        'metrics_only' → just num_activities, turnover_rate, reciprocity
        'tu_only'      → just the time-use duration columns (one per label in all_labels)
        'edges_only'   → just the edge-transition columns (names starting with 'edge_')
    """
    if feature_subset == 'all':
        return metrics_df
    if feature_subset == 'metrics_only':
        cols = ['num_activities', 'turnover_rate', 'reciprocity']
        return metrics_df[[c for c in cols if c in metrics_df.columns]]
    if feature_subset == 'tu_only':
        cols = [c for c in all_labels if c in metrics_df.columns]
        return metrics_df[cols]
    if feature_subset == 'edges_only':
        cols = [c for c in metrics_df.columns if str(c).startswith('edge_')]
        return metrics_df[cols]
    raise ValueError(
        f'Unknown feature_subset {feature_subset!r}; '
        "expected one of 'all', 'metrics_only', 'tu_only', 'edges_only'"
    )


def build_sequences(diaries, T=30, state_order=None):
    """
    Produce a DataFrame of activity sequences from cleaned diaries.

    Calls stage_diaries() + stage_sequence(seq_col='WHERE') + sequence().
    Assumes diaries has a 'WHERE' column (produced by `atus.map_tewhere` or `atus.load`).

    state_order : sequence of states, optional
        Forwarded to `sequence`; controls tie-break priority when mode()
        downsamples string-valued sequences. Pass your
        SEQUENCE_METRIC_SPECS['all_labels'] to make tie-breaking follow the
        alphabet's intended priority (first label wins ties over later ones).

    Note (2026-04-23): added state_order. Without it, string-valued sequences
    used alphabetical priority inside `sequence`, which can cause discrepancies
    when string states are used vs integer states.

    Returns
    -------
    atus_seq : pd.DataFrame
        Shape (N_respondents, 24*60/T). Each row is one respondent-day's activity sequence.
    """

    staged_diaries = stage_diaries(diaries)
    staged_sequence = stage_sequence(staged_diaries, seq_col='WHERE')
    atus_seq = sequence(staged_sequence, T=T, state_order=state_order)
    return atus_seq


def cluster_sequences(atus_seq, weights, K, sequence_metric_specs,
                      seed=None, n_restarts=10):
    """
    Clusters sequences using weighted k-medoids, streamlining calculation of sequence metrics
    Returns atus metrics, cluster results, and medoid thresholds.

    The feature_subset in sequence_metric_specs is applied to the metrics DataFrame
    before distance computation; the full (unsubsetted) metrics are returned so
    that `mobility.distance_to_atus` can re-apply the same subset.

    This function calls
        - compute_metrics_for_all_sequences()
        - metrics_cosine_D()
        - fit_weighted_kmedoids_with_restarts()
        - compute_dist_to_medoid_thresholds()

    Parameters
    ----------
    atus_seq : pd.DataFrame
        Output of `atus.build_sequences`.
    weights : array-like, shape (N,)
        Survey weights per respondent (e.g. from atus_meta['TUFINLWGT']).
    K : int
        Number of clusters.
    sequence_metric_specs : dict with keys
        'home_label', 'work_label', 'all_labels', 'feature_subset'.
    seed : int or None
        Random seed for the k-medoids restarts.
    n_restarts : int

    Returns
    -------
    atus_metrics : pd.DataFrame
        Full feature table indexed like atus_seq (not subsetted).
    cluster_results : dict
        {'medoids', 'labels', 'inertia', 'seed'} from
        fit_weighted_kmedoids_with_restarts.
    medoid_thresholds : dict
        {cluster_label: distance_threshold} at the 99th percentile.
    """

    from mobcalibrate.preprocessing import (
        compute_metrics_for_all_sequences,
        metrics_cosine_D,
        fit_weighted_kmedoids_with_restarts,
        compute_dist_to_medoid_thresholds,
    )

    home_label = sequence_metric_specs['home_label']
    work_label = sequence_metric_specs['work_label']
    all_labels = sequence_metric_specs['all_labels']
    feature_subset = sequence_metric_specs.get('feature_subset', 'all')

    # full feature table — returned to caller and used downstream by mobility.distance_to_atus
    atus_metrics = compute_metrics_for_all_sequences(
        atus_seq.values,
        home_label=home_label,
        work_label=work_label,
        all_labels=all_labels,
    )
    atus_metrics.index = atus_seq.index

    # subset applied to clustering distance matrix
    metrics_for_dist = select_feature_subset(atus_metrics, feature_subset, all_labels)

    D = metrics_cosine_D(metrics_for_dist, metrics_for_dist)

    cluster_results = fit_weighted_kmedoids_with_restarts(
        D, wA=np.asarray(weights), K=K,
        n_restarts=n_restarts, #random_state=seed,
    )

    medoid_thresholds = compute_dist_to_medoid_thresholds(
        D, cluster_results['medoids'], cluster_results['labels'],
    )

    return atus_metrics, cluster_results, medoid_thresholds