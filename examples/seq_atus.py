import pandas as pd
import numpy as np
from scipy.stats import mode




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

    def make_zeros_df(TUCASEIDs):
        n = len(TUCASEIDs)
        return pd.DataFrame(np.zeros((n,1440)), index=TUCASEIDs)

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


def sequence(staged, T, nan_policy='omit'):

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




"""
def sequence(diaries, staged_diaries, seq_col, seq_width=5):
    
    # format datetimes and upsample each activity into increment of 1 minute
    #diaries = deepcopy(diaries)
    
    sequences = []
    
    # group by TUCASEID and iterate; for each group:
    for caseid, grouped in staged_diaries.groupby('TUCASEID'):

        # resample into intervals of T mins
        seq = grouped.resample(f'{seq_width}min', on='start_time_seq')['TUACTIVITY_N']

        # sort by counts TUACTIVITY_N and select highest and create a sequence
        # this select the activity that takes the longest out of each 
        # T-minute increment to fill that sequence position
        counts = seq.value_counts()
        counts.name = 'duration'
        counts = counts.reset_index(level=1)
        counts = counts.sort_values(by=['start_time_seq','duration'], 
                                    ascending=[True,False])
        seq = counts.groupby('start_time_seq',as_index=False)[['TUACTIVITY_N']].first()

        # merge with seq_col (this is the value to fill the sequences)
        # e.g. could be 'WHERE', or 'WITH_WHO', or 'WITH_FRIEND', etc
        # transpose/convert sequence columns to get flatted sequence, then store in list
        # NOTE: store TUCASEID to use as sequence identifiers later
        seq = seq.merge(diaries[['TUACTIVITY_N', seq_col]], on='TUACTIVITY_N')[[seq_col]].T
        seq.index = [caseid]
        #if merge_char is not None:
        #      seq = seq.merge(merge_char, left_index=True, right_index=True, how='left')
        sequences.append(seq)

    # concat sequences in the respective lists and save to csv
    sequences = pd.concat(sequences)
    return sequences
"""
