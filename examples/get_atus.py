import pandas as pd
from copy import deepcopy
import datetime
from collections import defaultdict

# v20250121


def get_resp(resp_file, cps_file, years, cbsa_filter=None, age_min=None):
    """
        Returns a cleaned df of respondents from respondent file with characteristics from CPS file.

        - resp_file : dir to respondent file
        - cps_file : dir to CPS file
        - years : a tuple of year limits, e.g. (2004, 2019)
        - cbsa_filter : (optional) a list of CBSA codes to filter respondents by
        - age_min : (optional) an integer of minimum respondent age

    """

    resp_cols = ['TUCASEID','TUFNWGTP','TUYEAR','TUMONTH','TUDIARYDATE','TUDIARYDAY',
                 'TRDPFTPT','TRCHILDNUM','TELFS','TRSPPRES', 'TRYHHCHILD']
    df = pd.read_csv(resp_file, usecols=resp_cols)

    cps_cols = ['TUCASEID','TULINENO','PEHSPNON', 'PESEX', 'PRCITSHP', 'PEEDUCA',
    'PRTAGE', 'PTDTRACE', 'GTMETSTA','GTCBSA','GESTFIPS','GESTFIPS','GTCO',
    'HRHTYPE','HEFAMINC','HUFAMINC','HRYEAR4','HRMONTH','PEMARITL'
    #'PEIO1ICD', 'PEIO1OCD','PRDTIND1','PRDTOCC1', 'PRFAMTYP', 'PRMJOCGR'
    ]
    cps = pd.read_csv(cps_file, usecols=cps_cols, dtype={'GESTFIPS':str,'GTCO':str})
    cps = cps.query('TULINENO==1 and GTMETSTA==1')
    df = df.merge(cps, on='TUCASEID', how='left')
    df = df.query(f'TUYEAR in {years} and GTMETSTA==1')
    if cbsa_filter is not None:
        df = df.query(f'GTCBSA in {cbsa_filter}').reset_index(drop=True)
    if age_min is not None:
        df = df.query(f'PRTAGE >= {age_min}').reset_index(drop=True)

    df = clean_resp_df(df)

    return df



def clean_resp_df(df):

    """
        Internal func to recode variables in respondent file.

        Race:
            - 1 (NH White), 2 (NH Black), 3 (NH Other), 4 (Hisp)

    """

    def classify_race(x):
        if x[0]==1 and x[1]==2:
            return 1 # NH WHITE
        if x[0]==2 and x[1]==2:
            return 2 # NH BLACK
        if (x[0] not in [1,2]) and x[1]==2:
            return 3 # OTHER
        if x[1] == 1:
            return 4 # HISP

    
    def classify_income_group(x):
        income_level = x[1] if x[0] > 2009 else x[2] # if year > 2019 use HEFAMINC o/w use HUFAMINC
        
        # if income_level in list(range(1,7)):
        #     return 1 #income group 1 : 0-49,999
        # if income_level in list(range(7,12)):
        #     return 2 #income group 2 : 20,000 - 49,999
        # if income_level in list(range(12,15)):
        #     return 3 #incmoe group 3: 50k - 99,999
        # if income_level == 15:
        #     return 4
        # if income_level == 16:
        #     return 5
        # if income_level < 0:
        #     return 6 # missing income
        
        return income_level

    def classify_mar_child(x):
        if x[0] and not x[1]:
            return 2 # married without child
        if not x[0] and x[0]:
            return 3 # single with child
        if x[0] and x[1]:
            return 4 # married with child
        
        return 1 # else, single without child        
        
    df = deepcopy(df)
    
    cols = ['TUCASEID','TUYEAR','TUMONTH']
    df['TUDIARYDAY'] = df['TUDIARYDAY'].apply(lambda x: 'weekday' if x in [2,3,4,5,6] else 'weekend')
    df['TUDIARYDATE'] = df['TUDIARYDATE']
    df['TUFINLWGT'] = df['TUFNWGTP'] / 365
    df['GTCBSA'] = df['GTCBSA'].astype(int)
    df['GTCO'] = df[['GESTFIPS','GTCO']].apply(lambda x: x[0]+x[1] if x[1]!="000" else x[1], axis=1)
    df['FEMALE'] = (df['PESEX'] == 2).astype(int)
    df['AGE'] = df['PRTAGE']
    df['EMPLOYED'] = df['TRDPFTPT'].apply(lambda x: 1 if x!=-1 else 0)
    df['EMPLOY_TYPE'] = df['TRDPFTPT'].apply(lambda x: x if x!=-1 else 3)
    df['INCOME_GROUP'] = df[['TUYEAR','HEFAMINC','HUFAMINC']].apply(lambda x: classify_income_group(x), axis=1)
    df['COLLEGE'] = (df['PEEDUCA'] >= 43).astype(int) # 0=no bachelors; 1=bachelors or above
    df['MAR'] = (df['PEMARITL'] <= 2).astype(int) # 0=single/widowed/divocerd/separated; #1=married with/without spouse present
    df['CHILDREN'] = (df['TRCHILDNUM'] > 0).astype(int)
    df['MAR_CHILD'] = df[['MAR','CHILDREN']].apply(lambda x: classify_mar_child(x), axis=1)
    df['RACE'] = df[['PTDTRACE','PEHSPNON']].apply(lambda x: classify_race(x), axis=1)

    cols_to_keep = ['TUCASEID', 'TUYEAR', 'TUMONTH', 'TUDIARYDAY', 'TUDIARYDATE','GTCBSA','GTCO','TUFINLWGT']
    cols_to_keep += df.columns[-10:].tolist()
    df = df[cols_to_keep]
    
    return df


def get_who_df(who_file, resp_df):

    who_df = (pd.read_csv(who_file)
            .query('TUWHO_CODE not in [-1,-2,-3, 18,19]')
            .drop(columns=['TRWHONA','TULINENO'])
            .merge(resp_df[['TUCASEID']], on='TUCASEID')
            )

    whocode_lookup = {
            'HHFAMILY': [20,21,22,23,24,25,26,27],
            'NHFAMILY': [40,51,52,53],
            'FRIEND': [54],
            'CCC': [55, 59,60,61,62],
            'OTHER': [56,57,58]
        }

    who_types = list(whocode_lookup.keys())

    for who_type in who_types:
        who_df[who_type] = who_df['TUWHO_CODE'].apply(
            lambda x: 1 if x in whocode_lookup[who_type] else 0
            )

    who_df = who_df.groupby(['TUCASEID','TUACTIVITY_N'], as_index=False)[who_types].sum()
    who_df[who_types] = who_df[who_types].stack().apply(lambda x: 1 if x>=1 else 0).unstack()

    def group_who(row):

        if row['HHFAMILY'] == 1:
            return 'HHFAMILY' # with hh family only
        elif row['NHFAMILY'] == 1:
            return 'NHFAMILY'
        elif row['FRIEND'] == 1:
            return 'FRIEND'
        elif row['CCC'] == 1:
            return 'CCC'
        elif row['OTHER'] == 1:
            return 'OTHER'

            
    who_df['WITH_WHO'] = who_df[who_types].apply(lambda row: group_who(row), axis=1)
    
    return who_df

def get_diaries(act_file, resp_df, who_df=None):

    # load data and merge with resp_df and who_df
    act_cols = ['TUCASEID','TUACTIVITY_N','TUACTDUR24','TUSTARTTIM','TRTIER1P','TRTIER2P','TEWHERE']
    diaries = pd.read_csv(act_file, usecols=act_cols).merge(resp_df[['TUCASEID','TUYEAR','TUMONTH','TUDIARYDATE','TUDIARYDAY','TUFINLWGT']], on='TUCASEID')
    if who_df is not None:
        diaries = pd.merge(diaries, who_df, on=['TUCASEID','TUACTIVITY_N'], how='left')
    diaries = diaries.drop_duplicates(subset=['TUCASEID','TUACTIVITY_N'], keep='first')
    #diaries['WITH_WHO'] = diaries['WITH_WHO'].fillna('ALONE') # alone/NA = Z
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
    cols_to_keep = ['TUCASEID','TUFINLWGT','TUYEAR','TUDIARYDAY','start_time','stop_time','TUACTDUR24','TUACTIVITY_N','TUTIER1CODE','TUTIER2CODE','TEWHERE'] + list(who_df.columns[-5:])
    diaries = diaries[cols_to_keep]
    
    return diaries


def map_tewhere(mapping, diaries):
    diaries = deepcopy(diaries)
    #tewhere_map = defaultdict(lambda: "L", mapping)

    # for unreported TEWHERE for personal care activities, assume done at home
    tewhere_fill_home = diaries.query('TEWHERE==-1 and TUTIER1CODE==1').index
    diaries.loc[tewhere_fill_home,'TEWHERE'] = 1

    # impute other NAs backward
    diaries['TEWHERE'] = diaries['TEWHERE'].replace(to_replace=-1, method='bfill')
    diaries['WHERE'] = diaries['TEWHERE'].apply(lambda x: mapping[x])
    
    return diaries#.drop(columns=['TEWHERE'])