import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple, Dict

from .utils import make_joint_code
from .core import *




@dataclass
class Calibrator:
    # Pre-filterd mobility inputs
    home_cbgs: Union[np.ndarray, list]
    assigned_cluster_labels: Union[np.ndarray, list]

    # Calibration inputs/constraints
    acs_cbg_probs_df: pd.DataFrame
    acs_row_var_name: str
    acs_col_var_name: str
    acs_row_cats: Union[np.ndarray, list]
    acs_col_cats: Union[np.ndarray, list]
    acs_row_margin: Union[np.ndarray, list]
    acs_col_margin: Union[np.ndarray, list]
    atus_target_table: Union[pd.DataFrame, np.ndarray]
    target_pop_tot: int
    geoid_col: str = 'GEOID'
    num_replicates: int = 160

    # RNG
    seed: Optional[int] = None
    seedseq: List[np.random.SeedSequence] = field(init=False)
    #spawn_keys: List[Tuple[int, ...]] = field(default_factory=list)
    spawn_keys: Dict = field(default_factory=dict)
    #rng: np.random.Generator = field(init=False)

    # (For internal cache)
    num_row_cats: int = field(init=False)
    num_col_cats: int = field(init=False)
    num_strata: int = field(init=False)
    num_clusters: int = field(init=False)
    cdf_row_aligned: np.ndarray = field(init=False)
    cdf_col_aligned: np.ndarray = field(init=False)
    weights_data: np.ndarray = field(init=False)
    

    def __post_init__(self):
        tot_replicates = self.num_replicates + 1
        ss = np.random.SeedSequence(self.seed)
        self.seedseq = ss.spawn(tot_replicates)
        #self.rng = np.random.default_rng(self.seed)

        self.num_row_cats = len(self.acs_row_cats)
        self.num_col_cats = len(self.acs_col_cats)
        self.num_strata = self.atus_target_table.shape[0]
        self.num_clusters = self.atus_target_table.shape[1]

        acs_geoids, cdf_row, cdf_col = build_cdfs(self.acs_cbg_probs_df, self.acs_row_cats, self.acs_col_cats, geoid_col=self.geoid_col)
        cdf_row_aligned, cdf_col_aligned = align_cdfs_to_home_cbgs(self.home_cbgs, acs_geoids, cdf_row, cdf_col)
        self.cdf_row_aligned = cdf_row_aligned
        self.cdf_col_aligned = cdf_col_aligned

        num_data_cols = 5  # weight1, weight2, sampled_row_code, sampled_col_code, joint_strata_code
        self.weights_data = np.empty((tot_replicates, len(self.home_cbgs), num_data_cols), dtype=np.int64)


    def _create_weights(self, rng, replicate=0):

        # sample demographics from ACS CBG probabilities
        sampled_row_codes = sample_codes_from_cdf(self.cdf_row_aligned, rng)
        sampled_col_codes = sample_codes_from_cdf(self.cdf_col_aligned, rng)

        # calibrate stage 1 weights
        weights_stage1, diagnostics_stage1 = stage1_ipf(sampled_row_codes, sampled_col_codes, self.acs_row_margin, self.acs_col_margin, n_iter=10)
        
        # calibrate stage 2 weights
        weights_stage2, diagnostics_stage2 = stage2_rake(
            sampled_row_codes,
            sampled_col_codes,
            self.num_row_cats,
            self.num_col_cats,
            self.assigned_cluster_labels,
            weights_stage1,
            self.atus_target_table,
            factor_clip=None
        )

        # save weights and associated data from this replicate
        self.weights_data[replicate, :, 0] = np.rint(weights_stage1 * self.target_pop_tot).astype(int)
        self.weights_data[replicate, :, 1] = np.rint(weights_stage2 * self.target_pop_tot).astype(int)
        self.weights_data[replicate, :, 2] = sampled_row_codes
        self.weights_data[replicate, :, 3] = sampled_col_codes
        self.weights_data[replicate, :, 4] = make_joint_code(sampled_row_codes, sampled_col_codes, self.num_col_cats)

    def create_main_weights(self):
        s = self.seedseq[0]
        self.spawn_keys[0] = s.spawn_key
        rng = np.random.default_rng(s)
        self._create_weights(rng, replicate=0)
    
    def create_replicate_weights(self):
        if len(self.seedseq) == 1:
            print("num_replicates was set to 0 during initialization, returning without creating replicate weights")

        for r, s in enumerate(self.seedseq[1:]):
            self.spawn_keys[r+1] = s.spawn_key
            rng = np.random.default_rng(s)
            self._create_weights(rng, replicate=r+1)


    def weights_data_to_df(self, data):
        col_names = [
                'weight1', 
                'weight2', 
                'sampled_'+self.acs_row_var_name+'_code',
                'sampled_'+self.acs_col_var_name+'_code',
                'sampled_joint_stratum_code'
                ]
        return pd.DataFrame(data, columns=col_names).astype(int)

    def get_main_weights(self, return_df=True):
        data = self.weights_data[0, :, :]
        if return_df:
            return self.weights_data_to_df(data)
        else:
            return data
    

    def _get_weights_by_replicate_id(self, replicate_id, return_df=False):
        """
        return the weights from the specified replicate along with sampled demographic codes.
        if return_df = True, return the weights as a dataframe
        
        replicate_id = 0 for main weights
        replicate_id range from 1 to num_replicates for replicate weights
        """

        data = self.weights_data[replicate_id, :, :]

        if replicate_id > self.num_replicates:
            print("replicate id exceeds the number of replicates specified during init; returning nothing")
            return
        
        if return_df:
            return self.weights_data_to_df(data)
        else:
            return data
        

    def get_replicate_weights(self, return_df=False):
        """
        if return_df = False, return a numpy array with shape 
        (num_replicates, N, weights_data_cols) 
        where the first dimension is the replicates

        if return_df = True, return list of dataframes, 
        where the first dataframe is the weights data from replicate 1
        """
        if return_df:
            return [self._get_weights_by_replicate_id(r, return_df=True) for r in range(1, self.num_replicates+1)]
        else:
            return self.weights_data[1:, :, :]
    

    def get_all_weights(self, return_df=False):
        if return_df:
            return [self._get_weights_by_replicate_id(r, return_df=True) for r in range(self.num_replicates+1)]
        else:
            return self.weights_data
    

    def get_seed_info(self):
        # return the initial seed number to generate a root seedsequence
        # as well as a dict of spawn key for each replicate
        return self.seed, self.spawn_keys
