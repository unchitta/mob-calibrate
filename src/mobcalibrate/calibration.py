import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple, Dict, Literal

from .utils import make_joint_code
from .core import *




@dataclass
class Calibrator:
    # optional: include mobility user IDs in outputs
    unit_ids: Optional[Union[np.ndarray, list]] = None

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
    mode: Literal["census_only", "behavioural_full", "behavioural_only", "all_with_behavioural_only", "all"] = "behavioural_full"

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

        num_data_cols = 6  # weight1, weight_behav, weight2, sampled_row_code, sampled_col_code, joint_strata_code
        self.weights_data = np.empty((tot_replicates, len(self.home_cbgs), num_data_cols), dtype=np.int64)


    def _create_weights(self, rng, replicate=0, mode="all"):      

        # sample demographics from ACS CBG probabilities
        sampled_row_codes = sample_codes_from_cdf(self.cdf_row_aligned, rng)
        sampled_col_codes = sample_codes_from_cdf(self.cdf_col_aligned, rng)

        # check which stages we need to compute
        need_stage2 = mode in ["behavioural_full", "all_with_behavioural_only", "all"]
        need_behavioural_only = (mode == "behavioural_only")
        need_behavioural = (mode == "all_with_behavioural_only")

        # initialize vectors
        weights_stage1 = None
        weights_behavioural = None
        weights_stage2 = None

        # need only behavioural, no need to compute stage1
        if need_behavioural_only:
            weights_stage1 = np.full(len(self.assigned_cluster_labels), 1.0 / len(self.assigned_cluster_labels))
            
            # calibrate stage 2 weights
            weights_behavioural, diagnostics_stage2 = stage2_rake(
                sampled_row_codes,
                sampled_col_codes,
                self.num_row_cats,
                self.num_col_cats,
                self.assigned_cluster_labels,
                weights_stage1,
                self.atus_target_table,
                factor_clip=None
            )
            
        # need to compute stage1
        else:
            # calibrate stage 1 weights
            weights_stage1, diagnostics_stage1 = stage1_ipf(sampled_row_codes, sampled_col_codes, self.acs_row_margin,
                                                            self.acs_col_margin, n_iter=10)
            
            # also need stage2
            if need_stage2:
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
            
            # need behavioural along with stage1
            if need_behavioural:

                weights_tmp = np.full(len(self.assigned_cluster_labels), 1.0 / len(self.assigned_cluster_labels))
            
                # calibrate stage 2 weights
                weights_behavioural, diagnostics_stage2 = stage2_rake(
                    sampled_row_codes,
                    sampled_col_codes,
                    self.num_row_cats,
                    self.num_col_cats,
                    self.assigned_cluster_labels,
                    weights_tmp,
                    self.atus_target_table,
                    factor_clip=None
                )

        # save weights and associated data from this replicate if they exist
        # otherwise the dataframe defaults to empty
        if weights_stage1 is not None:
            self.weights_data[replicate, :, 0] = np.rint(weights_stage1 * self.target_pop_tot).astype(int)
        if weights_behavioural  is not None:
            self.weights_data[replicate, :, 1] = np.rint(weights_behavioural * self.target_pop_tot).astype(int)
        if weights_stage2 is not None:
            self.weights_data[replicate, :, 2] = np.rint(weights_stage2 * self.target_pop_tot).astype(int)
        self.weights_data[replicate, :, 3] = sampled_row_codes
        self.weights_data[replicate, :, 4] = sampled_col_codes
        self.weights_data[replicate, :, 5] = make_joint_code(sampled_row_codes, sampled_col_codes, self.num_col_cats)
        

    def create_main_weights(self, mode="all"):
        s = self.seedseq[0]
        self.spawn_keys[0] = s.spawn_key
        rng = np.random.default_rng(s)
        self._create_weights(rng, replicate=0, mode=mode)
    
    def create_replicate_weights(self, mode="all"):
        if len(self.seedseq) == 1:
            print("num_replicates was set to 0 during initialization, returning without creating replicate weights")

        for r, s in enumerate(self.seedseq[1:]):
            self.spawn_keys[r+1] = s.spawn_key
            rng = np.random.default_rng(s)
            self._create_weights(rng, replicate=r+1, mode=mode)


    def weights_data_to_df(self, data, mode="behavioural_full"):
        col_names = np.array([
                'weight_census', 
                'weight_behavioural_only',
                'weight_behavioural_full', 
                'sampled_'+self.acs_row_var_name+'_code',
                'sampled_'+self.acs_col_var_name+'_code',
                'sampled_joint_stratum_code'
                ])

        # return asked weights only
        # index depends on needed weights
        #### TODO ####
        # the following should be modified so that weight1 and weight2
        # are always returned no matter but, while the underlying calculations change
        # also the naming is very confusing currently
        ##############
        if mode == "census_only":
            idx = [0,3,4,5]
        elif mode == "behavioural_full":
            idx = [2,3,4,5]
        elif mode == "behavioural_only":
            idx = [1,3,4,5]
        elif mode == "all":
            idx = [0,2,3,4,5]
        elif mode == "all_with_behavioural_only":
            idx = list(range(len(col_names)))
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
 
        df = pd.DataFrame(data[:, idx], columns=col_names[idx]).astype(int)
 
        if self.user_ids is not None:
            df.insert(0, 'unit_id', self.unit_ids)
 
        return df

    def get_main_weights(self, return_df=True, mode="behavioural_full"):
        data = self.weights_data[0, :, :]

        if return_df:
            return self.weights_data_to_df(data, mode=mode)
        else:
            return data
    

    def _get_weights_by_replicate_id(self, replicate_id, return_df=False, mode="behavioural_full"):
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
            return self.weights_data_to_df(data, mode=mode)
        else:
            return data
        

    def get_replicate_weights(self, return_df=False, mode="behavioural_full"):
        """
        if return_df = False, return a numpy array with shape 
        (num_replicates, N, weights_data_cols) 
        where the first dimension is the replicates

        if return_df = True, return list of dataframes, 
        where the first dataframe is the weights data from replicate 1
        """
        if return_df:
            return [self._get_weights_by_replicate_id(r, return_df=True, mode=mode) for r in range(1, self.num_replicates+1)]
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
