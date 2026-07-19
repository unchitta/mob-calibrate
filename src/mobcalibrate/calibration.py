import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .core import (
    align_cdfs_to_home_cbgs,
    build_cdfs,
    sample_codes_from_cdf,
    stage1_ipf,
    stage1_rake,
    stage2_rake,
)
from .utils import make_joint_code


Mode = Literal['uniform', 'demographic', 'behavioral', 'demographic_behavioral']

_VALID_MODES = {'uniform', 'demographic', 'behavioral', 'demographic_behavioral'}
_MODES_NEEDING_STAGE1 = {'demographic', 'demographic_behavioral'}
_MODES_NEEDING_STAGE2 = {'behavioral', 'demographic_behavioral'}


@dataclass(frozen=True)
class CalibrationResult:
    """
    Immutable output for `Calibrator.create_weights`.

    The first dimension of `weights` / `sampled_codes` indexes replicates.
    Index 0 is the "main" estimate; indices 1..R-1 are variance replicates.
    """

    # per (replicate, unit) — int64, scaled to target_pop_tot and rounded
    weights: np.ndarray           # (R, N, 2) — [weight1, weight_final]
    sampled_codes: np.ndarray     # (R, N, 3) — [row, col, joint]

    # per unit
    unit_ids: np.ndarray          # (N,)

    # per replicate
    spawn_keys: Dict[int, Tuple]

    # per call
    mode: str
    seed: Optional[int]
    target_pop_tot: int
    row_var: str
    col_var: str
    row_cats: np.ndarray
    col_cats: np.ndarray

    # --- views ---

    def all_final_weights(self, to_df=True) -> np.ndarray:
        """(N, R) view of weight_final across all replicates; column 0 is the main."""
        w = self.weights[:, :, 1].T
        if to_df:
            w_columns = [f'weight_final_{r}' for r in range(self.num_replicates)]
            w = pd.DataFrame(w, columns=w_columns)
            w.insert(0, 'unit_id', self.unit_ids)
        return w

    def main_weights(self) -> np.ndarray:
        return self.weights[0]

    def replicate_weights(self) -> np.ndarray:
        return self.weights[1:]

    def all_weights(self) -> np.ndarray:
        return self.weights

    @property
    def num_replicates(self) -> int:
        return self.weights.shape[0]

    # --- materialize to DataFrame with metadata ---

    def to_df(self, replicate_id: int = 0) -> pd.DataFrame:
        w = self.weights[replicate_id]
        c = self.sampled_codes[replicate_id]
        return pd.DataFrame({
            'unit_id': self.unit_ids,
            'weight1': w[:, 0],
            'weight_final': w[:, 1],
            f'sampled_{self.row_var}_code': c[:, 0],
            f'sampled_{self.col_var}_code': c[:, 1],
            'sampled_joint_stratum_code': c[:, 2],
            'calibration_mode': self.mode
        })

    def to_long_df(self) -> pd.DataFrame:
        R, N, _ = self.weights.shape
        flat_w = self.weights.reshape(R * N, 2)
        flat_c = self.sampled_codes.reshape(R * N, 3)
        return pd.DataFrame({
            'replicate_id': np.repeat(np.arange(R), N),
            'unit_id': np.tile(self.unit_ids, R),
            'weight1': flat_w[:, 0],
            'weight_final': flat_w[:, 1],
            f'sampled_{self.row_var}_code': flat_c[:, 0],
            f'sampled_{self.col_var}_code': flat_c[:, 1],
            'sampled_joint_stratum_code': flat_c[:, 2],
            'calibration_mode': self.mode
        })

    # --- save/load results ---

    def save(self, path: Union[str, Path]) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CalibrationResult':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'expected CalibrationResult, got {type(obj).__name__}')
        return obj


@dataclass
class Calibrator:
    # unit IDs to identify mobility individuals or individual-days
    unit_ids: Optional[Union[np.ndarray, list]]

    # pre-filtered mobility inputs
    home_cbgs: Union[np.ndarray, list]
    assigned_cluster_labels: Union[np.ndarray, list]

    # calibration inputs/constraints
    acs_cbg_probs_df: pd.DataFrame
    acs_row_var_name: str
    acs_col_var_name: str
    acs_row_cats: Union[np.ndarray, list]
    acs_col_cats: Union[np.ndarray, list]
    target_pop_tot: int
    # stage 1 target: supply EITHER the joint OR both independent marginals.
    acs_joint_margin: Optional[Union[np.ndarray, list]] = None
    acs_row_margin: Optional[Union[np.ndarray, list]] = None
    acs_col_margin: Optional[Union[np.ndarray, list]] = None

    # required only for modes that run stage 2 (behavioral / demographic_behavioral)
    atus_target_table: Optional[Union[pd.DataFrame, np.ndarray]] = None

    geoid_col: str = 'GEOID'
    # total number of weight sets produced; index 0 is the point estimate
    num_replicates: int = 161

    # RNG
    seed: Optional[int] = None
    seedseq: List[np.random.SeedSequence] = field(init=False)

    # internal cache
    num_row_cats: int = field(init=False)
    num_col_cats: int = field(init=False)
    num_strata: int = field(init=False)
    num_clusters: int = field(init=False)
    cdf_row_aligned: np.ndarray = field(init=False)
    cdf_col_aligned: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.num_replicates < 1:
            raise ValueError('num_replicates must be >= 1 (index 0 is the point estimate)')

        ss = np.random.SeedSequence(self.seed)
        self.seedseq = ss.spawn(self.num_replicates)

        self.num_row_cats = len(self.acs_row_cats)
        self.num_col_cats = len(self.acs_col_cats)

        # stage-1 target: normalize joint to a flat (K_row*K_col,) array
        if self.acs_joint_margin is not None:
            jm = np.asarray(self.acs_joint_margin, dtype=np.float64)
            expected = self.num_row_cats * self.num_col_cats
            if jm.ndim == 2:
                if jm.shape != (self.num_row_cats, self.num_col_cats):
                    raise ValueError(
                        f'acs_joint_margin 2D shape {jm.shape} must be '
                        f'({self.num_row_cats}, {self.num_col_cats})'
                    )
                jm = jm.ravel()
            elif jm.ndim == 1:
                if jm.size != expected:
                    raise ValueError(
                        f'acs_joint_margin 1D length {jm.size} must be '
                        f'{expected} (= num_row_cats * num_col_cats)'
                    )
            else:
                raise ValueError(
                    f'acs_joint_margin must be 1D or 2D, got {jm.ndim}D'
                )
            self.acs_joint_margin = jm

        if self.atus_target_table is not None:
            self.num_strata = self.atus_target_table.shape[0]
            self.num_clusters = self.atus_target_table.shape[1]
        else:
            self.num_strata = 0
            self.num_clusters = 0

        acs_geoids, cdf_row, cdf_col = build_cdfs(
            self.acs_cbg_probs_df, self.acs_row_cats, self.acs_col_cats, geoid_col=self.geoid_col
        )
        cdf_row_aligned, cdf_col_aligned = align_cdfs_to_home_cbgs(
            self.home_cbgs, acs_geoids, cdf_row, cdf_col
        )
        self.cdf_row_aligned = cdf_row_aligned
        self.cdf_col_aligned = cdf_col_aligned


    def create_weights(self, mode: Mode = 'demographic_behavioral') -> CalibrationResult:
        if mode not in _VALID_MODES:
            raise ValueError(
                f'unknown mode {mode!r}; expected one of {sorted(_VALID_MODES)}'
            )
        if mode in _MODES_NEEDING_STAGE2 and self.atus_target_table is None:
            raise ValueError(
                f'mode={mode!r} runs stage-2 raking and requires atus_target_table'
            )
        if mode in _MODES_NEEDING_STAGE1:
            has_joint = self.acs_joint_margin is not None
            has_indep = self.acs_row_margin is not None and self.acs_col_margin is not None
            if has_joint == has_indep:
                raise ValueError(
                    f'mode={mode!r} runs stage-1; supply EITHER acs_joint_margin '
                    f'(for joint raking) OR both acs_row_margin and acs_col_margin '
                    f'(for IPF), but not both/neither'
                )

        R = self.num_replicates
        N = len(self.home_cbgs)
        weights = np.empty((R, N, 2), dtype=np.int64)
        sampled_codes = np.empty((R, N, 3), dtype=np.int64)
        spawn_keys: Dict[int, Tuple] = {}

        for r, s in enumerate(self.seedseq):
            spawn_keys[r] = s.spawn_key
            rng = np.random.default_rng(s)
            w1, w_final, row_codes, col_codes, joint_codes = self._compute_one_replicate(rng, mode)
            weights[r, :, 0] = np.rint(w1 * self.target_pop_tot).astype(np.int64)
            weights[r, :, 1] = np.rint(w_final * self.target_pop_tot).astype(np.int64)
            sampled_codes[r, :, 0] = row_codes
            sampled_codes[r, :, 1] = col_codes
            sampled_codes[r, :, 2] = joint_codes

        unit_ids = (
            np.asarray(self.unit_ids)
            if self.unit_ids is not None
            else np.arange(N)
        )

        result = CalibrationResult(
            weights=weights,
            sampled_codes=sampled_codes,
            unit_ids=unit_ids,
            spawn_keys=spawn_keys,
            mode=mode,
            seed=self.seed,
            target_pop_tot=self.target_pop_tot,
            row_var=self.acs_row_var_name,
            col_var=self.acs_col_var_name,
            row_cats=np.asarray(self.acs_row_cats),
            col_cats=np.asarray(self.acs_col_cats),
        )

        print(
            'Weights created, returning CalibrationResult object.\n'
            'Use all_final_weights() to obtain final weights.\n'
            'Use to_df() to obtain replicate-specific stage1 and stage2 weights with metadata.\n'
            'Use to_long_df() to obtain all replicate weights with metadata.\n'
        )
        return result

    def _compute_one_replicate(
        self, rng: np.random.Generator, mode: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sampled_row_codes = sample_codes_from_cdf(self.cdf_row_aligned, rng)
        sampled_col_codes = sample_codes_from_cdf(self.cdf_col_aligned, rng)
        joint_codes = make_joint_code(sampled_row_codes, sampled_col_codes, self.num_col_cats)
        N = len(self.home_cbgs)

        if mode in _MODES_NEEDING_STAGE1:
            if self.acs_joint_margin is not None:
                weight1, _ = stage1_rake(
                    sampled_row_codes,
                    sampled_col_codes,
                    self.num_col_cats,
                    self.acs_joint_margin,
                )
            else:
                weight1, _ = stage1_ipf(
                    sampled_row_codes,
                    sampled_col_codes,
                    self.acs_row_margin,
                    self.acs_col_margin,
                    n_iter=10,
                )
        else:
            weight1 = np.full(N, 1.0 / N)

        if mode in _MODES_NEEDING_STAGE2:
            weight_final, _ = stage2_rake(
                sampled_row_codes,
                sampled_col_codes,
                self.num_row_cats,
                self.num_col_cats,
                self.assigned_cluster_labels,
                weight1,
                self.atus_target_table,
                factor_clip=None,
            )
        else:
            weight_final = weight1

        return weight1, weight_final, sampled_row_codes, sampled_col_codes, joint_codes
