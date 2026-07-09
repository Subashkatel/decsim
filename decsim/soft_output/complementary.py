"""Complementary-gap soft output: ``g_comp = |w_comp - w_min|`` for MWPM (SoftOutputMetric seam)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..message import SoftOutput

if TYPE_CHECKING:
    import stim
    from ..detector_error_model import WindowErrorModel

_CITATION = "Toshio et al. 2510.25222 Sec. II.C"


def dem_to_matrices(dem: "stim.DetectorErrorModel"):
    """Flatten a decomposed DEM into ``(H[det x err], O[obs x err], weights=ln((1-p)/p))``."""
    import numpy as np

    num_det = dem.num_detectors
    num_obs = dem.num_observables
    h_cols: list = []
    o_cols: list = []
    weights: list = []

    def flush(dets, obs, weight) -> None:
        if not dets and not obs:
            return
        hcol = np.zeros(num_det, dtype=np.uint8)
        hcol[dets] = 1
        ocol = np.zeros(num_obs, dtype=np.uint8)
        ocol[obs] = 1
        h_cols.append(hcol)
        o_cols.append(ocol)
        weights.append(weight)

    for instr in dem.flattened():
        if instr.type != "error":
            continue
        prob = instr.args_copy()[0]
        weight = float(np.log((1 - prob) / prob)) if 0 < prob < 1 else 50.0
        dets: list = []
        obs: list = []
        for target in instr.targets_copy():
            if target.is_separator():
                flush(dets, obs, weight)
                dets, obs = [], []
            elif target.is_relative_detector_id():
                dets.append(target.val)
            elif target.is_logical_observable_id():
                obs.append(target.val)
        flush(dets, obs, weight)

    h = np.array(h_cols, dtype=np.uint8).T if h_cols else np.zeros((num_det, 0), np.uint8)
    o = np.array(o_cols, dtype=np.uint8).T if o_cols else np.zeros((num_obs, 0), np.uint8)
    return h, o, np.asarray(weights, dtype=float)


def _weights_from_priors(priors):
    """Convert fault probabilities into matching edge weights ``ln((1-p)/p)``."""
    import numpy as np

    priors = np.asarray(priors, dtype=float)
    return np.log((1.0 - priors) / priors)


class ComplementaryGapMetric:
    """Complementary-gap soft output for a single-observable decoding window."""

    name = "complementary_gap"

    def __init__(self, check, obs, weights):
        import numpy as np
        import pymatching

        self.check = np.asarray(check, dtype=np.uint8)
        self.obs = np.asarray(obs, dtype=np.uint8)
        self.weights = np.asarray(weights, dtype=float)
        if self.obs.shape[0] != 1:
            raise ValueError(
                "the complementary gap is defined for one observable; got "
                f"{self.obs.shape[0]}. Decode each logical operator with its own "
                "metric (paper Sec. II.C notes the multi-observable subtlety).")
        self._base = pymatching.Matching.from_check_matrix(self.check, weights=self.weights)
        check_aug = np.vstack([self.check, self.obs[0:1, :]])
        self._aug = pymatching.Matching.from_check_matrix(check_aug, weights=self.weights)

    @classmethod
    def from_dem(cls, dem: "stim.DetectorErrorModel") -> "ComplementaryGapMetric":
        """Build the metric from a (decomposed) stim DetectorErrorModel."""
        check, obs, weights = dem_to_matrices(dem)
        return cls(check, obs, weights)

    @classmethod
    def from_window_model(cls, model: "WindowErrorModel") -> "ComplementaryGapMetric":
        """Build the metric from a decsim WindowErrorModel (check/obs/priors)."""
        return cls(model.check, model.obs, _weights_from_priors(model.priors))

    def evaluate(self, syndrome) -> SoftOutput:
        """Return the :class:`SoftOutput` (logical value + gap) for one syndrome."""
        import numpy as np

        bits = np.asarray(syndrome, dtype=np.uint8).ravel()
        correction, w_min = self._base.decode(bits, return_weight=True)
        pred = int(((self.obs @ correction) % 2)[0]) if self.obs.shape[1] else 0
        forced = np.concatenate([bits, [pred ^ 1]]).astype(np.uint8)
        _, w_comp = self._aug.decode(forced, return_weight=True)
        return SoftOutput(
            logical_value=pred,
            gap=abs(float(w_comp) - float(w_min)),
            w_min=float(w_min),
            w_comp=float(w_comp),
        )
