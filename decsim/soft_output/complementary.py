"""Complementary-gap soft output: ``g_comp = |w_comp - w_min|`` for MWPM (SoftOutputMetric seam)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..message import SoftOutput, SoftOutputSource
from ..detector_error_model import GRAPHLIKE_FAULT_MODEL_REQUIRED
from ..mwpm_decoder.weights import matching_weights

if TYPE_CHECKING:
    import stim
    from ..detector_error_model import WindowErrorModel

_CITATION = "Toshio et al. 2510.25222 Sec. II.C"

COMPLEMENTARY_GAP_SOURCE = SoftOutputSource(
    method="complementary_gap",
    cluster_origin="mwpm_opposite_logical",
    growth_schedule="minimum_weight_matching",
    gap_units="log_likelihood_weight",
    correction="opposite_logical_constraint",
    references=("arXiv:2510.25222v1 Section II.C",),
)


def dem_to_matrices(dem: "stim.DetectorErrorModel"):
    """Flatten a decomposed DEM into ``(H[det x err], O[obs x err], weights=ln((1-p)/p))``."""
    import numpy as np

    from ..detector_error_model import (
        canonical_error_instructions,
        validate_graphlike_fault,
    )
    num_det = dem.num_detectors
    num_obs = dem.num_observables
    h_cols: list = []
    o_cols: list = []
    weights: list = []

    def append_component(dets, obs, weight) -> None:
        hcol = np.zeros(num_det, dtype=np.uint8)
        hcol[list(dets)] = 1
        ocol = np.zeros(num_obs, dtype=np.uint8)
        ocol[list(obs)] = 1
        h_cols.append(hcol)
        o_cols.append(ocol)
        weights.append(weight)

    for record in canonical_error_instructions(dem):
        prob = record.probability
        weight = float(matching_weights([prob])[0])
        for component in record.components:
            fault = validate_graphlike_fault(
                component.detectors,
                component.logical_observables,
                location=(
                    f"error {record.error_ordinal} component "
                    f"{component.component_ordinal}"
                ),
            )
            assert fault is not None
            detectors, logical_observables = fault
            append_component(
                detectors,
                logical_observables,
                weight,
            )

    h = np.array(h_cols, dtype=np.uint8).T if h_cols else np.zeros((num_det, 0), np.uint8)
    o = np.array(o_cols, dtype=np.uint8).T if o_cols else np.zeros((num_obs, 0), np.uint8)
    return h, o, np.asarray(weights, dtype=float)


class ComplementaryGapMetric:
    """Complementary-gap soft output for a single-observable decoding window."""

    name = "complementary_gap"

    def __init__(self, check, obs, weights):
        import numpy as np
        import pymatching

        from ..detector_error_model import validate_graphlike_matrices

        self.check = np.asarray(check, dtype=np.uint8)
        self.obs = np.asarray(obs, dtype=np.uint8)
        self.weights = np.asarray(weights, dtype=float)
        validate_graphlike_matrices(
            self.check,
            self.obs,
            location="complementary-gap model",
        )
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
        from ..detector_error_model import FaultRepresentation

        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        return cls(
            faults.check,
            faults.observables,
            matching_weights(faults.priors),
        )

    def evaluate(self, syndrome) -> SoftOutput:
        """Return the :class:`SoftOutput` (logical value + gap) for one syndrome."""
        import numpy as np

        bits = np.asarray(syndrome, dtype=np.uint8).ravel()
        correction, w_min = self._base.decode(bits, return_weight=True)
        pred = int(((self.obs @ correction) % 2)[0]) if self.obs.shape[1] else 0
        forced = np.concatenate([bits, [pred ^ 1]]).astype(np.uint8)
        _, w_comp = self._aug.decode(forced, return_weight=True)
        return SoftOutput(
            gap=abs(float(w_comp) - float(w_min)),
            source=COMPLEMENTARY_GAP_SOURCE,
            w_min=float(w_min),
            w_comp=float(w_comp),
        )


@dataclass(frozen=True)
class ComplementaryGapMetricFactory:
    """Stateless complementary-gap builder used by ``SoftOutputDecoder``."""

    source = COMPLEMENTARY_GAP_SOURCE
    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def from_window_model(
        self,
        model: "WindowErrorModel",
    ) -> ComplementaryGapMetric:
        return ComplementaryGapMetric.from_window_model(model)
