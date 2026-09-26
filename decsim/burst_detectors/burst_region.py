"""The positions and rounds a flag covers, and the priors they get.

The strong window model may raise the priors of the flagged region's
faults with its graph unchanged (IonQ 2608.25027 lines 334-340).
"""

import dataclasses
from typing import Optional

import numpy

import decsim.burst_detectors.flag_log as flag_log
import decsim.burst_detectors.layout as layout_module
import decsim.detector_error_model.fault_model_contracts as fault_models


@dataclasses.dataclass(frozen=True)
class BurstRegion:
    """The positions and rounds whose faults get the burst priors."""

    positions: frozenset
    first_round: int
    # None while the flag is open: every later round is in the burst
    last_round: Optional[int]
    prior_scale: float

    @classmethod
    def of_episode(
        cls,
        layout: layout_module.Layout,
        episode: flag_log.Episode,
        is_in_region: numpy.ndarray,
        flagged_rows: list,
    ) -> "BurstRegion":
        """The region's positions, and the scale its measured rate asks.

        flagged_rows are the region's per-position events over the
        rounds its row measures the rate on.
        """
        measured_rate = _mean_rate(flagged_rows, is_in_region)
        prior_scale = layout.prior_scale(is_in_region, measured_rate)
        positions = _positions_where(layout.positions, is_in_region)
        last_round = episode.last_firing_round
        if episode.is_open:
            last_round = None
        return cls(positions, episode.first_round, last_round, prior_scale)

    def raised(
        self, model: fault_models.WindowErrorModel
    ) -> fault_models.WindowErrorModel:
        """The model with every region fault's prior scaled, capped at 1/2.

        Only the prior vector changes: "we update only the corresponding
        entries of the prior vector ... No reconstruction of the Tanner
        graph" (IonQ 2608.25027 lines 334-340).
        """
        if self.prior_scale == 1.0:
            return model
        region_rows = self._region_rows(model)
        graphlike = self._raised_faults(model.graphlike_faults, region_rows)
        physical = self._raised_faults(model.physical_faults, region_rows)
        return dataclasses.replace(
            model, graphlike_faults=graphlike, physical_faults=physical
        )

    def _region_rows(self, model):
        """The model's rows at a region position inside the flagged rounds."""
        rows = []
        coordinates = model.detector_coordinates
        for row, detector_id in enumerate(model.detector_ids):
            position = model.defect_positions[detector_id]
            round_index = position[0]
            row_coordinates = coordinates[row]
            planar = tuple(row_coordinates[:2])
            if self._covers(planar, round_index):
                rows.append(row)
        return numpy.asarray(rows, dtype=numpy.int64)

    def _covers(self, planar: tuple, round_index: int) -> bool:
        if planar not in self.positions:
            return False
        if round_index < self.first_round:
            return False
        return self.last_round is None or round_index <= self.last_round

    def _raised_faults(self, faults, region_rows):
        if faults is None:
            return None
        region_check = faults.check[region_rows]
        hits = region_check.sum(axis=0)
        hit_counts = numpy.ravel(hits)
        is_touched = hit_counts > 0
        priors = numpy.array(faults.priors, dtype=numpy.float64)
        scaled = priors[is_touched] * self.prior_scale
        priors[is_touched] = numpy.minimum(scaled, layout_module.MAXIMUM_PRIOR)
        return dataclasses.replace(faults, priors=priors)


def _mean_rate(flagged_rows: list, is_in_region) -> float:
    """Events per region position per round over the flagged rows.

    A count flag may leave the region empty, which measures nothing.
    """
    region_size = numpy.count_nonzero(is_in_region)
    if region_size == 0:
        return 0.0
    totals = numpy.sum(flagged_rows, axis=0)
    region_events = numpy.sum(totals[is_in_region])
    samples = region_size * len(flagged_rows)
    return float(region_events) / samples


def _positions_where(positions: tuple, is_in_region) -> frozenset:
    members = []
    for position, is_member in zip(positions, is_in_region, strict=True):
        if is_member:
            members.append(position)
    return frozenset(members)
