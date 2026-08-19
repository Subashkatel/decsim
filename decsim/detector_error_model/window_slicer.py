"""The reusable slicing engine.

Builds one global model in every requested domain once, then slices it window
by window into WindowErrorModel products, with fault ownership either computed
incrementally as windows are sliced or supplied from outside.
"""

from __future__ import annotations

from typing import Optional

from .fault_model_contracts import (
    DecoderFaultModelRequirement,
    FaultRepresentation,
    WindowErrorModel,
)
from .stim_dem_catalog import _prepare_fault_catalogs
from .detector_chronology import (
    _coordinates_for_rows,
    _detector_position_in_round,
    resolve_detector_rounds,
)
from .window_placement import (
    WindowPlacementContext,
    _detectors_in_window,
    _local_physical_to_graphlike_detector_projection,
    _placed_faults_for_window,
    _validate_fault_exclusion_ranges,
)


class WindowSlicer:
    """Build local matrices from one global model in every requested domain.

    Direct callers may advance ownership window by window. Static planners can
    instead supply a precomputed owner set so construction order cannot change
    which window commits a fault.
    """

    def __init__(
        self,
        circuit,
        *,
        round_count: int,
        detector_rounds: Optional[dict] = None,
        fault_model_requirement: DecoderFaultModelRequirement,
    ):
        self.catalogs, self.catalog_link = _prepare_fault_catalogs(
            circuit,
            fault_model_requirement,
        )
        self.detector_coordinates = circuit.get_detector_coordinates()
        self.n_obs = circuit.num_observables
        self.round_of = resolve_detector_rounds(
            circuit, detector_rounds, round_count
        )
        self.pos_of = _detector_position_in_round(self.round_of)
        self.committed_elsewhere = {
            representation: set()
            for representation in self.catalogs
        }

    def slice_window(
        self,
        buffer_lo: int,
        commit_lo: int,
        commit_hi: int,
        buffer_hi: int,
        *,
        is_last: bool,
        fault_exclusion_ranges: tuple = (),
        explicitly_owned_faults: Optional[dict] = None,
        explicitly_prior_faults: Optional[dict] = None,
    ) -> WindowErrorModel:
        """Create one local model and update incremental ownership if used."""
        explicit_values = (
            explicitly_owned_faults,
            explicitly_prior_faults,
        )
        if (explicit_values[0] is None) != (explicit_values[1] is None):
            raise ValueError(
                "explicit owner and predecessor fault maps must be supplied together"
            )
        _validate_fault_exclusion_ranges(fault_exclusion_ranges)
        rows = _detectors_in_window(
            self.round_of,
            buffer_lo,
            buffer_hi,
            is_last=is_last,
        )
        row_index = {
            detector_id: row_number
            for row_number, detector_id in enumerate(rows)
        }
        lead_rows = {
            detector_id
            for detector_id in rows
            if self.round_of[detector_id] < commit_lo
        }
        context = WindowPlacementContext(
            rows=rows,
            row_index=row_index,
            lead_rows=lead_rows,
            round_of=self.round_of,
            n_obs=self.n_obs,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            is_last=is_last,
        )
        placed = {
            representation: _placed_faults_for_window(
                catalog=catalog,
                context=context,
                committed_elsewhere=self.committed_elsewhere[representation],
                explicitly_owned_faults=(
                    None
                    if explicitly_owned_faults is None
                    else explicitly_owned_faults[representation]
                ),
                explicitly_prior_faults=(
                    None
                    if explicitly_prior_faults is None
                    else explicitly_prior_faults[representation]
                ),
                fault_exclusion_ranges=fault_exclusion_ranges,
            )
            for representation, catalog in self.catalogs.items()
        }
        graphlike = placed.get(FaultRepresentation.GRAPHLIKE)
        physical = placed.get(FaultRepresentation.PHYSICAL)
        local_link = None
        if self.catalog_link is not None:
            local_link = _local_physical_to_graphlike_detector_projection(
                graphlike,
                physical,
                self.catalog_link,
            )
        residual_rows = set(rows) | {
            detector_id
            for fault_view in placed.values()
            for flips in fault_view.boundary_flips.values()
            for detector_id in flips
        }
        return WindowErrorModel(
            detector_ids=tuple(rows),
            detector_coordinates=_coordinates_for_rows(
                self.detector_coordinates, rows
            ),
            defect_positions={
                detector_id: (
                    self.round_of[detector_id],
                    self.pos_of[detector_id],
                )
                for detector_id in residual_rows
            },
            graphlike_faults=graphlike,
            physical_faults=physical,
            physical_to_graphlike_detector_projection=local_link,
        )
