"""Slices one circuit's fault catalog into window models, window by window.

The slicer builds the catalog once, in every representation the decoder
asked for, and indexes it by round: which detectors each round holds,
which rounds each fault touches, which faults touch each round. Slicing a
window then costs the window's own size, not the circuit's. That is the
shape of qLDPC's SlidingWindowDecoder, which cuts one whole-circuit
detector error model into (detection region, commit region) windows and
removes each committed error from the windows after it.

Called window after window in time order, the slicer advances ownership
itself: a fault one window commits is left out of every later window. A
plan compiled from a dependency graph supplies the owner sets instead, so
the order windows are built in cannot change which window commits a
fault.
"""

from typing import Optional

from decsim.detector_error_model import (
    detector_chronology,
    fault_model_contracts,
    stim_fault_catalog,
    window_placement,
)


class WindowSlicer:
    """Builds one window's matrices at a time from the circuit's catalog."""

    def __init__(
        self,
        circuit,
        *,
        round_count: int,
        detector_rounds: Optional[dict] = None,
        fault_model_requirement: (
            fault_model_contracts.DecoderFaultModelRequirement
        ),
    ):
        self.catalogs, self.catalog_link = (
            stim_fault_catalog.prepare_fault_catalogs(
                circuit, fault_model_requirement
            )
        )
        self.detector_coordinates = circuit.get_detector_coordinates()
        self.observable_count = circuit.num_observables
        self.round_by_detector = detector_chronology.resolve_detector_rounds(
            circuit, detector_rounds, round_count
        )
        self.position_by_detector = (
            detector_chronology.detector_position_in_round(
                self.round_by_detector
            )
        )
        self.committed_elsewhere = {
            representation: set() for representation in self.catalogs
        }
        self.detectors_by_round = detector_chronology.detectors_by_round(
            self.round_by_detector
        )
        self.fault_rounds = {}
        self.faults_by_round = {}
        for representation, catalog in self.catalogs.items():
            rounds_per_fault = self._rounds_per_fault(catalog)
            self.fault_rounds[representation] = rounds_per_fault
            self.faults_by_round[representation] = _faults_by_round(
                rounds_per_fault
            )

    def slice_window(
        self,
        first_buffer_round: int,
        first_commit_round: int,
        last_commit_round: int,
        last_buffer_round: int,
        *,
        is_last: bool,
        fault_exclusion_ranges: tuple = (),
        explicitly_owned_faults: Optional[dict] = None,
        explicitly_prior_faults: Optional[dict] = None,
    ) -> fault_model_contracts.WindowErrorModel:
        """One window's model; advances ownership unless owners are given.

        The explicit owner and prior maps come together or not at all.
        """
        has_owners = explicitly_owned_faults is not None
        has_priors = explicitly_prior_faults is not None
        if has_owners != has_priors:
            raise ValueError(
                "explicit owner and predecessor fault maps must be supplied "
                "together"
            )
        window_placement.validate_fault_exclusion_ranges(fault_exclusion_ranges)
        context = self._placement_context(
            first_buffer_round,
            first_commit_round,
            last_commit_round,
            last_buffer_round,
            is_last,
        )
        placed = {}
        for representation, catalog in self.catalogs.items():
            owned = _for_representation(explicitly_owned_faults, representation)
            prior = _for_representation(explicitly_prior_faults, representation)
            placed[representation] = self._place(
                catalog, context, fault_exclusion_ranges, owned, prior
            )
        return self._window_model(context.rows, placed)

    def _placement_context(
        self,
        first_buffer_round: int,
        first_commit_round: int,
        last_commit_round: int,
        last_buffer_round: int,
        is_last: bool,
    ) -> window_placement.WindowPlacementContext:
        rows = window_placement.detectors_in_window(
            self.detectors_by_round,
            first_buffer_round,
            last_buffer_round,
            is_last=is_last,
        )
        row_by_detector = {
            detector_id: row_number
            for row_number, detector_id in enumerate(rows)
        }
        return window_placement.WindowPlacementContext(
            rows=rows,
            row_by_detector=row_by_detector,
            observable_count=self.observable_count,
            first_commit_round=first_commit_round,
            last_commit_round=last_commit_round,
            is_last=is_last,
        )

    def _rounds_per_fault(self, catalog) -> tuple:
        """Each catalog fault's rounds, one per detector it flips."""
        rounds_per_fault = []
        for detectors in catalog.detector_sets:
            rounds = tuple(
                self.round_by_detector[detector_id] for detector_id in detectors
            )
            rounds_per_fault.append(rounds)
        return tuple(rounds_per_fault)

    def _place(
        self,
        catalog,
        context,
        fault_exclusion_ranges,
        explicitly_owned_faults,
        explicitly_prior_faults,
    ):
        representation = catalog.representation
        candidate_faults = self._candidate_faults(representation, context.rows)
        return window_placement.placed_faults_for_window(
            catalog=catalog,
            context=context,
            fault_rounds=self.fault_rounds[representation],
            candidate_faults=candidate_faults,
            committed_elsewhere=self.committed_elsewhere[representation],
            explicitly_owned_faults=explicitly_owned_faults,
            explicitly_prior_faults=explicitly_prior_faults,
            fault_exclusion_ranges=fault_exclusion_ranges,
        )

    def _candidate_faults(self, representation, rows) -> list[int]:
        """The faults touching any round of `rows`, in catalog order."""
        by_round = self.faults_by_round[representation]
        rounds = {self.round_by_detector[detector_id] for detector_id in rows}
        seen: set[int] = set()
        for round_index in sorted(rounds):
            faults = by_round.get(round_index, ())
            seen.update(faults)
        return sorted(seen)

    def _window_model(
        self, rows: list[int], placed: dict
    ) -> fault_model_contracts.WindowErrorModel:
        graphlike = placed.get(
            fault_model_contracts.FaultRepresentation.GRAPHLIKE
        )
        physical = placed.get(
            fault_model_contracts.FaultRepresentation.PHYSICAL
        )
        local_link = None
        if self.catalog_link is not None:
            local_link = window_placement.local_link_projection(
                graphlike, physical, self.catalog_link
            )
        coordinates = detector_chronology.coordinates_for_rows(
            self.detector_coordinates, rows
        )
        defect_positions = self._defect_positions(rows, placed)
        return fault_model_contracts.WindowErrorModel(
            detector_ids=tuple(rows),
            detector_coordinates=coordinates,
            defect_positions=defect_positions,
            graphlike_faults=graphlike,
            physical_faults=physical,
            physical_to_graphlike_detector_projection=local_link,
        )

    def _defect_positions(self, rows, placed) -> dict:
        """(round, position) of every row and every handed-off detector."""
        handed_off = set()
        for fault_view in placed.values():
            for flips in fault_view.boundary_flips.values():
                handed_off.update(flips)
        residual_rows = set(rows) | handed_off
        positions = {}
        for detector_id in residual_rows:
            positions[detector_id] = (
                self.round_by_detector[detector_id],
                self.position_by_detector[detector_id],
            )
        return positions


def _faults_by_round(rounds_per_fault: tuple) -> dict[int, list[int]]:
    """The faults touching each round, in catalog order."""
    by_round: dict[int, list[int]] = {}
    for fault_index, rounds in enumerate(rounds_per_fault):
        for round_index in set(rounds):
            faults = by_round.setdefault(round_index, [])
            faults.append(fault_index)
    return by_round


def _for_representation(maps_by_representation: Optional[dict], representation):
    """One representation's map, or None when no maps were given."""
    if maps_by_representation is None:
        return None
    return maps_by_representation[representation]
