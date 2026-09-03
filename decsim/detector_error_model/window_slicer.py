"""Slices one circuit's fault catalog into window models, window by window.

The catalog is built once and indexed by round, so a window costs its
own size; called in time order the slicer advances ownership itself, and
a plan compiled from a dependency graph supplies the owner sets instead
(the shape of qLDPC's SlidingWindowDecoder, which cuts one detector error
model into windows and removes each committed error from the later ones).
"""

import dataclasses
from collections.abc import Container
from typing import Optional

import stim

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
        circuit: stim.Circuit,
        *,
        round_count: int,
        detector_rounds: Optional[dict[int, int]] = None,
        fault_model_requirement: (
            fault_model_contracts.DecoderFaultModelRequirement
        ),
    ):
        self.catalogs, self.catalog_link = (
            stim_fault_catalog.prepare_fault_catalogs(
                circuit, fault_model_requirement
            )
        )
        self.observable_count = circuit.num_observables
        self.chronology = _index_chronology(
            circuit, detector_rounds, round_count
        )
        self.fault_index = _index_faults(
            self.catalogs, self.chronology.round_by_detector
        )
        self.committed_elsewhere = {
            representation: set() for representation in self.catalogs
        }

    def slice_window(
        self,
        first_buffer_round: int,
        first_commit_round: int,
        last_commit_round: int,
        last_buffer_round: int,
        *,
        is_last: bool,
        fault_exclusion_ranges: tuple[tuple[int, int], ...] = (),
        explicitly_owned_faults: Optional[
            dict[fault_model_contracts.FaultRepresentation, set[int]]
        ] = None,
        explicitly_prior_faults: Optional[
            dict[fault_model_contracts.FaultRepresentation, Container[int]]
        ] = None,
    ) -> fault_model_contracts.WindowErrorModel:
        """One window's model; advances ownership unless owners are given.

        `is_last` is given only for a window whose commit rounds reach
        round_count.
        """
        has_owners = explicitly_owned_faults is not None
        has_priors = explicitly_prior_faults is not None
        assert has_owners == has_priors, "owner and prior maps come together"
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
            self.chronology.detectors_by_round,
            first_buffer_round,
            last_buffer_round,
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

    def _place(
        self,
        catalog: fault_model_contracts.FaultCatalog,
        context: window_placement.WindowPlacementContext,
        fault_exclusion_ranges: tuple[tuple[int, int], ...],
        explicitly_owned_faults: Optional[set[int]],
        explicitly_prior_faults: Optional[Container[int]],
    ) -> fault_model_contracts.PlacedFaultModel:
        representation = catalog.representation
        candidate_faults = self._candidate_faults(representation, context.rows)
        return window_placement.placed_faults_for_window(
            catalog=catalog,
            context=context,
            fault_rounds=self.fault_index.fault_rounds[representation],
            candidate_faults=candidate_faults,
            committed_elsewhere=self.committed_elsewhere[representation],
            explicitly_owned_faults=explicitly_owned_faults,
            explicitly_prior_faults=explicitly_prior_faults,
            fault_exclusion_ranges=fault_exclusion_ranges,
        )

    def _candidate_faults(
        self,
        representation: fault_model_contracts.FaultRepresentation,
        rows: list[int],
    ) -> list[int]:
        """The faults touching any round of `rows`, in catalog order."""
        by_round = self.fault_index.faults_by_round[representation]
        round_by_detector = self.chronology.round_by_detector
        rounds = {round_by_detector[detector_id] for detector_id in rows}
        seen: set[int] = set()
        for round_index in sorted(rounds):
            faults = by_round.get(round_index, ())
            seen.update(faults)
        return sorted(seen)

    def _window_model(
        self,
        rows: list[int],
        placed: dict[
            fault_model_contracts.FaultRepresentation,
            fault_model_contracts.PlacedFaultModel,
        ],
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
            self.chronology.detector_coordinates, rows
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

    def _defect_positions(
        self,
        rows: list[int],
        placed: dict[
            fault_model_contracts.FaultRepresentation,
            fault_model_contracts.PlacedFaultModel,
        ],
    ) -> dict[int, tuple[int, int]]:
        """(round, position) of every row and every handed-off detector."""
        handed_off = set()
        for fault_view in placed.values():
            for flips in fault_view.boundary_flips.values():
                handed_off.update(flips)
        residual_rows = set(rows) | handed_off
        positions = {}
        for detector_id in residual_rows:
            positions[detector_id] = (
                self.chronology.round_by_detector[detector_id],
                self.chronology.position_by_detector[detector_id],
            )
        return positions


@dataclasses.dataclass(frozen=True)
class _ChronologyIndex:
    """Where every detector of the circuit sits in time.

    Its round, its position among that round's detectors, each round's
    detectors in Stim order, and Stim's coordinates for it.
    """

    round_by_detector: dict[int, int]
    position_by_detector: dict[int, int]
    detectors_by_round: dict[int, list[int]]
    detector_coordinates: dict[int, list[float]]


@dataclasses.dataclass(frozen=True)
class _FaultIndex:
    """The catalog faults indexed by round, per representation.

    `fault_rounds` lists each fault's rounds, one per detector it flips;
    `faults_by_round` lists the faults touching each round, in catalog
    order.
    """

    fault_rounds: dict[
        fault_model_contracts.FaultRepresentation, tuple[tuple[int, ...], ...]
    ]
    faults_by_round: dict[
        fault_model_contracts.FaultRepresentation, dict[int, list[int]]
    ]


def _index_chronology(
    circuit: stim.Circuit,
    detector_rounds: Optional[dict[int, int]],
    round_count: int,
) -> _ChronologyIndex:
    """Every detector's place in time, read off the circuit once."""
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, detector_rounds, round_count
    )
    position_by_detector = detector_chronology.detector_position_in_round(
        round_by_detector
    )
    detectors_by_round = detector_chronology.detectors_by_round(
        round_by_detector
    )
    detector_coordinates = circuit.get_detector_coordinates()
    return _ChronologyIndex(
        round_by_detector=round_by_detector,
        position_by_detector=position_by_detector,
        detectors_by_round=detectors_by_round,
        detector_coordinates=detector_coordinates,
    )


def _index_faults(
    catalogs: dict[
        fault_model_contracts.FaultRepresentation,
        fault_model_contracts.FaultCatalog,
    ],
    round_by_detector: dict[int, int],
) -> _FaultIndex:
    """Every catalog indexed by round, keyed by representation."""
    fault_rounds = {}
    faults_by_round = {}
    for representation, catalog in catalogs.items():
        rounds_per_fault = _rounds_per_fault(catalog, round_by_detector)
        fault_rounds[representation] = rounds_per_fault
        faults_by_round[representation] = _faults_by_round(rounds_per_fault)
    return _FaultIndex(
        fault_rounds=fault_rounds, faults_by_round=faults_by_round
    )


def _rounds_per_fault(
    catalog: fault_model_contracts.FaultCatalog,
    round_by_detector: dict[int, int],
) -> tuple[tuple[int, ...], ...]:
    """Each catalog fault's rounds, one per detector it flips."""
    rounds_per_fault = []
    for detectors in catalog.detector_sets:
        rounds = tuple(
            round_by_detector[detector_id] for detector_id in detectors
        )
        rounds_per_fault.append(rounds)
    return tuple(rounds_per_fault)


def _faults_by_round(
    rounds_per_fault: tuple[tuple[int, ...], ...],
) -> dict[int, list[int]]:
    """The faults touching each round, in catalog order."""
    by_round: dict[int, list[int]] = {}
    for fault_index, rounds in enumerate(rounds_per_fault):
        for round_index in set(rounds):
            faults = by_round.setdefault(round_index, [])
            faults.append(fault_index)
    return by_round


def _for_representation(
    maps_by_representation: Optional[
        dict[fault_model_contracts.FaultRepresentation, object]
    ],
    representation: fault_model_contracts.FaultRepresentation,
) -> Optional[object]:
    """One representation's map, or None when no maps were given."""
    if maps_by_representation is None:
        return None
    return maps_by_representation[representation]
