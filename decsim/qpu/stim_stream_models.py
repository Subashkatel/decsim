"""Window fault models of a growing repeated Stim circuit.

Stim error identities are their complete detector and observable effects,
not their changing catalog positions. WindowSlicer replays the qLDPC-style
ownership rule on each complete model, while published ids stay stable.
"""

import dataclasses
import math
from typing import Optional

import decsim.detector_error_model.detector_formation as formation
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.records.circuits as circuit_records
import decsim.records.windows as window_records

# Stim's independent-probability merges can change floating addition order
# when a repeated block is expanded. This tolerance changes no model weight.
_PRIOR_PROBABILITY_TOLERANCE = 1e-15


class GrowingStimModels:
    """Keep fault identities stable while a physical stream gains rounds."""

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        self.program = program
        self.requirement = requirement
        self.identities = _FaultIdentities()
        self.windows_by_index: dict[int, window_records.Window] = {}
        self.models_by_window_index: dict[
            int, fault_models.WindowErrorModel
        ] = {}
        self.final_round_count: Optional[int] = None

    def finish(self, round_count: int) -> None:
        """Bind model finalization to the source's actual data readout."""
        if self.final_round_count is not None:
            raise RuntimeError("a physical stream can only finish once")
        self.final_round_count = round_count

    def for_window(
        self, window: window_records.Window
    ) -> fault_models.WindowErrorModel:
        """Build one window, preserving every already-queued window's law."""
        slicer, source_ids, round_count = self._build_slicer(window)
        self.windows_by_index[window.window_index] = window
        requested_model = None
        for window_index in sorted(self.windows_by_index):
            if window_index > window.window_index:
                break
            earlier_window = self.windows_by_index[window_index]
            model = _slice(slicer, earlier_window, round_count)
            stable_model = _stable_model(model, source_ids)
            self._check_published(window_index, earlier_window, stable_model)
            requested_model = stable_model
        assert requested_model is not None, (
            "the requested window is in the stream"
        )
        self.models_by_window_index[window.window_index] = requested_model
        return requested_model

    def for_strong_window(
        self,
        window: window_records.Window,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict],
    ) -> fault_models.WindowErrorModel:
        """Re-decode with priors in the same namespace as weak corrections."""
        slicer, source_ids, round_count = self._build_slicer(window)
        local_priors = _local_prior_faults(prior_faults, source_ids)
        is_last = window.commit_hi == self.final_round_count
        last_buffer_round = min(window.buffer_hi, round_count)
        model = slicer.slice_window(
            window.start_round,
            window.commit_lo,
            window.commit_hi,
            last_buffer_round,
            is_last=is_last,
            fault_exclusion_ranges=fault_exclusion_ranges,
            explicitly_prior_faults=local_priors,
        )
        return _stable_model(model, source_ids)

    def _build_slicer(self, window):
        if window.closed_temporal_boundaries:
            raise ValueError(
                "live Stim streams require trailing-buffer feedback"
            )
        round_count = self.final_round_count
        if round_count is None:
            # One later physical round keeps synthetic final detectors
            # outside this memory window's read buffer.
            round_count = window.buffer_hi + 1
        circuit, measurement_rounds = self.program.assemble(round_count)
        table = formation.build_formation_table(
            circuit, round_count, measurement_rounds=measurement_rounds
        )
        detector_rounds = table.detector_rounds()
        slicer = window_slicer.WindowSlicer(
            circuit,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=self.requirement,
        )
        source_ids = self.identities.register(
            slicer.catalogs, slicer.catalog_link
        )
        return slicer, source_ids, round_count

    def _check_published(self, window_index, window, model) -> None:
        if not window.queued and not window.committed:
            return
        previous = self.models_by_window_index.get(window_index)
        assert previous is not None, "a queued window has an installed model"
        _check_same_model(previous, model)


class _FaultIdentities:
    """An append-only namespace for complete Stim fault effects."""

    def __init__(self) -> None:
        self.ids_by_representation: dict = {}

    def register(
        self,
        catalogs: dict[
            fault_models.FaultRepresentation, fault_models.FaultCatalog
        ],
        catalog_link: Optional[object],
    ) -> dict[fault_models.FaultRepresentation, tuple[int, ...]]:
        source_ids = {}
        for representation, catalog in catalogs.items():
            known = self.ids_by_representation.setdefault(representation, {})
            local_ids = []
            for index, detectors in enumerate(catalog.detector_sets):
                observables = catalog.observable_sets[index]
                decomposition = _decomposition_identity(
                    representation, index, catalogs, catalog_link
                )
                identity = (detectors, observables, decomposition)
                stable_id = known.setdefault(identity, len(known))
                local_ids.append(stable_id)
            source_ids[representation] = tuple(local_ids)
        return source_ids


def _decomposition_identity(representation, index, catalogs, catalog_link):
    if catalog_link is None:
        return ()
    if representation is not fault_models.FaultRepresentation.PHYSICAL:
        return ()
    graphlike = catalogs[fault_models.FaultRepresentation.GRAPHLIKE]
    column = catalog_link.getcol(index)
    column = column.tocoo()
    components = []
    for component in column.row:
        detectors = graphlike.detector_sets[component]
        observables = graphlike.observable_sets[component]
        components.append((detectors, observables))
    return tuple(sorted(components))


def _slice(slicer, window, round_count):
    last_buffer_round = min(window.buffer_hi, round_count)
    is_last = window.commit_hi == round_count
    return slicer.slice_window(
        window.start_round,
        window.commit_lo,
        window.commit_hi,
        last_buffer_round,
        is_last=is_last,
    )


def _stable_model(model, source_ids):
    graphlike = _stable_faults(model.graphlike_faults, source_ids)
    physical = _stable_faults(model.physical_faults, source_ids)
    return dataclasses.replace(
        model, graphlike_faults=graphlike, physical_faults=physical
    )


def _stable_faults(faults, source_ids):
    if faults is None:
        return None
    identities = source_ids[faults.representation]
    stable_ids = tuple(identities[index] for index in faults.source_fault_ids)
    return dataclasses.replace(faults, source_fault_ids=stable_ids)


def _local_prior_faults(prior_faults, source_ids):
    if prior_faults is None:
        return None
    local = {}
    for representation, stable_ids in prior_faults.items():
        available = source_ids[representation]
        reverse = {identity: index for index, identity in enumerate(available)}
        required = set(stable_ids)
        missing = required.difference(reverse)
        if missing:
            raise RuntimeError("committed faults changed during stream growth")
        local[representation] = {reverse[identity] for identity in stable_ids}
    return local


def _check_same_model(previous, current) -> None:
    if previous.detector_ids != current.detector_ids:
        raise RuntimeError("stream growth changed a queued window's detectors")
    if previous.defect_positions != current.defect_positions:
        raise RuntimeError("stream growth changed a queued window's handoff")
    _check_same_faults(previous.graphlike_faults, current.graphlike_faults)
    _check_same_faults(previous.physical_faults, current.physical_faults)


def _check_same_faults(previous, current) -> None:
    has_previous_faults = previous is not None
    has_current_faults = current is not None
    if has_previous_faults != has_current_faults:
        raise RuntimeError(
            "stream growth changed a queued fault representation"
        )
    if previous is None:
        return
    earlier = _columns_by_identity(previous)
    later = _columns_by_identity(current)
    if earlier.keys() != later.keys():
        raise RuntimeError("stream growth changed a queued window's faults")
    for identity, column in earlier.items():
        _check_same_column(column, later[identity])


def _columns_by_identity(faults):
    columns = {}
    for column_index, identity in enumerate(faults.source_fault_ids):
        check_column = faults.check.getcol(column_index)
        observable_column = faults.observables.getcol(column_index)
        handoff = faults.boundary_flips.get(column_index, ())
        signature = (
            tuple(check_column.indices),
            tuple(observable_column.indices),
            bool(faults.owned[column_index]),
            handoff,
        )
        probability = float(faults.priors[column_index])
        columns[identity] = (signature, probability)
    return columns


def _check_same_column(previous, current) -> None:
    if previous[0] != current[0]:
        raise RuntimeError("stream growth changed a queued correction's effect")
    same_probability = math.isclose(
        previous[1], current[1], rel_tol=0, abs_tol=_PRIOR_PROBABILITY_TOLERANCE
    )
    if not same_probability:
        raise RuntimeError("stream growth changed a queued fault's probability")
