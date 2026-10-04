"""Window fault models of a growing repeated Stim circuit.

Stim error identities are their complete detector and observable effects,
not their changing catalog positions. WindowSlicer replays the qLDPC-style
ownership rule on each complete model, while published ids stay stable.
"""

import dataclasses
from typing import Optional

import decsim.detector_error_model.detector_formation as formation
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.records.circuits as circuit_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.windows as window_records


class GrowingStimModels:
    """Keep fault identities stable while a physical stream gains rounds.

    A window's model is sliced from a circuit assembled to one round past
    its buffer, the terminal fragment that defines the logical observable,
    whose detectors lie past the window. Growth therefore leaves a queued
    window's model unchanged.
    """

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        self.program = program
        self.requirement = requirement
        self.identities = _FaultIdentities()
        self.windows_by_index: dict[int, window_records.Window] = {}
        self.final_round_count: Optional[int] = None

    def finish(self, round_count: int) -> None:
        """Bind model finalization to the source's actual data readout."""
        if self.final_round_count is not None:
            raise RuntimeError("a physical stream can only finish once")
        self.final_round_count = round_count

    def for_window(
        self, window: window_records.Window
    ) -> fault_models.WindowErrorModel:
        """One window's model, after the windows before it took theirs.

        The slicer hands a boundary fault on in window order (qLDPC's rule), and
        each call slices a circuit assembled to this window's horizon, so the
        earlier windows are sliced again on it first.
        """
        slicer, source_ids, round_count = self._build_slicer(window)
        self.windows_by_index[window.window_index] = window
        model = None
        for window_index in sorted(self.windows_by_index):
            if window_index > window.window_index:
                break
            earlier_window = self.windows_by_index[window_index]
            model = _slice(slicer, earlier_window, round_count)
        return _stable_model(model, source_ids)

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
        source_ids = self.identities.register(slicer.catalogs)
        return slicer, source_ids, round_count


class _FaultIdentities:
    """An append-only namespace for complete Stim fault effects."""

    def __init__(self) -> None:
        self.ids_by_representation: dict = {}

    def register(
        self, catalogs: fault_models.FaultCatalogs
    ) -> dict[fault_models.FaultRepresentation, tuple[int, ...]]:
        source_ids = {}
        for representation, catalog in catalogs.by_representation.items():
            known = self.ids_by_representation.setdefault(representation, {})
            local_ids = []
            for index, detectors in enumerate(catalog.detector_sets):
                observables = catalog.observable_sets[index]
                decomposition = _decomposition_identity(
                    representation, index, catalogs
                )
                identity = (detectors, observables, decomposition)
                stable_id = known.setdefault(identity, len(known))
                local_ids.append(stable_id)
            source_ids[representation] = tuple(local_ids)
        return source_ids


def _decomposition_identity(representation, index, catalogs):
    if catalogs.link is None:
        return ()
    if representation is not fault_models.FaultRepresentation.PHYSICAL:
        return ()
    graphlike = catalogs.by_representation[
        fault_models.FaultRepresentation.GRAPHLIKE
    ]
    column = catalogs.link.getcol(index)
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
