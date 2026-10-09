"""Sliding-window decoding written from its sources, beside decsim's.

The referee for the machine's windows. It reads Stim's detector error
model itself and lays out, fills and commits each window by the rules
below, never through decsim's window planner, slicer or ownership code;
only the decode of one window is a decsim decoder row, called on the
matrices built here.

Skoric et al. 2209.08552 lines 188-197: a window decodes its commit and
buffer rounds and keeps the corrections of its commit region; lines
260-261: it also commits the edges leaving the commit region. qLDPC's
SequentialWindowDecoder (qldpc/decoders/sinter.py lines 470-510, qldpc
0.3.3 as installed) gives the columns: every fault flipping a detector
of the window that no earlier window committed, of which the window
commits those flipping a detector of its commit rounds. Its decode loop
(lines 635-652) reads the bare syndrome plus the effect of every
correction committed so far. cudaqx's sliding window
(libs/qec/lib/decoders/sliding_window.cpp lines 311-322) commits the
same columns, everything before the next window's first column. Every
window strides its commit rounds, and the last commits what is left.

A union-find decode breaks ties by edge order, so the matrices keep
Stim's order: rows by detector index, columns by a fault's first
appearance in the model, the order of qLDPC's column masks.
"""

import dataclasses
import math

import numpy
import scipy.sparse

import decsim.records.fault_model_contracts as fault_models

WEAK = "weak"


@dataclasses.dataclass(frozen=True)
class Fault:
    """One graphlike fault: what it flips, its rounds, its probability."""

    detectors: tuple
    observables: tuple
    rounds: tuple
    prior: float


@dataclasses.dataclass(frozen=True)
class Commit:
    """What one window commits: the detectors and observables it flips."""

    window_index: int
    tier: str
    detectors: tuple
    observables: tuple


class SlidingWindowReferee:
    """Decodes a memory circuit's shots window by window, from Stim alone.

    Every window commits window_rounds rounds and reads window_rounds
    more; weak_decoder is a built decsim decoder row.
    """

    def __init__(
        self, circuit, round_count: int, window_rounds: int, weak_decoder
    ) -> None:
        rounds_by_detector = detector_rounds(circuit, round_count)
        self.faults = circuit_faults(circuit, rounds_by_detector)
        self.rounds_by_detector = rounds_by_detector
        self.round_count = round_count
        self.window_rounds = window_rounds
        self.weak_decoder = weak_decoder
        self.observable_count = circuit.num_observables

    def weak_commits(self, detection_events) -> list:
        """Each window's commit, in window order."""
        syndrome = numpy.array(detection_events, dtype=numpy.uint8)
        committed = set()
        commits = []
        strides = self.round_count / self.window_rounds
        window_count = math.ceil(strides)
        for window_index in range(window_count):
            commit = self._weak_commit(window_index, syndrome, committed)
            commits.append(commit)
        return commits

    def _weak_commit(self, window_index: int, syndrome, committed: set):
        """Decode one window and commit its commit region's faults."""
        commit_lo = window_index * self.window_rounds + 1
        stride_end = commit_lo + self.window_rounds - 1
        commit_hi = min(stride_end, self.round_count)
        last_round = commit_hi + self.window_rounds
        rows = self._rows(commit_lo, last_round)
        columns = self._columns(rows, committed)
        owned = self._owned(columns, commit_lo, commit_hi)
        selected = self._weak_selection(rows, columns, owned, syndrome)
        return self._commit(
            window_index, WEAK, columns, owned, selected, syndrome, committed
        )

    def _rows(self, first_round: int, last_round: int) -> list:
        """The detectors of the rounds a window reads, in index order."""
        rows = []
        for detector, round_index in self.rounds_by_detector.items():
            if first_round <= round_index <= last_round:
                rows.append(detector)
        return sorted(rows)

    def _columns(self, rows: list, committed: set) -> list:
        """The faults flipping a row that no earlier window committed."""
        row_set = set(rows)
        columns = []
        for fault_index, fault in enumerate(self.faults):
            if fault_index in committed:
                continue
            if row_set.isdisjoint(fault.detectors):
                continue
            columns.append(fault_index)
        return columns

    def _owned(self, columns: list, commit_lo: int, commit_hi: int) -> list:
        """Whether each column flips a detector of the commit rounds."""
        owned = []
        for fault_index in columns:
            rounds = self.faults[fault_index].rounds
            touches = any(
                commit_lo <= round_index <= commit_hi for round_index in rounds
            )
            owned.append(touches)
        return owned

    def _weak_selection(self, rows, columns, owned, syndrome) -> list:
        """The weak row's correction over the window's columns."""
        placed = self._placed_faults(rows, columns, owned)
        window_syndrome = syndrome[rows]
        backend = self.weak_decoder.compile(placed, None)
        answer = self.weak_decoder.decode_window(
            backend, None, placed, window_syndrome
        )
        return list(answer.selected_faults)

    def _placed_faults(self, rows, columns, owned):
        """The window's matrices as the record a decoder row reads."""
        check = self._check_matrix(rows, columns)
        observables = self._observable_matrix(columns)
        column_priors = []
        for fault_index in columns:
            column_priors.append(self.faults[fault_index].prior)
        priors = numpy.array(column_priors)
        owned_mask = numpy.array(owned, dtype=bool)
        return fault_models.PlacedFaultModel(
            representation=fault_models.FaultRepresentation.GRAPHLIKE,
            check=check,
            priors=priors,
            observables=observables,
            owned=owned_mask,
            source_fault_ids=tuple(columns),
            boundary_flips={},
        )

    def _check_matrix(self, rows: list, columns: list):
        """Rows by columns: the window's detectors each fault flips."""
        row_by_detector = {}
        for row, detector in enumerate(rows):
            row_by_detector[detector] = row
        entry_rows = []
        entry_columns = []
        for column, fault_index in enumerate(columns):
            fault = self.faults[fault_index]
            local_rows = _local_rows(fault, row_by_detector)
            entry_rows.extend(local_rows)
            column_entries = [column] * len(local_rows)
            entry_columns.extend(column_entries)
        shape = (len(rows), len(columns))
        return _ones_at(entry_rows, entry_columns, shape)

    def _observable_matrix(self, columns: list):
        """Observables by columns: every observable each fault flips."""
        entry_rows = []
        entry_columns = []
        for column, fault_index in enumerate(columns):
            for observable in self.faults[fault_index].observables:
                entry_rows.append(observable)
                entry_columns.append(column)
        shape = (self.observable_count, len(columns))
        return _ones_at(entry_rows, entry_columns, shape)

    def _commit(
        self, window_index, tier, columns, owned, selected, syndrome, committed
    ) -> Commit:
        """Take the owned columns out of later windows; apply the selected.

        A selected fault's whole detector effect leaves the syndrome, so a
        later window reads the bare syndrome plus every committed
        correction (qLDPC's decode loop).
        """
        detectors = set()
        observables = [0] * self.observable_count
        for column, fault_index in enumerate(columns):
            if not owned[column]:
                continue
            committed.add(fault_index)
            if not selected[column]:
                continue
            fault = self.faults[fault_index]
            flipped_rows = list(fault.detectors)
            syndrome[flipped_rows] ^= 1
            detectors.symmetric_difference_update(fault.detectors)
            for observable in fault.observables:
                observables[observable] ^= 1
        flipped = tuple(sorted(detectors))
        return Commit(window_index, tier, flipped, tuple(observables))


def prediction(commits: list) -> tuple:
    """A shot's predicted observables: every commit's flips together."""
    flips = [0] * len(commits[0].observables)
    for commit in commits:
        for observable, flip in enumerate(commit.observables):
            flips[observable] ^= flip
    return tuple(flips)


def detector_rounds(circuit, round_count: int) -> dict:
    """Each detector's round, from the last of Stim's coordinates.

    Layer t of a generated circuit is formed after round t, so it is
    round t + 1; the data readout's layer, one past the last round, is
    read with the last round.
    """
    coordinates = circuit.get_detector_coordinates()
    rounds = {}
    for detector, detector_coordinates in coordinates.items():
        layer = int(detector_coordinates[-1])
        next_round = layer + 1
        rounds[detector] = min(next_round, round_count)
    return rounds


def circuit_faults(circuit, rounds_by_detector: dict) -> list:
    """The decomposed model's components, parallel ones merged.

    Each `^`-separated component of an error is one graphlike fault, and
    components flipping the same detectors and observables merge as
    independent errors, p(1-q) + q(1-p), PyMatching's merge for a Stim
    model.
    """
    model = circuit.detector_error_model(decompose_errors=True)
    priors_by_identity = {}
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        arguments = instruction.args_copy()
        probability = arguments[0]
        targets = instruction.targets_copy()
        for identity in _components(targets):
            earlier = priors_by_identity.get(identity, 0.0)
            merged = _merged(earlier, probability)
            priors_by_identity[identity] = merged
    faults = []
    for identity, prior in priors_by_identity.items():
        detectors, observables = identity
        rounds = []
        for detector in detectors:
            rounds.append(rounds_by_detector[detector])
        fault = Fault(detectors, observables, tuple(rounds), prior)
        faults.append(fault)
    return faults


def _components(targets) -> list:
    """An error's components as (detectors, observables), each sorted."""
    components = []
    detectors = []
    observables = []
    for target in targets:
        if target.is_separator():
            _close(components, detectors, observables)
            detectors = []
            observables = []
            continue
        if target.is_relative_detector_id():
            detectors.append(target.val)
            continue
        observables.append(target.val)
    _close(components, detectors, observables)
    return components


def _close(components: list, detectors: list, observables: list) -> None:
    sorted_detectors = tuple(sorted(detectors))
    sorted_observables = tuple(sorted(observables))
    components.append((sorted_detectors, sorted_observables))


def _local_rows(fault: Fault, row_by_detector: dict) -> list:
    """The window rows of the fault's detectors that lie in the window."""
    local_rows = []
    for detector in fault.detectors:
        if detector in row_by_detector:
            local_rows.append(row_by_detector[detector])
    return local_rows


def _merged(earlier: float, probability: float) -> float:
    return earlier * (1 - probability) + probability * (1 - earlier)


def _ones_at(entry_rows: list, entry_columns: list, shape: tuple):
    ones = numpy.ones(len(entry_rows), dtype=numpy.uint8)
    return scipy.sparse.csc_matrix((ones, (entry_rows, entry_columns)), shape)
