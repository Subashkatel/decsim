"""Sliding-window decoding written from its sources, beside decsim's.

The referee for the machine's windows. It reads Stim's detector error
model itself and lays out, fills and commits each window by the rules
below, never through decsim's window planner, slicer, ownership code or
strong regions; only the decode of one weak window is a decsim decoder
row, called on the matrices built here.

Skoric et al. 2209.08552 Sec. I B (lines 195-206): a window decodes its
commit and buffer rounds and keeps the corrections of its commit
region; Fig. 2 (line 271): it also commits the edges leaving the commit
region. qLDPC's SequentialWindowDecoder (qldpc/decoders/sinter.py lines
470-510, qldpc 0.3.3 as installed) gives the columns: every fault
flipping a detector of the window that no earlier window committed, of
which the window commits those flipping a detector of its commit
rounds. Its decode loop (lines 635-652) reads the bare syndrome plus the
effect of every correction committed so far. cudaqx's sliding window
(libs/qec/lib/decoders/sliding_window.cpp lines 311-322) commits the
same columns, everything before the next window's first column. Every
window strides its commit rounds, and the last commits what is left.

An escalated window opens Toshio et al.'s double window (2510.25222
Sec. III C and Fig. 12, lines 1242-1251 and 1260-1263): the strong
decoder takes r_com + 2 r_buf rounds from the escalated commit and
decodes them once the weak decoder has determined the boundary
conditions at both ends, and the weak chain restarts past them, here
reading back one buffer region. The strong input is the bare syndrome
plus the corrections committed by the decodes before it (Bombin et
al. 2303.04846, lines 822-835 of tmp/papers/2303.04846.txt), and its
decoder is PyMatching on the region's faults. The restart window is
decoded first, so it commits the faults crossing into the region
(Skoric Fig. 2).

A union-find decode breaks ties by edge order, so the matrices keep
Stim's order: rows by detector index, columns by a fault's first
appearance in the model, the order of qLDPC's column masks.
"""

import dataclasses
import math
from typing import Optional

import numpy
import pymatching
import scipy.sparse

import decsim.records.fault_model_contracts as fault_models

WEAK = "weak"
STRONG = "strong"


@dataclasses.dataclass(frozen=True)
class Fault:
    """One graphlike fault: what it flips, its rounds, its probability."""

    detectors: tuple
    observables: tuple
    rounds: tuple
    prior: float


@dataclasses.dataclass(frozen=True)
class Commit:
    """What one window commits: the detectors and observables it flips.

    A double window's observables include those its weak decode's
    crossing commit flips, also kept apart in crossing_observables.
    """

    window_index: int
    tier: str
    detectors: tuple
    observables: tuple
    crossing_observables: tuple = ()


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

    def commits(self, detection_events, escalated_windows=frozenset()) -> list:
        """Each commit of one shot, given the windows whose verdict escalated.

        A kept window commits its weak decode. An escalated window
        commits only what its weak decode owns behind its first round,
        the restart's crossing faults, which pin the region before it
        and its own; those flips are booked with its strong commit, as
        decsim's frame books them.
        """
        syndrome = numpy.array(detection_events, dtype=numpy.uint8)
        shot = _Shot(syndrome, set(), [])
        strides = self.round_count / self.window_rounds
        window_count = math.ceil(strides)
        window_index = 0
        first_round = None
        while window_index < window_count:
            window = self._window(window_index, first_round)
            if window_index in escalated_windows:
                window_index, first_round = self._escalate(window, shot)
                continue
            self._keep(window, shot)
            window_index += 1
            first_round = None
        self._decode_waiting_region(shot)
        return shot.commits

    def _window(self, window_index: int, first_round: Optional[int]):
        """The window's rounds; a restart reads back from first_round."""
        commit_lo = window_index * self.window_rounds + 1
        stride_end = commit_lo + self.window_rounds - 1
        commit_hi = min(stride_end, self.round_count)
        last_round = commit_hi + self.window_rounds
        if first_round is None:
            first_round = commit_lo
        return _Window(
            window_index, first_round, commit_lo, commit_hi, last_round
        )

    def _keep(self, window: "_Window", shot: "_Shot") -> None:
        """Commit the weak decode; a region waiting on it is decoded next."""
        columns, owned, selected = self._weak_decode(window, shot)
        commit = self._commit(
            window.index, WEAK, columns, owned, selected, shot
        )
        shot.commits.append(commit)
        self._decode_waiting_region(shot)

    def _escalate(self, window: "_Window", shot: "_Shot") -> tuple:
        """Open the window's region; the restart's index and first round."""
        columns, owned, selected = self._weak_decode(window, shot)
        crossing = self._crossing(columns, owned, window.commit_lo)
        crossing_commit = self._commit(
            window.index, STRONG, columns, crossing, selected, shot
        )
        self._decode_waiting_region(shot)
        region_end = window.commit_lo + 3 * self.window_rounds - 1
        last_round = min(region_end, self.round_count)
        shot.waiting_region = _Region(
            window.index,
            window.commit_lo,
            last_round,
            crossing_commit.observables,
        )
        restart_strides = last_round / self.window_rounds
        restart_index = math.ceil(restart_strides)
        restart_first_round = last_round - self.window_rounds + 1
        return restart_index, restart_first_round

    def _decode_waiting_region(self, shot: "_Shot") -> None:
        """Decode the region whose far face has just been committed."""
        region = shot.waiting_region
        if region is None:
            return
        shot.waiting_region = None
        rows = self._rows(region.first_round, region.last_round)
        columns = self._columns(rows, shot.committed)
        owned = self._owned(columns, region.first_round, region.last_round)
        placed = self._placed_faults(rows, columns, owned)
        region_syndrome = shot.syndrome[rows]
        selected = _matching_selection(placed, region_syndrome)
        commit = self._commit(
            region.window_index, STRONG, columns, owned, selected, shot
        )
        observables = _flips_together(
            commit.observables, region.crossing_observables
        )
        booked = dataclasses.replace(
            commit,
            observables=observables,
            crossing_observables=region.crossing_observables,
        )
        shot.commits.append(booked)

    def _weak_decode(self, window: "_Window", shot: "_Shot") -> tuple:
        """The window's columns, which it owns, and the weak correction."""
        rows = self._rows(window.first_round, window.last_round)
        columns = self._columns(rows, shot.committed)
        owned = self._owned(columns, window.commit_lo, window.commit_hi)
        selected = self._weak_selection(rows, columns, owned, shot.syndrome)
        return columns, owned, selected

    def _crossing(self, columns: list, owned: list, commit_lo: int) -> list:
        """Whether each column is owned and flips a round before commit_lo."""
        crossing = []
        for column, fault_index in enumerate(columns):
            rounds = self.faults[fault_index].rounds
            earliest = min(rounds)
            is_crossing = owned[column] and earliest < commit_lo
            crossing.append(is_crossing)
        return crossing

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
        self, window_index, tier, columns, owned, selected, shot
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
            shot.committed.add(fault_index)
            if not selected[column]:
                continue
            fault = self.faults[fault_index]
            flipped_rows = list(fault.detectors)
            shot.syndrome[flipped_rows] ^= 1
            detectors.symmetric_difference_update(fault.detectors)
            for observable in fault.observables:
                observables[observable] ^= 1
        flipped = tuple(sorted(detectors))
        return Commit(window_index, tier, flipped, tuple(observables))


@dataclasses.dataclass(frozen=True)
class _Window:
    """One weak window: the rounds it reads and the rounds it commits."""

    index: int
    first_round: int
    commit_lo: int
    commit_hi: int
    last_round: int


@dataclasses.dataclass(frozen=True)
class _Region:
    """A strong region waiting for its far face to be committed."""

    window_index: int
    first_round: int
    last_round: int
    crossing_observables: tuple


@dataclasses.dataclass
class _Shot:
    """One shot's running syndrome, committed faults and commits."""

    syndrome: numpy.ndarray
    committed: set
    commits: list
    waiting_region: Optional[_Region] = None


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


def _matching_selection(placed, region_syndrome) -> list:
    """PyMatching's minimum-weight correction over the region's columns.

    Log-odds weights with parallel faults merged as independent errors,
    the convention of PyMatching's loader for a Stim model.
    """
    priors = placed.priors
    negated = -priors
    log_survival = numpy.log1p(negated)
    log_prior = numpy.log(priors)
    weights = log_survival - log_prior
    check = placed.check.copy()
    matching = pymatching.Matching.from_check_matrix(
        check,
        weights=weights,
        error_probabilities=priors,
        merge_strategy="independent",
    )
    selected = matching.decode(region_syndrome)
    return list(selected)


def _flips_together(first: tuple, second: tuple) -> tuple:
    flips = []
    for first_flip, second_flip in zip(first, second, strict=True):
        flip = first_flip ^ second_flip
        flips.append(flip)
    return tuple(flips)


def _ones_at(entry_rows: list, entry_columns: list, shape: tuple):
    ones = numpy.ones(len(entry_rows), dtype=numpy.uint8)
    return scipy.sparse.csc_matrix((ones, (entry_rows, entry_columns)), shape)
