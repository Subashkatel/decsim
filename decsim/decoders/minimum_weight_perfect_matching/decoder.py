"""The PyMatching adapter: minimum-weight perfect matching on one window.

PyMatching's Matching.from_check_matrix on the window's graphlike faults
with log-odds weights, parallel faults merged as independent errors
(the convention of Stim detector error models and of PyMatching's DEM
loader), one matching per live window model, warmed on three columns
before the first timed call. Higgott and Gidney, Sparse Blossom
(2303.15933) is the algorithm behind
the call. The same graph with the observable row appended as one more
check answers a forced-class job: the appended detector's bit pins the
observable's parity, so the solve is the minimum weight inside that
logical class (Gidney et al. 2312.04522 Sec. "Complementary gap").
"""

import dataclasses
from typing import Optional

import numpy
import pymatching
import scipy.sparse

import decsim.decoders.decoder as decoder_module
import decsim.decoders.minimum_weight_perfect_matching.weights as weights
import decsim.detector_error_model.fault_model_contracts as fault_models

WARM_UP_COLUMNS = 3


@dataclasses.dataclass(frozen=True)
class MatchingGraphs:
    """One window's matching graph, and the one that pins its observable.

    forced is None for a window whose model carries no single nonzero
    observable row: there is no parity to pin, so no class can be
    forced and the plain graph answers a forced job with no weight.
    """

    plain: pymatching.Matching
    forced: Optional[pymatching.Matching]


class PyMatchingDecoder(decoder_module.WindowDecoderBase):
    """Decode one window with PyMatching.

    Measured mode times ``matching.decode`` only, not syndrome extraction
    or result construction. Measurements are of one call on one thread:
    with several units they do not prove parallel hardware.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    answers_forced_logical_class = True

    def compile(self, faults, model=None) -> MatchingGraphs:
        """The matching graphs of one placed model, both warm."""
        del model
        # PyMatching normalises the matrix it is given in place; the placed
        # matrix is frozen, so it gets a copy (one per model, cached).
        # Faults sharing the same detector endpoints are independent error
        # mechanisms and merge into one edge with the combined probability,
        # the convention of Stim detector error models and of PyMatching's
        # DEM loader; a window restriction folds distinct faults onto the
        # same in-window endpoints, so a window graph always has some.
        check = faults.check.copy()
        edge_weights = self._weights_for(faults)
        error_probabilities = weights.finite_priors(faults.priors)
        matching = pymatching.Matching.from_check_matrix(
            check,
            weights=edge_weights,
            error_probabilities=error_probabilities,
            merge_strategy="independent",
        )
        _warm_up(matching, faults)
        forced = self._forced_matching(faults, edge_weights)
        return MatchingGraphs(plain=matching, forced=forced)

    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """The matching's correction, or an empty one marked invalid.

        PyMatching raises when a syndrome has odd parity in a boundaryless
        component; no valid plan produces one, so that case is reported as
        an empty correction with INVALID_CORRECTION rather than ending the
        run.
        """
        del model
        try:
            return backend.plain.decode(syndrome), None
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            fault_count = faults.check.shape[1]
            empty = numpy.zeros(fault_count, dtype=numpy.uint8)
            invalid = decoder_module.BackendDecodeStatus.INVALID_CORRECTION
            return empty, invalid

    def decode_forced_window(
        self, backend, model, faults, syndrome, forced_logical_class: int
    ) -> tuple:
        """The lightest correction whose observable parity is the class.

        The appended detector carries the class bit, so the matching
        must flip the observable an even or an odd number of times
        (Gidney et al. 2312.04522 Sec. "Complementary gap"). A window
        that pins no observable has no forced solve, and the plain
        decode answers it with no weight, which leaves the confidence
        without a gap and escalates the window.
        """
        if backend.forced is None:
            selected, decode_status = self.decode_window(
                backend, model, faults, syndrome
            )
            return selected, decode_status, None
        pinned = _with_pinned_observable(syndrome, forced_logical_class)
        selected, weight = backend.forced.decode(pinned, return_weight=True)
        return selected, None, float(weight)

    def _weights_for(self, faults):
        return weights.matching_weights(faults.priors)

    def _forced_matching(
        self, faults, edge_weights
    ) -> Optional[pymatching.Matching]:
        """The graph with the observable row as one more check, warm."""
        observables = faults.observables
        if observables.shape[0] != 1:
            return None
        dense_observables = observables.toarray()
        dense_row = dense_observables[0]
        if not dense_row.any():
            return None
        if not _stays_graphlike_with_observable(faults.check, dense_row):
            return None
        observable_check = scipy.sparse.csc_matrix(observables[0:1, :])
        stacked = scipy.sparse.vstack([faults.check, observable_check])
        pinned_check = stacked.tocsc()
        error_probabilities = weights.finite_priors(faults.priors)
        matching = pymatching.Matching.from_check_matrix(
            pinned_check,
            weights=edge_weights,
            error_probabilities=error_probabilities,
            merge_strategy="independent",
        )
        _warm_up_forced(matching, faults, dense_row)
        return matching


class UnweightedPyMatchingDecoder(PyMatchingDecoder):
    """Weight-oblivious MWPM: same matching graph, every edge at weight 1.

    A deliberately coarse weak tier (a hardware matcher without
    weighted-edge support): at circuit-level noise it decodes measurably
    worse than weighted MWPM because hook-error paths are no longer
    penalized.
    """

    def _weights_for(self, faults):
        fault_count = len(faults.priors)
        return numpy.ones(fault_count)


def _warm_up(matching, faults) -> None:
    """Decode a few one-column syndromes so the graph is built and warm.

    PyMatching builds its internal graph lazily and finishes warming
    only once it has matched real defects; a running software decoder
    has the window graph prebuilt and warm, so decode a few defects
    before the first timed call. Each warm-up syndrome is the detector
    set of one column, which that column alone explains, so it is
    satisfiable on any graph (a boundaryless toric component would
    reject an arbitrary detector pair).
    """
    check = faults.check
    detector_count = check.shape[0]
    warmed = 0
    for column in range(check.shape[1]):
        rows = _column_rows(check, column)
        if rows.size == 0:
            continue
        syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
        syndrome[rows] = 1
        matching.decode(syndrome)
        warmed += 1
        if warmed == WARM_UP_COLUMNS:
            return


def _stays_graphlike_with_observable(check, observable_row) -> bool:
    """Whether the pinned graph is still a matching graph.

    A matching edge touches at most two detectors, so a fault that
    already flips two detectors and the observable has no edge on the
    pinned graph (PyMatching refuses such a column). Such a window has
    no forced solve, which leaves its confidence without a gap.
    """
    dense_check = _dense_check(check)
    column_weights = dense_check.sum(axis=0)
    pinned_weights = column_weights + observable_row
    highest = pinned_weights.max(initial=0)
    is_graphlike = highest <= 2
    return bool(is_graphlike)


def _dense_check(check):
    if scipy.sparse.issparse(check):
        return check.toarray()
    return numpy.asarray(check)


def _warm_up_forced(matching, faults, observable_row) -> None:
    """Warm the pinned graph on the same one-column syndromes.

    Each column alone explains its detector set, and the appended
    detector carries that column's own observable bit, so the pinned
    syndrome is satisfiable.
    """
    check = faults.check
    detector_count = check.shape[0]
    warmed = 0
    for column in range(check.shape[1]):
        rows = _column_rows(check, column)
        if rows.size == 0:
            continue
        syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
        syndrome[rows] = 1
        observable_bit = int(observable_row[column])
        pinned = _with_pinned_observable(syndrome, observable_bit)
        matching.decode(pinned)
        warmed += 1
        if warmed == WARM_UP_COLUMNS:
            return


def _with_pinned_observable(syndrome, observable_bit: int):
    """The syndrome with the pinned graph's observable detector appended."""
    extended = numpy.concatenate([syndrome, [observable_bit]])
    return extended.astype(numpy.uint8)


def _column_rows(check, column: int):
    start = check.indptr[column]
    end = check.indptr[column + 1]
    return check.indices[start:end]
