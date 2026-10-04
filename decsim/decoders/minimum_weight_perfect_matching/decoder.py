"""The PyMatching adapter: minimum-weight perfect matching on one window.

PyMatching's Matching.from_check_matrix on the window's graphlike faults
with log-odds weights, parallel faults merged as independent errors (the
convention of Stim detector error models and PyMatching's DEM loader),
one matching per live window model. Sparse Blossom (Higgott and Gidney
2303.15933) is the algorithm behind the call. The same graph with the
observable row appended as one more check answers a forced-class job:
the appended detector's bit pins the observable's parity, so the solve
is the minimum weight inside that logical class (Gidney et al.
2312.04522 Sec. "Complementary gap").
"""

import dataclasses
import math
from typing import Optional

import numpy
import pymatching
import scipy.sparse

import decsim.config as config
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.weights as weights
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source

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

    Measured mode times ``matching.decode`` alone, one call on one
    thread: with several units it does not prove parallel hardware.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    decoder_evidence = decoding_records.FORCED_CLASS_SOLVES

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row as a run names it: measured, or at a preset latency.

        preset_latency_microseconds prices every decode at one fixed core
        latency (decoders.py PresetLatencyDecoder) instead of the host
        clock; the matching still decodes every window.
        """

        preset_latency_microseconds: Optional[float] = None

        def __post_init__(self) -> None:
            _check_preset_latency(self.preset_latency_microseconds)

        @property
        def name(self):
            """The row's word, or its preset latency in its place."""
            if self.preset_latency_microseconds is None:
                return "pymatching"
            return self.preset_latency_microseconds

        def build(self) -> "PyMatchingDecoder":
            """A fresh decoder of these settings."""
            if self.preset_latency_microseconds is None:
                return PyMatchingDecoder()
            latency_model = decoders.PresetLatencyDecoder(
                self.preset_latency_microseconds
            )
            return PyMatchingDecoder(latency_model)

    def __init__(self, latency_model=None):
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        # the latency model prices the decode and leaves the graph alone
        row = type(self)
        self.compile_key = (row, None)
        self.forced_solve_unavailable = trace_source.TraceSource()

    def compile(self, faults, model=None) -> MatchingGraphs:
        """The matching graphs of one placed model, both warm."""
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
        unpinnable_reason = _unpinnable_observable_reason(faults)
        if unpinnable_reason is not None:
            self.forced_solve_unavailable.fire(model, unpinnable_reason)
            return MatchingGraphs(plain=matching, forced=None)
        forced = self._forced_matching(faults, edge_weights)
        return MatchingGraphs(plain=matching, forced=forced)

    def decode_window(self, backend, model, faults, syndrome):
        """The matching's correction, or an empty one marked invalid.

        PyMatching raises on a syndrome with odd parity in a
        boundaryless component; no valid plan produces one, so it is
        reported as an empty correction with INVALID_CORRECTION, which
        leaves its shot unscored, rather than ending the run.
        """
        del model
        try:
            selected = backend.plain.decode(syndrome)
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            return _no_matching_decode(faults)
        return decoding_records.WindowDecode(selected)

    def decode_forced_window(
        self, backend, model, faults, syndrome, forced_logical_class: int
    ):
        """The lightest correction whose observable parity is the class.

        The appended detector carries the class bit (Gidney et al.
        2312.04522 Sec. "Complementary gap"). A window that pins no
        observable has no forced solve: the plain decode answers with no
        weight, and the window escalates. A class no matching reaches
        has no correction and probability zero, so its weight is +inf,
        the gap being the log-likelihood ratio of the two classes' best
        hypotheses (Sec. 4).
        """
        if backend.forced is None:
            return self.decode_window(backend, model, faults, syndrome)
        pinned = _with_pinned_observable(syndrome, forced_logical_class)
        try:
            selected, weight = backend.forced.decode(pinned, return_weight=True)
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            no_matching = _no_matching_decode(faults)
            return dataclasses.replace(
                no_matching, forced_class_weight=math.inf
            )
        return decoding_records.WindowDecode(
            selected, forced_class_weight=float(weight)
        )

    def _weights_for(self, faults):
        return weights.matching_weights(faults.priors)

    def _forced_matching(self, faults, edge_weights) -> pymatching.Matching:
        """The graph with the observable row as one more check, warm."""
        observables = faults.observables
        dense_observables = observables.toarray()
        dense_row = dense_observables[0]
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

    A deliberately coarse weak tier, a hardware matcher without weighted
    edges: at circuit-level noise it decodes worse than weighted MWPM
    because hook-error paths are no longer penalized.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row has no settings of its own."""

        # the word the reports name this row by
        name = "unweighted_pymatching"

        def build(self) -> "UnweightedPyMatchingDecoder":
            """A fresh decoder."""
            return UnweightedPyMatchingDecoder()

    def _weights_for(self, faults):
        fault_count = len(faults.priors)
        return numpy.ones(fault_count)


def _check_preset_latency(microseconds) -> None:
    """A preset core latency is a finite number at least zero, or none."""
    if microseconds is None:
        return
    is_finite = config.is_number(microseconds) and math.isfinite(microseconds)
    if is_finite and microseconds >= 0:
        return
    raise ValueError(
        "preset_latency_microseconds must be a finite nonnegative number "
        f"of microseconds, or None (got {microseconds!r})"
    )


def _no_matching_decode(faults) -> decoding_records.WindowDecode:
    """PyMatching found no perfect matching: no correction, marked invalid."""
    fault_count = faults.check.shape[1]
    return backend_outcome.no_correction_decode(
        decoding_records.BackendDecodeStatus.INVALID_CORRECTION,
        decoding_records.BackendFailureReason.NO_PERFECT_MATCHING,
        fault_count,
    )


def _warm_up(matching, faults) -> None:
    """Decode a few one-column syndromes so the graph is built and warm.

    PyMatching builds its graph lazily and warms only once it has
    matched real defects, while a running decoder has it prebuilt. Each
    warm-up syndrome is one column's detector set, which that column
    alone explains, so it is satisfiable on any graph.
    """
    syndromes = _warm_up_syndromes(faults.check)
    for _column, syndrome in syndromes:
        matching.decode(syndrome)


def _warm_up_syndromes(check) -> list:
    """(column, syndrome) for the first columns that flip any detector."""
    detector_count = check.shape[0]
    syndromes = []
    for column in range(check.shape[1]):
        if len(syndromes) == WARM_UP_COLUMNS:
            break
        rows = basis_split.column_rows(check, column)
        if not rows:
            continue
        syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
        row_indices = list(rows)
        syndrome[row_indices] = 1
        syndromes.append((column, syndrome))
    return syndromes


def _unpinnable_observable_reason(faults) -> Optional[str]:
    """Why this model pins no logical class, or None when it pins one.

    The observable row is appended as one more detector whose bit is the
    class (Gidney et al. 2312.04522 lines 828-833), which needs one
    nonzero observable row and every observable-flipping fault to touch
    at most one detector.
    """
    observables = faults.observables
    row_count = observables.shape[0]
    if row_count != 1:
        return (
            f"the window model carries {row_count} logical observable "
            "rows and the appended detector pins exactly one"
        )
    dense_observables = observables.toarray()
    dense_row = dense_observables[0]
    if not dense_row.any():
        return "no fault of the window flips its logical observable"
    if _stays_graphlike_with_observable(faults.check, dense_row):
        return None
    return (
        "an observable-flipping fault already flips two detectors, so "
        "the appended observable detector gives it three and the pinned "
        "graph is no matching graph; the recipe wants the observable "
        "along a boundary (Gidney et al. 2312.04522 lines 828-833)"
    )


def _stays_graphlike_with_observable(check, observable_row) -> bool:
    """Whether the pinned graph is still a matching graph.

    A matching edge touches at most two detectors, so a fault that
    already flips two detectors and the observable has no edge on the
    pinned graph (PyMatching refuses such a column).
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
    syndromes = _warm_up_syndromes(faults.check)
    for column, syndrome in syndromes:
        observable_bit = int(observable_row[column])
        pinned = _with_pinned_observable(syndrome, observable_bit)
        matching.decode(pinned)


def _with_pinned_observable(syndrome, observable_bit: int):
    """The syndrome with the pinned graph's observable detector appended."""
    extended = numpy.concatenate([syndrome, [observable_bit]])
    return extended.astype(numpy.uint8)
