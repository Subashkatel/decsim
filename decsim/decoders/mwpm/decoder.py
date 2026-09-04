"""The PyMatching adapter: minimum-weight perfect matching on one window.

PyMatching's Matching.from_check_matrix on the window's graphlike faults
with log-odds weights, parallel faults merged as independent errors
(the convention of Stim detector error models and of PyMatching's DEM
loader), one matching per live window model, warmed on three columns
before the first timed call. Higgott and Gidney, Sparse Blossom
(2303.15933, tmp/uf-decoder-research/papers) is the algorithm behind
the call.
"""

import time
import weakref
from typing import Optional

import numpy
import pymatching

import decsim.decoders.mwpm.weights as weights
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message
import decsim.ports as ports

WARM_UP_COLUMNS = 3


class PyMatchingDecoder:
    """Decode one window with PyMatching.

    Simulated latency comes from a latency model, or, with
    ``latency_model=None``, from the measured wall clock of the matching
    call itself (software decoder on this host). Measured mode times
    ``matching.decode`` only, not syndrome extraction or result
    construction, and needs the DecoderEngine, which decodes first and
    holds the unit busy for the measured time; ``latency()`` alone cannot
    know the time before the call. Measurements are of one call on one
    thread: with several units they do not prove parallel hardware.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, latency_model: Optional[ports.Decoder] = None):
        self.latency_model = latency_model
        self.measures_wall_clock = latency_model is None
        self.last_decode_ns: Optional[int] = None
        self.matching_by_model: dict = {}

    def run_seed_children(self) -> tuple:
        """The latency model that controls simulated service time."""
        path = (message.RunSeedPathSegment("field", "latency_model"),)
        child = message.RunSeedChild(path, self.latency_model)
        return (child,)

    def latency(self, job: message.DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        if self.measures_wall_clock:
            raise RuntimeError(
                "measured wall-clock timing needs the DecoderEngine: it "
                "decodes first and charges the measured time"
            )
        return self.latency_model.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """Run real minimum-weight matching on the job's window error model."""
        model = job.dem
        if model is None:
            return message.DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(
            fault_models.FaultRepresentation.GRAPHLIKE
        )
        matching = self.compile(faults)
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        started_ns = time.perf_counter_ns()
        selected, decode_status = _match_best_effort(matching, syndrome, faults)
        finished_ns = time.perf_counter_ns()
        self.last_decode_ns = finished_ns - started_ns
        return window_decode_results.result_from_selected_faults(
            job, model, faults, selected, decode_status=decode_status
        )

    def compile(self, faults):
        """The matching graph of one placed model, kept while the model lives.

        The cache entry lives exactly as long as the placed model: id()
        values are recycled by CPython, and a dead entry would otherwise
        accumulate once per distinct window model of a long run.
        """
        model_identity = id(faults)
        entry = self.matching_by_model.get(model_identity)
        if entry is not None:
            reference, matching = entry
            if reference() is faults:
                return matching
        matching = self._build_matching(faults)
        _warm_up(matching, faults)

        def discard_dead_model(reference) -> None:
            current = self.matching_by_model.get(model_identity)
            if current is not None and current[0] is reference:
                del self.matching_by_model[model_identity]

        reference = weakref.ref(faults, discard_dead_model)
        self.matching_by_model[model_identity] = (reference, matching)
        return matching

    def _build_matching(self, faults):
        fault_identity.validate_graphlike_matrices(
            faults.check,
            faults.observables,
            location="PyMatching window model",
        )
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
        return pymatching.Matching.from_check_matrix(
            check,
            weights=edge_weights,
            error_probabilities=error_probabilities,
            merge_strategy="independent",
        )

    def _weights_for(self, faults):
        return weights.matching_weights(faults.priors)


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


def _match_best_effort(matching, syndrome, faults) -> tuple:
    """The matching's correction, or an empty one marked invalid.

    PyMatching raises when a syndrome has odd parity in a boundaryless
    component; no valid plan produces one, so that case is reported as
    an empty correction with INVALID_CORRECTION rather than ending the
    run.
    """
    try:
        return matching.decode(syndrome), None
    except ValueError as error:
        if "perfect matching" not in str(error):
            raise
        fault_count = faults.check.shape[1]
        empty = numpy.zeros(fault_count, dtype=numpy.uint8)
        invalid = window_decode_results.BackendDecodeStatus.INVALID_CORRECTION
        return empty, invalid


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


def _column_rows(check, column: int):
    start = check.indptr[column]
    end = check.indptr[column + 1]
    return check.indices[start:end]
