"""Runtime PyMatching decoder adapter."""

from __future__ import annotations

import time

from typing import TYPE_CHECKING, Optional

from ..window_decode_results import (
    BackendDecodeStatus,
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ...message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
)
from ...detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)

if TYPE_CHECKING:
    from ...protocols import Decoder


def _match_best_effort(matching, syndrome, faults):
    """PyMatching raises when a syndrome has odd parity in a boundaryless
    component; no valid plan produces one, so that case is reported as an
    empty correction with INVALID_CORRECTION rather than ending the run."""
    import numpy as np

    try:
        return matching.decode(syndrome), None
    except ValueError as error:
        if "perfect matching" not in str(error):
            raise
        return (np.zeros(faults.check.shape[1], dtype=np.uint8),
                BackendDecodeStatus.INVALID_CORRECTION)


class PyMatchingDecoder:
    """Decode one window with PyMatching; simulated latency comes from a
    latency model, or, with ``latency_model=None``, from the measured wall
    clock of the matching call itself (software decoder on this host).

    Measured mode times ``matching.decode`` only, not syndrome extraction or
    result construction, and needs the DecoderEngine, which decodes first and
    holds the unit busy for the measured time; ``latency()`` alone cannot know
    the time before the call. Measurements are of one call on one thread: with
    several units they do not prove parallel hardware.
    """

    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, latency_model: Optional[Decoder] = None):
        self.latency_model = latency_model
        self.measures_wall_clock = latency_model is None
        self.last_decode_ns: Optional[int] = None
        self._matchings: dict = {}

    def run_seed_children(self):
        """Expose the latency model that controls simulated service time."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "latency_model"),),
                self.latency_model,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        if self.measures_wall_clock:
            raise RuntimeError("measured wall-clock timing needs the DecoderEngine: "
                               "it decodes first and charges the measured time")
        return self.latency_model.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Run real minimum-weight matching on the job's window error model."""
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        matching = self._matching_for_model(faults)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, faults)
        started_ns = time.perf_counter_ns()
        selected, decode_status = _match_best_effort(matching, syndrome, faults)
        self.last_decode_ns = time.perf_counter_ns() - started_ns
        return result_from_selected_faults(job, model, faults, selected,
                                           decode_status=decode_status)

    def _matching_for_model(self, faults):
        """Return a cached matching graph for this window model."""
        import weakref
        import pymatching

        entry = self._matchings.get(id(faults))
        matching = entry[1] if entry is not None and entry[0]() is faults else None
        if matching is None:
            from ...detector_error_model.fault_identity_validation import (
                validate_graphlike_matrices,
            )

            validate_graphlike_matrices(
                faults.check,
                faults.observables,
                location="PyMatching window model",
            )
            # PyMatching normalises the matrix it is given in place; the placed
            # matrix is frozen, so it gets a copy (one per model, cached)
            matching = pymatching.Matching.from_check_matrix(
                faults.check.copy(), weights=self._weights_for(faults))
            # PyMatching builds its internal graph lazily and finishes warming
            # only once it has matched real defects; a running software decoder
            # has the window graph prebuilt and warm, so decode a few defects
            # before the first timed call. Each warm-up syndrome is the detector
            # set of one column, which that column alone explains, so it is
            # satisfiable on any graph (a boundaryless toric component would
            # reject an arbitrary detector pair).
            import numpy
            detectors = faults.check.shape[0]
            warmed = 0
            check = faults.check
            for column in range(check.shape[1]):
                rows = check.indices[check.indptr[column]:check.indptr[column + 1]]
                if rows.size == 0:
                    continue
                syndrome = numpy.zeros(detectors, dtype=numpy.uint8)
                syndrome[rows] = 1
                matching.decode(syndrome)
                warmed += 1
                if warmed == 3:
                    break
            # the cache entry lives exactly as long as the placed model: id()
            # values are recycled by CPython, and a dead entry would otherwise
            # accumulate once per distinct window model of a long run
            model_identity = id(faults)

            def discard_dead_model(reference) -> None:
                current = self._matchings.get(model_identity)
                if current is not None and current[0] is reference:
                    del self._matchings[model_identity]

            self._matchings[model_identity] = (
                weakref.ref(faults, discard_dead_model), matching)
        return matching

    def _weights_for(self, faults):
        from .weights import matching_weights
        return matching_weights(faults.priors)


class UnweightedPyMatchingDecoder(PyMatchingDecoder):
    """Weight-oblivious MWPM: same matching graph, every edge at weight 1.

    A deliberately COARSE weak tier (a hardware matcher without weighted-edge
    support): at circuit-level noise it decodes measurably worse than
    weighted MWPM because hook-error paths are no longer penalized."""

    def _weights_for(self, faults):
        import numpy as np
        return np.ones(len(faults.priors))
