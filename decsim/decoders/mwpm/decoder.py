"""Runtime PyMatching decoder adapter."""

from __future__ import annotations

import time

from typing import TYPE_CHECKING, Optional

from ..window_decode_results import (
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
        selected = matching.decode(syndrome)
        self.last_decode_ns = time.perf_counter_ns() - started_ns
        return result_from_selected_faults(job, model, faults, selected)

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
            matching = pymatching.Matching.from_check_matrix(
                faults.check, weights=self._weights_for(faults))
            # PyMatching builds its internal graph lazily and finishes warming
            # only once it has matched real defects; a running software decoder
            # has the window graph prebuilt and warm, so decode a few defect
            # pairs before the first timed call.
            import numpy
            detectors = faults.check.shape[0]
            for first in range(0, min(detectors - 1, 6), 2):
                syndrome = numpy.zeros(detectors, dtype=numpy.uint8)
                syndrome[first] = syndrome[first + 1] = 1
                matching.decode(syndrome)
            self._matchings[id(faults)] = (weakref.ref(faults), matching)
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
