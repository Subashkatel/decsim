"""Runtime PyMatching decoder adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..adapters.window_decode_results import (
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
)
from ..detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)

if TYPE_CHECKING:
    from ..protocols import Decoder


class PyMatchingDecoder:
    """Decode one window with PyMatching and report simulated latency separately."""

    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, latency_model: Decoder):
        self.latency_model = latency_model
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
        selected = matching.decode(syndrome)
        return result_from_selected_faults(job, model, faults, selected)

    def _matching_for_model(self, faults):
        """Return a cached matching graph for this window model."""
        import weakref
        import pymatching

        entry = self._matchings.get(id(faults))
        matching = entry[1] if entry is not None and entry[0]() is faults else None
        if matching is None:
            from ..detector_error_model import validate_graphlike_matrices

            validate_graphlike_matrices(
                faults.check,
                faults.observables,
                location="PyMatching window model",
            )
            matching = pymatching.Matching.from_check_matrix(
                faults.check, weights=self._weights_for(faults))
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
