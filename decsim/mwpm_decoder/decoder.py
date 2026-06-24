"""Runtime PyMatching decoder adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..adapters.window_decode_results import (
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ..message import DecodeJob, DecodeResult

if TYPE_CHECKING:
    from ..protocols import Decoder


class PyMatchingDecoder:
    """Decode one window with PyMatching and report simulated latency separately."""

    def __init__(self, latency_model: Decoder):
        self.latency_model = latency_model
        self._matchings: dict = {}

    def latency(self, job: DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Run real minimum-weight matching on the job's window error model."""
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        matching = self._matching_for_model(model)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, model)
        selected = matching.decode(syndrome)
        return result_from_selected_faults(job, model, selected)

    def _matching_for_model(self, model):
        """Return a cached matching graph for this window model."""
        import weakref
        import numpy as np
        import pymatching

        entry = self._matchings.get(id(model))
        matching = entry[1] if entry is not None and entry[0]() is model else None
        if matching is None:
            weights = np.log((1 - model.priors) / model.priors)
            matching = pymatching.Matching.from_check_matrix(model.check, weights=weights)
            self._matchings[id(model)] = (weakref.ref(model), matching)
        return matching
