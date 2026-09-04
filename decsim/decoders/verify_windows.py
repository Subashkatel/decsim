"""The referee: every window re-decoded by Tesseract and compared.

After every tier decode, the official Tesseract backend re-decodes the
same window input and the owned observable contributions are compared;
the count of disagreements is the run's accuracy audit. Never priced:
the engine reads timing from the inner decoder alone. The linked fault
models are built whole-circuit, with memory linear in circuit length
(verified through d=9 x 1000 rounds).
"""

import numpy

import decsim.decoders.tesseract.window_decoder as tesseract_window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


class TesseractCheckedDecoder:
    """A decoder whose every result Tesseract re-decodes and checks."""

    def __init__(self, inner):
        self.inner = inner
        self.referee = tesseract_window_decoder.TesseractWindowDecoder()
        # The referee reads the physical view, the tier the graphlike one.
        self.fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
        self.windows_checked = 0
        self.window_disagreements = 0

    @property
    def measures_wall_clock(self) -> bool:
        """Whether the inner decoder prices its measured wall clock."""
        return getattr(self.inner, "measures_wall_clock", False)

    @property
    def last_decode_ns(self):
        """The inner decoder's last measured decode, in nanoseconds."""
        return self.inner.last_decode_ns

    def run_seed_children(self) -> tuple:
        """The referee under its own path, the inner decoder's under inner."""
        referee_path = (message.RunSeedPathSegment("field", "referee"),)
        children = [message.RunSeedChild(referee_path, self.referee)]
        inner_children = getattr(self.inner, "run_seed_children", None)
        if inner_children is None:
            return tuple(children)
        inner_segment = (message.RunSeedPathSegment("field", "inner"),)
        for child in inner_children():
            path = inner_segment + child.relative_path
            inner_child = message.RunSeedChild(path, child.child)
            children.append(inner_child)
        return tuple(children)

    def latency(self, job: message.DecodeJob) -> int:
        """The inner decoder's latency; the referee is never priced."""
        return self.inner.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The inner result, after the referee has checked it."""
        result = self.inner.decode(job)
        model = job.dem
        if model is None or result.logical_observables is None:
            return result
        syndrome = window_decode_results.payload_syndrome(job)
        outcome = self.referee.decode(model, syndrome)
        succeeded = window_decode_results.BackendDecodeStatus.SUCCEEDED
        if outcome.status is not succeeded:
            return result
        referee_flips = _owned_observable_flips(model, outcome)
        self.windows_checked += 1
        if referee_flips != tuple(result.logical_observables):
            self.window_disagreements += 1
        return result


def _owned_observable_flips(model, outcome) -> tuple:
    """The observables the referee's correction flips, owned faults only."""
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    referee_correction = numpy.asarray(outcome.physical_correction, dtype=bool)
    owned_correction = referee_correction & physical.owned
    observables = physical.observables.astype(numpy.int64)
    corrections = owned_correction.astype(numpy.int64)
    flip_counts = observables @ corrections
    flip_counts = numpy.asarray(flip_counts)
    flat_counts = flip_counts.ravel()
    flips = []
    for count in flat_counts:
        parity = int(count) % 2
        flips.append(parity)
    return tuple(flips)
