"""The referee: every window re-decoded by Tesseract and compared.

After every tier decode, the official Tesseract backend re-decodes the
same window input and the owned observable contributions are compared;
each comparison fires window_checked, and the audit that counts them is
a listener (observe/referee_audit.py). Never priced: the engine reads
timing from the inner decoder alone. The linked fault models are built
whole-circuit, with memory linear in circuit length (verified through
d=9 x 1000 rounds).
"""

from typing import Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.decoders.tesseract.window_decoder as tesseract_window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message
import decsim.observe.trace_source as trace_source
import decsim.records.seeds as seed_records


class TesseractCheckedDecoder(decoder_module.DecoderBase):
    """A decoder whose every result Tesseract re-decodes and checks.

    Timing is the inner decoder's in every respect; the referee's own
    call is never charged.

    Trace source: window_checked(window_key, is_agreement) once per
    window the referee reached a verdict on.
    """

    def __init__(self, inner: decoder_module.DecoderBase):
        self.inner = inner
        self.referee = tesseract_window_decoder.TesseractWindowDecoder()
        # The referee reads the physical view, the tier the graphlike one.
        self.fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
        self.window_checked = trace_source.TraceSource()

    def run_seed_children(self) -> tuple:
        """The referee under its own path, the inner decoder's under inner."""
        referee_path = (seed_records.RunSeedPathSegment("field", "referee"),)
        children = [seed_records.RunSeedChild(referee_path, self.referee)]
        inner_children = getattr(self.inner, "run_seed_children", None)
        if inner_children is None:
            return tuple(children)
        inner_segment = (seed_records.RunSeedPathSegment("field", "inner"),)
        for child in inner_children():
            path = inner_segment + child.relative_path
            inner_child = seed_records.RunSeedChild(path, child.child)
            children.append(inner_child)
        return tuple(children)

    def latency(self, job: message.DecodeJob) -> int:
        """The inner decoder's latency; the referee is never priced."""
        return self.inner.latency(job)

    def occupancy(self, job: message.DecodeJob) -> Optional[int]:
        """The inner decoder's occupancy; None when it is measured."""
        return self.inner.occupancy(job)

    def pipeline_depth(self, job: message.DecodeJob) -> int:
        """The inner decoder's pipeline depth."""
        return self.inner.pipeline_depth(job)

    def cancel(self, job: message.DecodeJob) -> None:
        """Stop the inner decoder's job."""
        self.inner.cancel(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The inner result, after the referee has checked it."""
        result = self.inner.decode(job)
        return self._checked(job, result)

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """The inner decoder's measured call; the referee's is untimed."""
        result, elapsed_ns = self.inner.decode_timed(job)
        checked = self._checked(job, result)
        return checked, elapsed_ns

    def _checked(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> message.DecodeResult:
        model = job.dem
        if model is None or result.logical_observables is None:
            return result
        syndrome = decoder_module.payload_syndrome(job)
        outcome = self.referee.decode(model, syndrome)
        succeeded = decoder_module.BackendDecodeStatus.SUCCEEDED
        if outcome.status is not succeeded:
            return result
        referee_flips = _owned_observable_flips(model, outcome)
        window_key = (job.op_id, job.window_id)
        is_agreement = referee_flips == tuple(result.logical_observables)
        self.window_checked.fire(window_key, is_agreement)
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
