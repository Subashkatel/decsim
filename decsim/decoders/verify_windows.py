"""The referee: every window re-decoded by Tesseract and compared.

After every tier decode, the official Tesseract backend re-decodes the
same window input and the owned observable contributions are compared;
each comparison fires window_checked, and the audit that counts them is
a listener (observe/referee_audit.py). Never priced: the inner decoder
starts every job on the engine, and the referee checks the result it
delivers. The linked fault models are built
whole-circuit, with memory linear in circuit length (verified through
d=9 x 1000 rounds). A run checks a tier by writing the referee's record
around that tier's decoder record.
"""

import dataclasses
from typing import Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.decoders.tesseract.decoder as tesseract_decoder
import decsim.decoders.tesseract.window_decoder as tesseract_window_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.seeds as seed_records
import decsim.trace_source as trace_source


class TesseractCheckedDecoder(decoder_module.DecoderBase):
    """A decoder whose every result Tesseract re-decodes to check.

    Timing is the inner decoder's in every respect; the referee's own
    call is never charged.

    Trace source: window_checked(window_key, is_agreement) once per
    window the referee reached a verdict on.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The referee written around the record of the decoder it checks.

        It reads as that record: the results name the inner decoder, and a
        confidence signal reads the inner decoder's weight step and
        timing, since the referee changes no result and costs no time.
        """

        # the checked decoder's own Settings record
        inner: ports.DecoderSettings

        @property
        def name(self) -> str:
            """The inner decoder's word, which the results name the tier by."""
            return self.inner.name

        @property
        def weight_step(self) -> Optional[float]:
            """The inner decoder's weight step, None when it declares none."""
            return getattr(self.inner, "weight_step", None)

        @property
        def timing(self) -> cycle_count_module.Timing:
            """The inner decoder's timing: its cycle count or the host's."""
            return self.inner.timing

        def build(self) -> "TesseractCheckedDecoder":
            """The inner decoder, built, with the referee around it."""
            inner = self.inner.build()
            return TesseractCheckedDecoder(inner)

    def __init__(self, inner: decoder_module.DecoderBase) -> None:
        self.inner = inner
        referee_settings = tesseract_decoder.TesseractDecoder.Settings()
        self.referee = tesseract_window_decoder.TesseractWindowDecoder(
            referee_settings
        )
        # The referee reads the physical view and the inner decoder its
        # own. Neither reads a link, and a linked build keeps faults of
        # other decompositions apart, which would hand a physical-view
        # inner decoder another problem than an unchecked run gives it.
        inner_requirement = inner.fault_model_requirement
        self.fault_model_requirement = inner_requirement.joined(
            fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
        )
        self.decoder_evidence = inner.decoder_evidence
        self.missing_evidence_reasons = inner.missing_evidence_reasons
        self.window_checked = trace_source.TraceSource()

    def run_seed_children(self) -> tuple:
        """The referee under its own path, the inner decoder's under theirs.

        The inner decoder's children keep the paths they have unchecked,
        so a checked run draws what the unchecked run draws.
        """
        referee_path = (seed_records.RunSeedPathSegment("field", "referee"),)
        children = [seed_records.RunSeedChild(referee_path, self.referee)]
        inner_children = getattr(self.inner, "run_seed_children", None)
        if inner_children is None:
            return tuple(children)
        inner_seeded = inner_children()
        children.extend(inner_seeded)
        return tuple(children)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The inner decoder's latency; the referee is never priced."""
        return self.inner.latency(job)

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The inner decoder's occupancy; None when it is measured."""
        return self.inner.occupancy(job)

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine: engine_module.Engine,
        on_result: decoder_module.OnResult,
    ) -> None:
        """Start the job on the inner decoder; its result is checked."""

        def on_inner_result(
            result: Optional[decoding_records.DecodeResult],
        ) -> None:
            if result is None:
                on_result(result)
                return
            checked = self._checked(job, result)
            on_result(checked)

        self.inner.start(job, engine, on_inner_result)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Stop the inner decoder's job."""
        self.inner.cancel(job)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The inner result, after the referee has checked it."""
        result = self.inner.decode(job)
        return self._checked(job, result)

    def _checked(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> decoding_records.DecodeResult:
        model = job.detector_error_model
        if model is None or result.logical_observables is None:
            return result
        syndrome = decoder_module.payload_syndrome(job)
        # a failed backend outcome skips the window, so a syndrome that
        # does not fit is refused here rather than dropped from the audit
        physical = model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        decoder_module.check_syndrome_size(job, syndrome, physical)
        outcome = self.referee.decode(model, syndrome)
        succeeded = decoding_records.BackendDecodeStatus.SUCCEEDED
        if outcome.status is not succeeded:
            return result
        referee_flips = _owned_observable_flips(model, outcome)
        window_key = (job.operation_id, job.window_id)
        is_agreement = referee_flips == tuple(result.logical_observables)
        self.window_checked.fire(window_key, is_agreement)
        return result


def _owned_observable_flips(model, outcome) -> tuple:
    """The observables the referee's correction flips, owned faults only."""
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    referee_correction = numpy.asarray(outcome.physical_correction, dtype=bool)
    owned_correction = referee_correction & physical.owned
    observables = physical.observables
    flips = decoder_module.parity_product(observables, owned_correction)
    return decoder_module.int_tuple(flips)
