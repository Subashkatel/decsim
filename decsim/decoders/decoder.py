"""The Decoder port's defaults, and the template every window decoder shares.

sinter's abstract class with defaults
(sinter/_decoding/_decoding_decoder_class.py:58-102: `Decoder` raises
NotImplementedError where a row has nothing to say, and
`decode_via_files` is written once in terms of
`compile_decoder_for_dem`). DecoderBase gives a row the port's timing
methods from its latency; WindowDecoderBase gives a real decoder its
frame (the model's faults, the payload syndrome, the size check, the
result built from the selected faults), one compiled backend per live
window model (sinter's CompiledDecoder), and the measured clock when it
carries no latency model.
"""

import abc
import time
import weakref
from typing import Callable, Optional

import decsim.config as config
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message

OnResult = Callable[[Optional[message.DecodeResult]], None]


class DecoderBase(abc.ABC):
    """A row of the decoder table: decode and latency, timing from those.

    start prices the job with latency and decodes when that time ends;
    a decoder whose occupancy is None is measured on the host clock
    instead: decode_timed runs now and the result is delivered after
    the measured ticks. cancel does nothing, occupancy is latency, and
    the pipeline depth is one: the unit holds compute for the whole
    decode.
    """

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED

    @abc.abstractmethod
    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The window's correction and its logical observables."""

    @abc.abstractmethod
    def latency(self, job: message.DecodeJob) -> int:
        """The whole job's service time in ticks, known at dispatch."""

    def start(
        self, job: message.DecodeJob, engine, on_result: OnResult
    ) -> None:
        """Run the job on the engine; on_result runs once at its output.

        A cancelled job and a job without a window (on_done) deliver
        None: their unit time is charged, no correction is computed.
        """
        occupancy = self.occupancy(job)
        if occupancy is None:
            self._start_measured(job, engine, on_result)
            return
        latency_ticks = self.latency(job)
        engine.schedule(
            latency_ticks,
            lambda: self._deliver(job, on_result),
            label=f"decode_done({job.label})",
        )

    def cancel(self, job: message.DecodeJob) -> None:
        """Stop a started job; the default decoder has nothing to stop."""
        del job

    def occupancy(self, job: message.DecodeJob) -> Optional[int]:
        """Ticks the unit's compute is held from the start; None if measured."""
        return self.latency(job)

    def pipeline_depth(self, job: message.DecodeJob) -> int:
        """Decodes that may be in flight on one unit; one is no pipeline."""
        del job
        return 1

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """(result, nanoseconds the decode took on the host clock)."""
        started_ns = time.perf_counter_ns()
        result = self.decode(job)
        finished_ns = time.perf_counter_ns()
        return result, finished_ns - started_ns

    def _deliver(self, job: message.DecodeJob, on_result: OnResult) -> None:
        result = None
        if not job.cancelled and job.on_done is None:
            result = self.decode(job)
        on_result(result)

    def _start_measured(
        self, job: message.DecodeJob, engine, on_result: OnResult
    ) -> None:
        """Run the real call now; hold the unit for as long as it took."""
        result = None
        elapsed_ns = 0
        if not job.cancelled:
            result, elapsed_ns = self.decode_timed(job)
        elapsed_microseconds = elapsed_ns / 1000.0
        ticks = config.microseconds_to_ticks(elapsed_microseconds)
        engine.schedule(
            ticks, lambda: on_result(result), label=f"decode_done({job.label})"
        )


class WindowDecoderBase(DecoderBase):
    """A real decoder over one window's fault model.

    A row names its fault representation and implements compile(faults,
    model), the backend for one placed model, and decode_window(backend,
    model, faults, syndrome), one call on it returning (selected faults,
    decode status). With a latency model the manager prices the decode; with
    none (latency_model=None, the table's rows) the decode is measured
    on this host, timing the backend call only: the compile, the payload
    syndrome and the result construction are setup the hardware does not
    pay per window.
    """

    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE

    def __init__(self, latency_model: Optional[DecoderBase] = None):
        self.latency_model = latency_model
        self.compiled_by_model: dict = {}

    def run_seed_children(self) -> tuple:
        """The latency model that controls simulated service time."""
        path = (message.RunSeedPathSegment("field", "latency_model"),)
        child = message.RunSeedChild(path, self.latency_model)
        return (child,)

    @abc.abstractmethod
    def compile(self, faults, model):
        """The backend for one window model, built once while it lives."""

    @abc.abstractmethod
    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """(selected faults, decode status or None) of one backend call."""

    def latency(self, job: message.DecodeJob) -> int:
        """The latency model's time; a measured decoder has none in advance."""
        if self.latency_model is None:
            raise NotImplementedError(
                "a decoder measured on the host clock has no latency before "
                "its call; the unit holds it for the measured time"
            )
        return self.latency_model.latency(job)

    def occupancy(self, job: message.DecodeJob) -> Optional[int]:
        """The latency model's time, or None when measured."""
        if self.latency_model is None:
            return None
        return self.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The correction of one window, or the empty result without a model."""
        result, _elapsed_ns = self.decode_timed(job)
        return result

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """(result, nanoseconds of the backend call alone)."""
        model = job.dem
        if model is None:
            return message.DecodeResult(job.op_id, job.window_id), 0
        faults = model.require_faults(self.fault_representation)
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        backend = self.compiled_for(faults, model)
        started_ns = time.perf_counter_ns()
        selected, decode_status = self.decode_window(
            backend, model, faults, syndrome
        )
        finished_ns = time.perf_counter_ns()
        result = window_decode_results.result_from_selected_faults(
            job, model, faults, selected, decode_status=decode_status
        )
        return result, finished_ns - started_ns

    def compiled_for(self, faults, model):
        """The placed model's backend, compiled once and kept while it lives.

        The cache entry lives exactly as long as the placed model: id()
        values are recycled by CPython, and a dead entry would otherwise
        accumulate once per distinct window model of a long run.
        """
        model_identity = id(faults)
        entry = self.compiled_by_model.get(model_identity)
        if entry is not None:
            reference, backend = entry
            if reference() is faults:
                return backend
        backend = self.compile(faults, model)

        def discard_dead_model(reference) -> None:
            current = self.compiled_by_model.get(model_identity)
            if current is not None and current[0] is reference:
                del self.compiled_by_model[model_identity]

        reference = weakref.ref(faults, discard_dead_model)
        self.compiled_by_model[model_identity] = (reference, backend)
        return backend
