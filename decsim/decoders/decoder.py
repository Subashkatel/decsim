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
carries no latency model. The helpers after the classes are that frame:
the payload syndrome, the size check, the result from the selected
faults, and the status a best-effort result carries.
"""

import abc
import enum
import time
import weakref
from typing import Callable, Optional

import numpy

import decsim.config as config
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records
import decsim.records.windows as window_records

OnResult = Callable[[Optional[decoding_records.DecodeResult]], None]


class BackendDecodeStatus(enum.Enum):
    """Backend-neutral disposition of one window decode attempt."""

    SUCCEEDED = "succeeded"
    LOW_CONFIDENCE = "low_confidence"
    NONCONVERGED = "nonconverged"
    INVALID_CORRECTION = "invalid_correction"
    EMPTY_MODEL_UNSATISFIABLE = "empty_model_unsatisfiable"
    BACKEND_ERROR = "backend_error"


class DecoderBase(abc.ABC):
    """A row of the decoder table: decode and latency, timing from those.

    start prices the job with latency and decodes when that time ends;
    a decoder whose occupancy is None is measured on the host clock
    instead: decode_timed runs now and the result is delivered after
    the measured ticks. cancel does nothing, occupancy is latency, and
    the pipeline depth is one: the unit holds compute for the whole
    decode. stage_recorded is the port's stage source (data_path.md
    section 5's data-side callback): a row with internal stages replaces
    it with one of its own and fires a record per stage, and a row
    without leaves this silent one, so the machine connects the stage
    listeners to every row by name. window_checked is the same shape for
    a row that audits its own answer against a referee.
    """

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED
    stage_recorded = trace_source.SILENT
    window_checked = trace_source.SILENT
    # a row that can pin its solve to one logical class and report that
    # class's minimum weight says so here; the yaml refuses a confidence
    # built from forced solves over a row that cannot
    answers_forced_logical_class = False

    @abc.abstractmethod
    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The window's correction and its logical observables."""

    @abc.abstractmethod
    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The whole job's service time in ticks, known at dispatch."""

    def start(
        self, job: decoding_records.DecodeJob, engine, on_result: OnResult
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

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Stop a started job; the default decoder has nothing to stop."""
        del job

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """Ticks the unit's compute is held from the start; None if measured."""
        return self.latency(job)

    def pipeline_depth(self, job: decoding_records.DecodeJob) -> int:
        """Decodes that may be in flight on one unit; one is no pipeline."""
        del job
        return 1

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """(result, nanoseconds the decode took on the host clock)."""
        started_ns = time.perf_counter_ns()
        result = self.decode(job)
        finished_ns = time.perf_counter_ns()
        return result, finished_ns - started_ns

    def _deliver(
        self, job: decoding_records.DecodeJob, on_result: OnResult
    ) -> None:
        result = None
        if not job.cancelled and job.on_done is None:
            result = self.decode(job)
        on_result(result)

    def _start_measured(
        self, job: decoding_records.DecodeJob, engine, on_result: OnResult
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
        path = (seed_records.RunSeedPathSegment("field", "latency_model"),)
        child = seed_records.RunSeedChild(path, self.latency_model)
        return (child,)

    @abc.abstractmethod
    def compile(self, faults, model):
        """The backend for one window model, built once while it lives."""

    @abc.abstractmethod
    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """(selected faults, decode status or None) of one backend call."""

    def decode_forced_window(
        self, backend, model, faults, syndrome, forced_logical_class: int
    ) -> tuple:
        """(selected faults, decode status, the class's weight) of one solve.

        The minimum-weight correction inside one logical class, the
        weight a complementary gap subtracts (Gidney et al. 2312.04522
        Sec. "Complementary gap"). The weight is None when the window
        pins no observable and the class cannot be forced. A row whose
        answers_forced_logical_class is False never reaches this.
        """
        del backend, model, faults, syndrome, forced_logical_class
        row = type(self)
        row_name = row.__name__
        raise RuntimeError(
            f"{row_name} was asked for a forced-class solve and does not "
            "answer one; its row declares answers_forced_logical_class False"
        )

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The latency model's time; a measured decoder has none in advance."""
        if self.latency_model is None:
            raise NotImplementedError(
                "a decoder measured on the host clock has no latency before "
                "its call; the unit holds it for the measured time"
            )
        return self.latency_model.latency(job)

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The latency model's time, or None when measured."""
        if self.latency_model is None:
            return None
        return self.latency(job)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The correction of one window, or the empty result without a model."""
        result, _elapsed_ns = self.decode_timed(job)
        return result

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """(result, nanoseconds of the backend call alone)."""
        model = job.detector_error_model
        if model is None:
            return decoding_records.DecodeResult(
                job.operation_id, job.window_id
            ), 0
        faults = model.require_faults(self.fault_representation)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, faults)
        backend = self.compiled_for(faults, model)
        forced_class = job.forced_logical_class
        if forced_class is None:
            return self._plain_decode(job, backend, model, faults, syndrome)
        return self._forced_decode(
            job, backend, model, faults, syndrome, forced_class
        )

    def _plain_decode(self, job, backend, model, faults, syndrome) -> tuple:
        """(the window's own result, nanoseconds of the backend call)."""
        started_ns = time.perf_counter_ns()
        selected, decode_status = self.decode_window(
            backend, model, faults, syndrome
        )
        finished_ns = time.perf_counter_ns()
        result = result_from_selected_faults(
            job, model, faults, selected, decode_status=decode_status
        )
        return result, finished_ns - started_ns

    def _forced_decode(
        self, job, backend, model, faults, syndrome, forced_class: int
    ) -> tuple:
        """(the class's result and weight, nanoseconds of the backend call)."""
        started_ns = time.perf_counter_ns()
        selected, decode_status, weight = self.decode_forced_window(
            backend, model, faults, syndrome, forced_class
        )
        finished_ns = time.perf_counter_ns()
        result = result_from_selected_faults(
            job, model, faults, selected, decode_status=decode_status
        )
        result.forced_class_weight = weight
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


def parity_product(matrix, vector):
    """The matrix times the vector over GF(2), as a flat array."""
    matrix_integers = matrix.astype(numpy.int64)
    vector_integers = vector.astype(numpy.int64)
    product = matrix_integers @ vector_integers
    product = numpy.asarray(product)
    flat = product.ravel()
    return flat % 2


def bit_tuple(array) -> tuple:
    """The array's entries as a tuple of ints."""
    bits = []
    for bit in array:
        bits.append(int(bit))
    return tuple(bits)


def payload_syndrome(job: decoding_records.DecodeJob):
    """Concatenate payload bits into one syndrome vector."""
    if not job.payloads:
        return numpy.zeros(0, dtype=numpy.uint8)
    bit_arrays = []
    for payload in job.payloads:
        if payload.bits is None:
            continue
        bits = numpy.asarray(payload.bits, dtype=numpy.uint8)
        bit_arrays.append(bits)
    return numpy.concatenate(bit_arrays)


def check_syndrome_size(
    job: decoding_records.DecodeJob, syndrome, placed_faults
) -> None:
    """Fail when payload bits and detector rows do not line up."""
    detector_count = placed_faults.check.shape[0]
    if syndrome.size == detector_count:
        return
    raise ValueError(
        f"{job.label}: payload bits ({syndrome.size}) do not match the window "
        f"error model's detectors ({detector_count}). The device and the "
        "cluster's model build must use the same folded-round convention."
    )


def result_from_selected_faults(
    job: decoding_records.DecodeJob,
    model,
    placed_faults,
    selected,
    decode_status=None,
) -> decoding_records.DecodeResult:
    """Keep the owned selected faults and convert them into a DecodeResult.

    ``decode_status`` marks a best-effort correction (None = succeeded).
    """
    selected = numpy.asarray(selected, dtype=numpy.uint8)
    fault_count = placed_faults.check.shape[1]
    if selected.ndim != 1 or selected.shape[0] != fault_count:
        raise ValueError(
            f"{job.label}: selected correction has shape {selected.shape}; "
            f"expected ({fault_count},)"
        )
    selected_faults = selected.astype(bool)
    committed = selected_faults & placed_faults.owned
    observable_flips = parity_product(placed_faults.observables, committed)
    residual_detector_ids = _detector_ids_from_columns(
        placed_faults.boundary_flips, committed
    )
    defects = _defects_from_detector_ids(model, residual_detector_ids)
    correction = committed.astype(numpy.uint8)
    logical_observables = bit_tuple(observable_flips)
    boundary_data = window_records.DependencyResidual(
        detector_ids=residual_detector_ids, defects=defects
    )
    return decoding_records.DecodeResult(
        job.operation_id,
        job.window_id,
        correction=correction,
        logical_observables=logical_observables,
        boundary_data=boundary_data,
        decode_status=decode_status,
    )


def _detector_ids_from_columns(detector_flips, committed) -> tuple[int, ...]:
    """XOR complete global detector identities across selected columns."""
    detector_ids = set()
    nonzero = numpy.nonzero(committed)
    for column_index in nonzero[0]:
        flips = detector_flips.get(int(column_index), ())
        detector_ids.symmetric_difference_update(flips)
    return tuple(sorted(detector_ids))


def _defects_from_detector_ids(model, detector_ids) -> Optional[dict]:
    defects: dict = {}
    for detector_id in detector_ids:
        round_index, position = model.defect_positions[detector_id]
        mask = defects.setdefault(round_index, [])
        length = position + 1
        _extend_to(mask, length)
        mask[position] ^= 1
    if not defects:
        return None
    return defects


def _extend_to(mask: list, length: int) -> None:
    missing = length - len(mask)
    if missing > 0:
        padding = [0] * missing
        mask.extend(padding)
