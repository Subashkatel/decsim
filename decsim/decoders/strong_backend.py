"""The strong decoder on a device: a FIFO queue in front of its capacity.

StrongBackendDecoder answers the Decoder port for a row whose decode
runs on a StrongBackend (decsim/ports.py). The decoder manager's units
hand it decodes once their input has landed; up to the backend's
capacity run at once and the rest wait in arrival order. Arrival order
is what every referent here serves in: an IonQ decoding core
"processes those blocks sequentially" (2608.25027 lines 509-511), and a
CUDA-Q dispatcher serves its ring's slots in turn, `current_slot =
(current_slot + 1) % num_slots` (cuda-quantum
realtime/lib/daemon/dispatcher/dispatch_kernel.cu:213-265).

A region decoded with its X and Z parts apart is two requests to that
queue, each priced by its own size, and its answer is ready when both
are. Whether the two overlap is the backend's capacity, as it is on a
real dispatcher: one CUDA-Q dispatcher's decode holds it until the
decode ends, so two run in series (dispatch_kernel.cu v0.15.2 lines
575-589), while the host path gives each graph entry "its own worker,
enabling pipelined execution" (cuda-quantum host_api.md lines
1104-1106), and IBM runs an X decoder and a Z decoder side by side
(2510.21600 line 451).
"""

import collections
import dataclasses
from typing import Any, Callable, Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.seeding as seeding

# <tier>_decoder.bases names one of these rows on a strong backend row:
# whether a region's X and Z detectors are decoded together, which keeps
# the correlation a Y error makes between them, or apart, as two
# smaller problems that model X and Z errors as independent (Relay-BP
# 2506.01779 lines 289-296; IBM's FPGA runs X, Z or XYZ, 2510.21600
# line 291). The value says whether the region is split.
BASIS_DECODES = {"together": False, "apart": True}


class StrongBackendDecoder(decoder_module.DecoderBase):
    """The Decoder port served by a StrongBackend, FIFO past its capacity.

    A decode's time is known only once the device has it, so the unit
    declares no occupancy in advance, the measured decoder's shape
    (DecoderBase). A cancelled waiting decode leaves the queue. A
    cancelled running decode holds the device until its service time
    ends and its result is dropped: a GPU does not stop a running
    kernel, it schedules other work "as the currently running ...
    kernel's thread blocks finish" (CUDA C++ Programming Guide,
    preemption).

    The row is transparent to the seed walk: the backend's seeded parts
    sit at the paths they would hold without it, so a backend that
    answers with decsim's own decoder draws what that decoder's row
    draws under the same run seed.
    """

    def __init__(
        self, backend: ports.StrongBackend, bases: str = "together"
    ) -> None:
        self.backend = backend
        self.splits_by_basis = BASIS_DECODES[bases]
        if self.splits_by_basis:
            row = type(self)
            row_requirement = row.fault_model_requirement
            self.fault_model_requirement = row_requirement.joined(
                fault_models.DETECTOR_BASES_REQUIRED
            )
        # (job, on_result) in arrival order, waiting for the device
        self._waiting: collections.deque = collections.deque()
        # id() of a split region -> its part jobs, while any may run
        self._parts_by_region: dict = {}
        self._running_count = 0
        # id() of each job cancelled while it ran; the job stays alive in
        # its finish event until then, so its id is not reused meanwhile
        self._cancelled_job_ids: set = set()

    def run_seed_children(self) -> tuple:
        """The backend's own children, at the paths it names."""
        if not isinstance(self.backend, seeding.RunSeedComposite):
            return ()
        return self.backend.run_seed_children()

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The device's answer for one region, outside the queue."""
        if not self.splits_by_basis:
            ticket = self.backend.submit(job, 0)
            return self.backend.result(ticket)
        results = {}
        parts = _part_jobs(job)
        for basis, part in parts.items():
            ticket = self.backend.submit(part, 0)
            results[basis] = self.backend.result(ticket)
        return _joined_result(job, results)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """A device's time is its own; there is none before it runs."""
        raise NotImplementedError(
            "a strong decode's time is the device's answer; the queue holds "
            "the unit until the device says"
        )

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """None: the device prices the decode when it runs it."""
        del job
        return None

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine,
        on_result: decoder_module.OnResult,
    ) -> None:
        """Queue the decode, or its two parts; each runs once there is room."""
        if not self.splits_by_basis:
            self._waiting.append((job, on_result))
            self._run_waiting(engine)
            return
        parts = _part_jobs(job)
        region_id = id(job)
        part_jobs = parts.values()
        self._parts_by_region[region_id] = tuple(part_jobs)
        forget = self._forget(job)
        join = _RegionJoin(job, len(parts), on_result, forget)
        for basis, part in parts.items():
            on_part = join.on_part(basis)
            self._waiting.append((part, on_part))
        self._run_waiting(engine)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Drop a waiting decode; a running one finishes and says nothing."""
        region_id = id(job)
        parts = self._parts_by_region.pop(region_id, (job,))
        for part in parts:
            self._cancel_one(part)

    def _forget(self, job: decoding_records.DecodeJob) -> Callable:
        """What a split region's join calls once both answers are in."""
        region_id = id(job)
        return lambda: self._parts_by_region.pop(region_id, None)

    def _cancel_one(self, job: decoding_records.DecodeJob) -> None:
        for index, entry in enumerate(self._waiting):
            if entry[0] is job:
                del self._waiting[index]
                return
        job_id = id(job)
        self._cancelled_job_ids.add(job_id)

    def _run_waiting(self, engine) -> None:
        """Start waiting decodes in arrival order while the device has room."""
        capacity = self.backend.capacity()
        while self._waiting and self._running_count < capacity:
            job, on_result = self._waiting.popleft()
            self._run(job, on_result, engine)

    def _run(
        self,
        job: decoding_records.DecodeJob,
        on_result: decoder_module.OnResult,
        engine,
    ) -> None:
        ticket = self.backend.submit(job, self._running_count)
        self._running_count += 1
        service_ticks = self.backend.service_ticks(ticket)
        engine.schedule(
            service_ticks,
            lambda: self._finish(job, ticket, on_result, engine),
            label=f"decode_done({job.label})",
        )

    def _finish(
        self,
        job: decoding_records.DecodeJob,
        ticket: Any,
        on_result: decoder_module.OnResult,
        engine,
    ) -> None:
        """The device is free again; the decode's answer goes out."""
        self._running_count -= 1
        job_id = id(job)
        was_cancelled = job_id in self._cancelled_job_ids
        self._cancelled_job_ids.discard(job_id)
        self._run_waiting(engine)
        if was_cancelled:
            return
        result = self.backend.result(ticket)
        on_result(result)


class _RegionJoin:
    """The answers of one region's parts, joined once all have come."""

    def __init__(
        self,
        job: decoding_records.DecodeJob,
        part_count: int,
        on_result: decoder_module.OnResult,
        on_joined: Callable,
    ) -> None:
        self.job = job
        self.part_count = part_count
        self.on_result = on_result
        self.on_joined = on_joined
        self.results: dict = {}

    def on_part(self, basis: str) -> decoder_module.OnResult:
        """The callback one part's answer arrives on."""
        return lambda result: self._arrive(basis, result)

    def _arrive(
        self, basis: str, result: decoding_records.DecodeResult
    ) -> None:
        self.results[basis] = result
        if len(self.results) < self.part_count:
            return
        self.on_joined()
        joined = _joined_result(self.job, self.results)
        self.on_result(joined)


def _part_jobs(job: decoding_records.DecodeJob) -> dict:
    """The region's X part and Z part as jobs of their own.

    Each carries its part's model and its part's rows of the region's
    syndrome, as one fragment, which is all a backend reads of a job.
    """
    model = job.detector_error_model
    syndrome = decoder_module.payload_syndrome(job)
    parts = {}
    part_models = basis_split.split_by_basis(model)
    for basis, part_model in part_models.items():
        rows = basis_split.rows_of_basis(model, basis)
        part_syndrome = syndrome[rows]
        parts[basis] = _part_job(job, basis, part_model, part_syndrome)
    return parts


def _part_job(
    job: decoding_records.DecodeJob, basis: str, part_model, part_syndrome
) -> decoding_records.DecodeJob:
    bits = decoder_module.bit_tuple(part_syndrome)
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=job.operation_id,
        patch_ids=(),
        round_index=0,
        bits=bits,
        size_bits=len(bits),
        fragment_index=0,
    )
    label = f"{job.label} {basis}"
    return dataclasses.replace(
        job, detector_error_model=part_model, payloads=[fragment], label=label
    )


def _joined_result(
    job: decoding_records.DecodeJob, results: dict
) -> decoding_records.DecodeResult:
    """One region's answer from its parts'.

    The observables and the detectors flipped are XORs of the parts',
    which never share a detector; the correction is the parts' columns
    in basis order; the region is best effort when either part is.
    """
    model = job.detector_error_model
    parts = [results[basis] for basis in sorted(results)]
    residuals = [part.boundary_data for part in parts]
    crossings = [part.crossing_commit for part in parts]
    crossing_residuals = [crossing.residual for crossing in crossings]
    crossing_residual = _joined_residual(model, crossing_residuals)
    crossing_observables = _joined_observables(crossings)
    crossing = window_records.CrossingCommit(
        crossing_residual, crossing_observables
    )
    corrections = [part.correction for part in parts]
    correction = numpy.concatenate(corrections)
    iterations = [part.iterations or 0 for part in parts]
    iteration_count = sum(iterations)
    observables = _joined_observables(parts)
    residual = _joined_residual(model, residuals)
    status = _joined_status(parts)
    return decoding_records.DecodeResult(
        job.operation_id,
        job.window_id,
        correction=correction,
        logical_observables=observables,
        boundary_data=residual,
        crossing_commit=crossing,
        decode_status=status,
        iterations=iteration_count,
    )


def _joined_residual(model, residuals: list):
    """The parts' flipped detectors, which no two parts share."""
    detector_ids = []
    for residual in residuals:
        detector_ids.extend(residual.detector_ids)
    detector_ids.sort()
    return decoder_module.dependency_residual(model, tuple(detector_ids))


def _joined_observables(parts: list) -> tuple:
    """The XOR of the parts' observables; each part flips only its own."""
    flips = numpy.zeros(len(parts[0].logical_observables), dtype=numpy.uint8)
    for part in parts:
        flips ^= numpy.asarray(part.logical_observables, dtype=numpy.uint8)
    return decoder_module.bit_tuple(flips)


def _joined_status(parts: list) -> Optional[Any]:
    """The first part's best-effort status; None when both succeeded."""
    for part in parts:
        if part.decode_status is not None:
            return part.decode_status
    return None
