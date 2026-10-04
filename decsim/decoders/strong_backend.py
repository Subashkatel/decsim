"""The strong decoder on a device: FIFO queues in front of its resources.

StrongBackendDecoder answers the Decoder port for a row whose decode
runs on a StrongBackend (decsim/ports.py), once its input has landed. A
decode walks the steps the device states, each holding a resource, the
dispatcher or a worker; each resource serves up to its count at once and
the rest wait in arrival order, as an IonQ decoding core "processes
those blocks sequentially" (2608.25027 lines 509-511) and a CUDA-Q
dispatcher serves its ring's slots in turn (cuda-quantum
realtime/lib/daemon/dispatcher/dispatch_kernel.cu:213-265).

A region decoded with its X and Z parts apart is two requests to that
queue, each priced by its own size, and its answer is ready when both
are. Whether they overlap is the backend's capacity: one CUDA-Q
dispatcher's decode holds it to the end, so two run in series
(dispatch_kernel.cu v0.15.2 lines 575-589), the host path gives each
graph entry its own worker (cuda-quantum host_api.md lines 1104-1106),
and IBM runs an X and a Z decoder side by side (2510.21600 line 451).
"""

import collections
import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.basis_split as basis_split
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.seeding as seeding

# a strong backend row's and the relay_bp row's bases names one of these:
# whether a region's X and Z detectors are decoded together, which keeps
# the correlation a Y error makes between them, or apart, as two
# smaller problems that model X and Z errors as independent (Relay-BP
# 2506.01779 lines 289-296; IBM's FPGA runs X, Z or XYZ, 2510.21600
# line 291). The value says whether the region is split.
BASIS_DECODES = {"together": False, "apart": True}

# The resource every request enters by: a CUDA-Q dispatcher on the
# device path, the host monitor on the host path, both in slot order.
DISPATCHER = "dispatcher"
# One graph and its stream on the host path, idle again when the stream
# is done (cuda-quantum host_api.md lines 1065-1113).
WORKER = "worker"


class StrongBackendDecoder(decoder_module.DecoderBase):
    """The Decoder port served by a StrongBackend, FIFO past its capacity.

    A decode's time is known only once the device has it, so the unit
    declares no occupancy in advance. A cancelled decode still waiting
    for the dispatcher leaves the queue; a cancelled running one walks
    its steps to the end and its result is dropped, since a GPU does not
    stop a running kernel (CUDA C++ Programming Guide, preemption). The
    row is transparent to the seed walk: a backend that answers with
    decsim's own decoder draws what that decoder's row draws under the
    same run seed.
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
        capacities = backend.capacities()
        self.resources = _Resources(capacities)
        # id() of a split region -> its part jobs, while any may run
        self._parts_by_region: dict = {}

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
        parts = part_jobs(job)
        for basis, part in parts.items():
            ticket = self.backend.submit(part, 0)
            results[basis] = self.backend.result(ticket)
        return joined_result(job, results)

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
        engine: engine_module.Engine,
        on_result: decoder_module.OnResult,
    ) -> None:
        """Queue the decode, or its two parts; each runs once there is room."""
        if not self.splits_by_basis:
            self._arrive(job, on_result, engine)
            return
        parts = part_jobs(job)
        region_id = id(job)
        region_parts = parts.values()
        self._parts_by_region[region_id] = tuple(region_parts)
        forget = self._forget(job)
        join = _RegionJoin(job, parts, on_result, forget)
        for basis, part in parts.items():
            on_part = join.on_part(basis)
            self._arrive(part, on_part, engine)

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
        if self.resources.withdraw(job):
            return
        self.resources.silence(job)

    def _arrive(
        self,
        job: decoding_records.DecodeJob,
        on_result: decoder_module.OnResult,
        engine,
    ) -> None:
        """A decode queues for the dispatcher, in arrival order."""
        walk = _Walk(job, on_result)
        if self.resources.acquire(DISPATCHER, walk):
            self._enter(walk, engine)
            return
        walk.queued_ticks = engine.now

    def _enter(self, walk: "_Walk", engine) -> None:
        """Holding the dispatcher, the decode is handed to the device."""
        running = self.resources.running_count()
        walk.ticket = self.backend.submit(walk.job, running)
        self.resources.enter(walk)
        walk.steps = self.backend.steps(walk.ticket)
        walk.held = DISPATCHER
        self._next_step(walk, engine)

    def _next_step(self, walk: "_Walk", engine) -> None:
        """Run the walk's next step once its resource is held, or finish."""
        if walk.index == len(walk.steps):
            self._finish(walk, engine)
            return
        step = walk.steps[walk.index]
        needed = step.resource
        if needed is None or needed == walk.held:
            self._run_step(walk, engine)
            return
        if self.resources.acquire(needed, walk):
            self._run_step(walk, engine)
            return
        walk.queued_ticks = engine.now

    def _run_step(self, walk: "_Walk", engine) -> None:
        """The step's time; a resource it moves off is freed at its end."""
        step = walk.steps[walk.index]
        released = None
        if step.resource != walk.held:
            released = walk.held
        walk.held = step.resource
        engine.schedule(
            step.ticks,
            lambda: self._step_done(walk, released, engine),
            label=f"{step.name}_done({walk.job.label})",
        )

    def _step_done(
        self, walk: "_Walk", released: Optional[str], engine
    ) -> None:
        walk.index += 1
        if released is not None:
            self._release(released, engine)
        self._next_step(walk, engine)

    def _release(self, resource: str, engine) -> None:
        """Free one unit; the decode waiting longest for it goes on."""
        waiting = self.resources.release(resource)
        if waiting is None:
            return
        waited = engine.now - waiting.queued_ticks
        waiting.job.backend_queue_wait_ticks += waited
        if waiting.ticket is None:
            self._enter(waiting, engine)
            return
        self._run_step(waiting, engine)

    def _finish(self, walk: "_Walk", engine) -> None:
        """The device is done with the decode; its answer goes out."""
        self.resources.leave(walk)
        if walk.held is not None:
            self._release(walk.held, engine)
        if walk.cancelled:
            return
        result = self.backend.result(walk.ticket)
        walk.on_result(result)


def part_jobs(job: decoding_records.DecodeJob) -> dict:
    """The region's X part and Z part as jobs of their own.

    Each carries its part's model and its rows of the region's syndrome
    as one fragment. The relay_bp row splits a weak window the same way.
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


def joined_result(
    job: decoding_records.DecodeJob, results: dict
) -> decoding_records.DecodeResult:
    """One region's answer from its parts'.

    The observables and the detectors flipped are XORs of the parts',
    which never share a detector; the correction is the parts' columns
    in basis order; the region is best effort when either part is, and
    uncorrected when either part's backend produced no correction.
    """
    model = job.detector_error_model
    parts = [results[basis] for basis in sorted(results)]
    residuals = [part.boundary_data for part in parts]
    crossing = _joined_crossing(model, parts)
    corrections = [part.correction for part in parts]
    correction = numpy.concatenate(corrections)
    iterations = [part.iterations or 0 for part in parts]
    iteration_count = sum(iterations)
    observables = _joined_observables(parts)
    residual = _joined_residual(model, residuals)
    status = _joined_status(parts)
    no_correction_reason = _joined_no_correction_reason(parts)
    return decoding_records.DecodeResult(
        job.operation_id,
        job.window_id,
        correction=correction,
        logical_observables=observables,
        boundary_data=residual,
        crossing_commit=crossing,
        decode_status=status,
        iterations=iteration_count,
        no_correction_reason=no_correction_reason,
    )


@dataclasses.dataclass
class _Walk:
    """One decode on its way through the device's steps."""

    job: decoding_records.DecodeJob
    on_result: decoder_module.OnResult
    ticket: Any = None  # an opaque identity only the backend reads
    steps: tuple = ()
    # the step to run next, and the resource the decode holds now
    index: int = 0
    held: Optional[str] = None
    # the tick it joined the queue of the resource it waits for now
    queued_ticks: int = 0
    # cancelled on the device: it walks to the end and reports nothing
    cancelled: bool = False


class _Resources:
    """The device's resources, shared by the decodes on it.

    A freed unit passes straight to the decode waiting longest for it. A
    decode is on the device from its submit to its last step's end.
    """

    def __init__(self, capacities) -> None:
        self.free = dict(capacities)
        self.waiting = {name: collections.deque() for name in capacities}
        # id() of each job on the device -> its walk; the walk keeps the
        # job alive, so its id is not reused meanwhile
        self.on_device: dict = {}

    def running_count(self) -> int:
        """The decodes on the device now."""
        return len(self.on_device)

    def enter(self, walk: _Walk) -> None:
        """The decode is submitted to the device."""
        job_id = id(walk.job)
        self.on_device[job_id] = walk

    def leave(self, walk: _Walk) -> None:
        """The decode's last step has ended."""
        job_id = id(walk.job)
        del self.on_device[job_id]

    def silence(self, job: decoding_records.DecodeJob) -> None:
        """Mark a decode on the device to report nothing when it ends."""
        job_id = id(job)
        walk = self.on_device.get(job_id)
        if walk is not None:
            walk.cancelled = True

    def acquire(self, resource: str, walk: _Walk) -> bool:
        """Take a unit now, or queue for one; True when taken."""
        queue = self.waiting[resource]
        if self.free[resource] > 0 and not queue:
            self.free[resource] -= 1
            return True
        queue.append(walk)
        return False

    def release(self, resource: str) -> Optional[_Walk]:
        """Free a unit; the next decode waiting for it, which now holds it."""
        queue = self.waiting[resource]
        if queue:
            return queue.popleft()
        self.free[resource] += 1
        return None

    def withdraw(self, job: decoding_records.DecodeJob) -> bool:
        """Take a decode not yet on the device out of the dispatcher's queue."""
        queue = self.waiting[DISPATCHER]
        for walk in queue:
            if walk.job is job:
                queue.remove(walk)
                return True
        return False


class _RegionJoin:
    """The answers of one region's parts, joined once all have come.

    The region waited as long as the part that ended last, since both
    arrived with it: its time on the device is that part's waits and
    that part's steps.
    """

    def __init__(
        self,
        job: decoding_records.DecodeJob,
        parts: dict,
        on_result: decoder_module.OnResult,
        on_joined: Callable,
    ) -> None:
        self.job = job
        self.parts = parts
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
        if len(self.results) < len(self.parts):
            return
        last_part = self.parts[basis]
        self.job.backend_queue_wait_ticks = last_part.backend_queue_wait_ticks
        self.on_joined()
        joined = joined_result(self.job, self.results)
        self.on_result(joined)


def _part_job(
    job: decoding_records.DecodeJob, basis: str, part_model, part_syndrome
) -> decoding_records.DecodeJob:
    bits = decoder_module.int_tuple(part_syndrome)
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


def _joined_crossing(model, parts: list) -> window_records.CrossingCommit:
    """The parts' crossing commits, joined by the rule their results are."""
    crossings = [part.crossing_commit for part in parts]
    crossing_residuals = [crossing.residual for crossing in crossings]
    crossing_residual = _joined_residual(model, crossing_residuals)
    crossing_observables = _joined_observables(crossings)
    return window_records.CrossingCommit(
        crossing_residual, crossing_observables
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
    return decoder_module.int_tuple(flips)


def _joined_status(
    parts: list,
) -> Optional[decoding_records.BackendDecodeStatus]:
    """The first part's best-effort status; None when both succeeded."""
    for part in parts:
        if part.decode_status is not None:
            return part.decode_status
    return None


def _joined_no_correction_reason(
    parts: list,
) -> Optional[decoding_records.BackendFailureReason]:
    """The first part's reason for no correction; None when both had one."""
    for part in parts:
        if part.no_correction_reason is not None:
            return part.no_correction_reason
    return None
