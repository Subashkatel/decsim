"""The strong decoder's queue in front of a device, against its closed form.

The closed form is the first-come first-served multi-server recursion
(Kiefer and Wolfowitz, "On the theory of queues with many servers",
1955): taken in arrival order, a decode starts at the later of its
arrival and the earliest time one of the c servers frees, and ends its
service time later. The device here is a stand-in with fixed service
times, so every start and end is known by hand.
"""

import dataclasses
import functools

import numpy
import pytest

import decsim.decoders.decoder as decoder_module
import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.decoders.relay_belief_propagation.window_decoder as relay_window
import decsim.decoders.strong_backend as strong_backend
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
from tests.decoders import windows


@dataclasses.dataclass(frozen=True)
class _Ticket:
    job: decoding_records.DecodeJob
    service_ticks: int


class _FixedDevice:
    """A device of capacity c whose decode of a job takes a fixed time."""

    def __init__(self, capacity: int, ticks_by_label: dict, engine) -> None:
        self.capacity_count = capacity
        self.ticks_by_label = ticks_by_label
        self.engine = engine
        # (label, running, tick) of every submit, in order
        self.submits: list = []

    def capacities(self) -> dict:
        return {strong_backend.DISPATCHER: self.capacity_count}

    def submit(self, request, running: int) -> _Ticket:
        entry = (request.label, running, self.engine.now)
        self.submits.append(entry)
        ticks = self.ticks_by_label[request.label]
        return _Ticket(request, ticks)

    def steps(self, ticket: _Ticket) -> tuple:
        decode = decoding_records.Step(
            "decode", ticket.service_ticks, strong_backend.DISPATCHER
        )
        return (decode,)

    def result(self, ticket: _Ticket) -> decoding_records.DecodeResult:
        job = ticket.job
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


def _job(label: str, window_id: int) -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1, window_id=window_id, round_count=3, label=label
    )


def test_decodes_past_the_capacity_start_in_arrival_order_as_servers_free():
    """Capacity two, three decodes at tick 0 lasting 10, 20 and 5 ticks.

    The recursion: A and B start at 0; C waits for the earliest free
    server, A's at 10, and ends at 15. A decode submitted while one runs
    is told so.
    """
    engine = engine_module.Engine()
    device = _FixedDevice(2, {"A": 10, "B": 20, "C": 5}, engine)
    decoder = strong_backend.StrongBackendDecoder(device)
    finished = []

    def record_finish(result) -> None:
        del result
        finished.append(engine.now)

    first = _job("A", 0)
    second = _job("B", 1)
    third = _job("C", 2)
    decoder.start(first, engine, record_finish)
    decoder.start(second, engine, record_finish)
    decoder.start(third, engine, record_finish)
    engine.run()
    assert device.submits == [("A", 0, 0), ("B", 1, 0), ("C", 1, 10)]
    assert finished == [10, 15, 20]


def test_a_cancelled_waiting_decode_leaves_the_queue():
    engine = engine_module.Engine()
    device = _FixedDevice(1, {"A": 10, "B": 20, "C": 5}, engine)
    decoder = strong_backend.StrongBackendDecoder(device)
    finished = []

    def record_finish(result) -> None:
        del result
        finished.append(engine.now)

    first = _job("A", 0)
    cancelled = _job("B", 1)
    third = _job("C", 2)
    decoder.start(first, engine, record_finish)
    decoder.start(cancelled, engine, record_finish)
    decoder.start(third, engine, record_finish)
    decoder.cancel(cancelled)
    engine.run()
    assert device.submits == [("A", 0, 0), ("C", 0, 10)]
    assert finished == [10, 15]


def test_a_cancelled_running_decode_holds_the_device_and_reports_nothing():
    """A GPU does not stop a running kernel, so the next decode waits."""
    engine = engine_module.Engine()
    device = _FixedDevice(1, {"A": 10, "B": 5}, engine)
    decoder = strong_backend.StrongBackendDecoder(device)
    finished = []

    def record_finish(result) -> None:
        del result
        finished.append(engine.now)

    first = _job("A", 0)
    decoder.start(first, engine, record_finish)
    second = _job("B", 1)
    decoder.start(second, engine, record_finish)
    decoder.cancel(first)
    engine.run()
    assert device.submits == [("A", 0, 0), ("B", 0, 10)]
    assert finished == [15]


def test_a_cancel_after_the_decode_ended_leaves_its_next_run_alone():
    """A cancel that finds the job neither waiting nor running is a no-op.

    The same job object decoded again reports: a cancel that came after
    the first run's answer does not reach the second.
    """
    engine = engine_module.Engine()
    device = _FixedDevice(1, {"A": 10}, engine)
    decoder = strong_backend.StrongBackendDecoder(device)
    finished = []

    def record_finish(result) -> None:
        del result
        finished.append(engine.now)

    job = _job("A", 0)
    decoder.start(job, engine, record_finish)
    engine.run()
    decoder.cancel(job)
    decoder.start(job, engine, record_finish)
    engine.run()
    assert finished == [10, 20]


def test_a_device_with_no_seeded_parts_names_no_seed_children():
    engine = engine_module.Engine()
    device = _FixedDevice(1, {}, engine)
    decoder = strong_backend.StrongBackendDecoder(device)
    assert decoder.run_seed_children() == ()


# A region decoded with X and Z apart: two requests on the same queue.

REQUIREMENT = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED.joined(
    fault_models.DETECTOR_BASES_REQUIRED
)


def _region_job(label: str, window_id: int) -> decoding_records.DecodeJob:
    """A d = 3 memory's whole window over three rounds, one shot on it."""
    circuit = windows.memory_circuit(3, 3, 0.003)
    model = windows.whole_circuit_window(circuit, 3, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0], window_id)
    job.label = label
    return job


class _FixedAnsweringDevice(_FixedDevice):
    """The fixed-time device, answering each part with an empty correction."""

    def result(self, ticket: _Ticket) -> decoding_records.DecodeResult:
        job = ticket.job
        nothing = window_records.DependencyResidual()
        no_correction = numpy.zeros(0, dtype=numpy.uint8)
        no_crossing = window_records.CrossingCommit(nothing, (0,))
        return decoding_records.DecodeResult(
            job.operation_id,
            job.window_id,
            correction=no_correction,
            logical_observables=(0,),
            boundary_data=nothing,
            crossing_commit=no_crossing,
        )


def _served(capacity: int, regions: tuple) -> dict:
    """Each region's end under the Kiefer-Wolfowitz recursion.

    regions is (arrival, X part ticks, Z part ticks) in arrival order;
    the X part arrives first, and a region ends when its later part does.
    """
    free_at = [0] * capacity
    ends = {}
    for label, (arrival, *part_ticks) in enumerate(regions):
        part_ends = []
        for ticks in part_ticks:
            server = free_at.index(min(free_at))
            start = max(arrival, free_at[server])
            end = start + ticks
            free_at[server] = end
            part_ends.append(end)
        ends[str(label)] = max(part_ends)
    return ends


def _part_ticks_by_label(regions: tuple) -> dict:
    """Each region's X and Z part ticks, by the part's label."""
    ticks_by_label = {}
    for label, (_, x_ticks, z_ticks) in enumerate(regions):
        ticks_by_label[f"{label} X"] = x_ticks
        ticks_by_label[f"{label} Z"] = z_ticks
    return ticks_by_label


def _schedule_starts(engine, decoder, ended: dict, arrivals: tuple, jobs):
    """Job n starts at arrivals[n], and its end tick lands in ended[str(n)]."""
    for index, (arrival, job) in enumerate(zip(arrivals, jobs, strict=True)):
        record = _recorder(ended, str(index), engine)
        start = functools.partial(decoder.start, job, engine, record)
        engine.schedule(arrival, start)


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize(
    "regions",
    [
        ((0, 10, 20),),
        ((0, 10, 20), (0, 5, 5)),
        ((0, 7, 3), (5, 4, 9), (12, 6, 2)),
    ],
)
def test_a_split_regions_parts_queue_as_two_requests(capacity, regions):
    """Series on one server, side by side on two: the recursion decides."""
    engine = engine_module.Engine()
    ticks_by_label = _part_ticks_by_label(regions)
    device = _FixedAnsweringDevice(capacity, ticks_by_label, engine)
    decoder = strong_backend.StrongBackendDecoder(device, "apart")
    ended = {}
    arrivals = tuple(arrival for arrival, _, _ in regions)
    jobs = [_region_job(str(index), index) for index in range(len(regions))]
    _schedule_starts(engine, decoder, ended, arrivals, jobs)
    engine.run()
    assert ended == _served(capacity, regions)


def _recorder(ended: dict, label: str, engine):
    def record(result) -> None:
        del result
        ended[label] = engine.now

    return record


def _seeded_relay():
    """The relay_bp row with its gamma table seeded as a run seeds it."""
    decoder = relay.RelayBeliefPropagationDecoder()
    reservation = decoder.window_decoder.reserve_run_seed(5)
    decoder.window_decoder.commit_run_seed(reservation)
    return decoder


class _RelayDevice:
    """One server that answers with decsim's Relay-BP at once."""

    def __init__(self) -> None:
        self.decoder = _seeded_relay()

    def capacities(self) -> dict:
        return {strong_backend.DISPATCHER: 1}

    def submit(self, request, running: int):
        del running
        return self.decoder.decode(request)

    def steps(self, ticket) -> tuple:
        del ticket
        decode = decoding_records.Step("decode", 1, strong_backend.DISPATCHER)
        return (decode,)

    def result(self, ticket):
        return ticket


def _decoded_alone(reference, parts, shot) -> tuple:
    """The XOR of each part's observables, and every detector it flipped."""
    observables = numpy.zeros(1, dtype=numpy.uint8)
    flipped = []
    for part in parts:
        part_job = windows.job_for(part, shot)
        answer = reference.decode(part_job)
        observables ^= numpy.asarray(answer.logical_observables, dtype="uint8")
        flipped.extend(answer.boundary_data.detector_ids)
    return observables, flipped


def test_a_split_regions_answer_joins_its_two_parts_decoded_alone():
    """Observables XOR and flipped detectors union, as each part says."""
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(3, 3, 0.003)
    model = windows.whole_circuit_window(circuit, 3, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    device = _RelayDevice()
    decoder = strong_backend.StrongBackendDecoder(device, "apart")
    reference = _seeded_relay()
    part_by_basis = basis_split.split_by_basis(model)
    parts = part_by_basis.values()
    observables, flipped = _decoded_alone(reference, parts, detection_events[0])
    joined = decoder.decode(job)
    expected_observables = decoder_module.bit_tuple(observables)
    assert joined.logical_observables == expected_observables
    assert joined.boundary_data.detector_ids == tuple(sorted(flipped))


class _CrashingRelayDecoder:
    """A relay-bp decoder whose every decode raises, a crashed backend."""

    def __init__(self, *arguments, **keywords) -> None:
        del arguments, keywords

    def decode_detailed(self, syndrome):
        del syndrome
        raise RuntimeError("the relay-bp backend crashed")


def _crashing_relay_type():
    return _CrashingRelayDecoder


def test_a_split_region_with_no_correction_in_a_part_carries_its_reason(
    monkeypatch,
):
    """A part whose backend raised leaves the joined region uncorrected."""
    monkeypatch.setattr(
        relay_window, "_load_relay_decoder_type", _crashing_relay_type
    )
    job = _region_job("0", 0)
    device = _RelayDevice()
    decoder = strong_backend.StrongBackendDecoder(device, "apart")
    joined = decoder.decode(job)
    backend_error = decoding_records.BackendDecodeStatus.BACKEND_ERROR
    reasons = decoding_records.BackendFailureReason
    assert joined.decode_status is backend_error
    assert joined.no_correction_reason is reasons.UPSTREAM_EXCEPTION


def test_decoding_apart_asks_the_window_model_for_detector_types():
    engine = engine_module.Engine()
    device = _FixedDevice(1, {}, engine)
    decoder = strong_backend.StrongBackendDecoder(device, "apart")
    assert decoder.fault_model_requirement.detector_bases


def test_a_cancelled_split_region_withdraws_both_its_parts():
    engine = engine_module.Engine()
    ticks = {"A X": 10, "A Z": 20, "B X": 5, "B Z": 5}
    device = _FixedAnsweringDevice(1, ticks, engine)
    decoder = strong_backend.StrongBackendDecoder(device, "apart")
    ended = {}
    kept = _region_job("A", 0)
    cancelled = _region_job("B", 1)
    record_kept = _recorder(ended, "A", engine)
    record_cancelled = _recorder(ended, "B", engine)
    decoder.start(kept, engine, record_kept)
    decoder.start(cancelled, engine, record_cancelled)
    decoder.cancel(cancelled)
    engine.run()
    assert device.submits == [("A X", 0, 0), ("A Z", 0, 10)]
    assert ended == {"A": 30}


# A decode as steps on resources: the host path's monitor and workers.


class _HostPathDevice:
    """One monitor and some workers: launch on a worker, then its work."""

    def __init__(self, workers: int, launch: int, work: int) -> None:
        self.workers = workers
        self.launch = launch
        self.work = work

    def capacities(self) -> dict:
        return {
            strong_backend.DISPATCHER: 1,
            strong_backend.WORKER: self.workers,
        }

    def submit(self, request, running: int):
        del running
        return request

    def steps(self, ticket) -> tuple:
        del ticket
        worker = strong_backend.WORKER
        launch = decoding_records.Step("launch", self.launch, worker)
        work = decoding_records.Step("work", self.work, worker)
        return (launch, work)

    def result(self, ticket) -> decoding_records.DecodeResult:
        return decoding_records.DecodeResult(ticket.operation_id, 0)


def _host_monitor(arrivals: tuple, workers: int, launch: int, work: int):
    """The host monitor's loop by hand (cuda-quantum host_api.md 1065-1113).

    In slot order it waits for an idle worker, launches the graph on it
    and moves on; the worker is idle again when its stream is done.
    """
    monitor_free = 0
    worker_free = [0] * workers
    ends = []
    for arrival in arrivals:
        start = max(arrival, monitor_free)
        worker = worker_free.index(min(worker_free))
        start = max(start, worker_free[worker])
        monitor_free = start + launch
        end = monitor_free + work
        worker_free[worker] = end
        ends.append(end)
    return ends


@pytest.mark.parametrize("workers", [1, 2, 3])
@pytest.mark.parametrize("arrivals", [(0, 0, 0), (0, 2, 4, 6), (0, 20, 21, 50)])
def test_host_path_decodes_overlap_on_workers_as_the_monitor_launches(
    workers, arrivals
):
    engine = engine_module.Engine()
    device = _HostPathDevice(workers, 3, 10)
    decoder = strong_backend.StrongBackendDecoder(device)
    ended = {}
    jobs = [_job(str(index), index) for index in range(len(arrivals))]
    _schedule_starts(engine, decoder, ended, arrivals, jobs)
    engine.run()
    expected = _host_monitor(arrivals, workers, 3, 10)
    assert [ended[str(index)] for index in range(len(arrivals))] == expected
