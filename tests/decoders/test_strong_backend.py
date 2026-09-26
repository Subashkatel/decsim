"""The strong decoder's queue in front of a device, against its closed form.

The closed form is the first-come first-served multi-server recursion
(Kiefer and Wolfowitz, "On the theory of queues with many servers",
1955): taken in arrival order, a decode starts at the later of its
arrival and the earliest time one of the c servers frees, and ends its
service time later. The device here is a stand-in with fixed service
times, so every start and end is known by hand.
"""

import dataclasses

import decsim.decoders.strong_backend as strong_backend
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records


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

    def capacity(self) -> int:
        return self.capacity_count

    def submit(self, request, running: int) -> _Ticket:
        entry = (request.label, running, self.engine.now)
        self.submits.append(entry)
        ticks = self.ticks_by_label[request.label]
        return _Ticket(request, ticks)

    def service_ticks(self, ticket: _Ticket) -> int:
        return ticket.service_ticks

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


def test_a_device_with_no_seeded_parts_names_no_seed_children():
    engine = engine_module.Engine()
    device = _FixedDevice(1, {}, engine)
    decoder = strong_backend.StrongBackendDecoder(device)
    assert decoder.run_seed_children() == ()
