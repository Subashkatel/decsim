"""The QPU cycle clock emits one round per cycle on every live patch.

Sources: Google 2207.06431 and 2408.13687 (every measure qubit is read out
each cycle; cadences of 921 ns and 1.1 us), Krinner 2112.03708 (1.1 us),
Yang 2605.04892 (1.25 us); SWIPER device_manager._generate_syndrome_round
(an active patch without an instruction emits an idle round); QubiC
2404.15260 Sec. IV (a command starts on the boundary at or after its
arrival, the boundary itself included); validation matrix row C4. One
microsecond is 1_000_000 ticks.
"""

import pytest

from decsim.engine import Engine
from decsim.message import Operation, RunOperationBody
from decsim.qpu.cycle_clock import QPUDevice
from decsim.qpu.syndrome_devices import TimingOnlyDevice


class ReadoutLog:
    def __init__(self, engine):
        self.engine = engine
        self.round_ticks = []
        self.idle_ticks = []
        self.completion_ticks = []

    def accept_qpu_readout(self, payload, route):
        del route
        self.round_ticks.append((self.engine.now, payload.round_index))

    def note_idle(self, operation_id, patch, round_index):
        del operation_id
        self.idle_ticks.append((self.engine.now, patch, round_index))

    def note_completion(self, operation):
        self.completion_ticks.append((self.engine.now, operation.id))


def clocked_qpu(cycle_ticks):
    engine = Engine(verbose=False)
    log = ReadoutLog(engine)
    source = TimingOnlyDevice()
    qpu = QPUDevice(
        engine,
        source,
        cycle_ticks,
        readout_receiver=log,
        completion_receiver=log.note_completion,
        idle_receiver=log.note_idle,
    )
    return engine, qpu, log


def memory_body(operation_id, round_count, cycle_ticks):
    operation = Operation(
        id=operation_id, name="memory", qubits=(0,), patches=(0,)
    )
    return RunOperationBody(operation, cycle_ticks, round_count, round_count)


def test_every_round_lands_on_a_boundary_of_the_1100_ns_cycle():
    engine, qpu, log = clocked_qpu(1_100_000)
    body = memory_body(1, 3, 1_100_000)
    qpu.issue(body)
    engine.schedule(3_300_000, qpu.finish)
    engine.run()
    assert log.round_ticks == [(1_100_000, 1), (2_200_000, 2), (3_300_000, 3)]
    assert log.completion_ticks == [(3_300_000, 1)]


def test_a_command_arriving_mid_cycle_starts_on_the_next_boundary():
    engine, qpu, log = clocked_qpu(921_000)
    body = memory_body(1, 1, 921_000)

    def issue_body():
        qpu.issue(body)

    engine.schedule(500_000, issue_body)
    engine.schedule(1_842_000, qpu.finish)
    engine.run()
    started = [event.tick for event in qpu.command_events]
    assert started == [500_000, 921_000]
    assert log.round_ticks == [(1_842_000, 1)]


def test_a_command_arriving_on_a_boundary_starts_on_that_boundary():
    engine, qpu, log = clocked_qpu(1_250_000)
    body = memory_body(1, 1, 1_250_000)

    def issue_body():
        qpu.issue(body)

    engine.schedule(1_250_000, issue_body)
    engine.schedule(2_500_000, qpu.finish)
    engine.run()
    kinds_and_ticks = [(event.kind, event.tick) for event in qpu.command_events]
    assert kinds_and_ticks == [("ARRIVED", 1_250_000), ("STARTED", 1_250_000)]
    assert log.round_ticks == [(2_500_000, 1)]


def test_the_boundary_at_or_after_a_readout_is_never_before_it():
    engine, qpu, log = clocked_qpu(921_000)
    assert qpu.boundary_at_or_after(0) == 0
    assert qpu.boundary_at_or_after(1) == 921_000
    assert qpu.boundary_at_or_after(921_000) == 921_000
    assert qpu.boundary_at_or_after(921_001) == 1_842_000
    engine, qpu, log = clocked_qpu(1_250_000)
    assert qpu.boundary_at_or_after(1_300_000) == 2_500_000
    with pytest.raises(ValueError, match="nonnegative"):
        qpu.boundary_at_or_after(-1)


def test_an_idle_patch_emits_one_round_per_cycle_until_finish():
    engine, qpu, log = clocked_qpu(10)
    body = memory_body(1, 2, 10)
    qpu.issue(body)
    engine.schedule(45, qpu.finish)
    engine.run()
    assert log.round_ticks == [(10, 1), (20, 2)]
    assert log.idle_ticks == [(30, 0, 1), (40, 0, 2), (50, 0, 3)]
    assert engine.now == 50


def test_a_body_without_detector_data_holds_its_patch_silently():
    engine, qpu, log = clocked_qpu(10)
    operation = Operation(id=1, name="wait", qubits=(0,), patches=(0,))
    body = RunOperationBody(operation, 10, 3, 3, emits_detector_data=False)
    qpu.issue(body)
    engine.schedule(30, qpu.finish)
    engine.run()
    assert log.round_ticks == []
    assert log.idle_ticks == []
    assert log.completion_ticks == [(30, 1)]


def test_a_command_with_another_cadence_is_refused():
    engine, qpu, log = clocked_qpu(10)
    body = memory_body(1, 2, 11)
    with pytest.raises(ValueError, match="cadence"):
        qpu.issue(body)


def test_an_instant_emitter_must_finalize_a_stream_round():
    engine, qpu, log = clocked_qpu(10)
    operation = Operation(id=1, name="tail", qubits=(0,), patches=(0,))
    body = RunOperationBody(operation, 10, 0, 3)
    with pytest.raises(ValueError, match="finalize"):
        qpu.issue(body)
