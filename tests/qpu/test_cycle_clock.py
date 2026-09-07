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

import decsim.engine
import decsim.observe.command_events as command_events_module
import decsim.observe.log_writers as log_writers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.program as program_records
import decsim.records.rounds as round_records


class ReadoutLog:
    """Keeps every readout, idle round and completion with its tick."""

    def __init__(self, engine):
        self.engine = engine
        self.round_ticks = []
        self.readouts = []
        self.idle_ticks = []
        self.completion_ticks = []

    def accept_qpu_readout(self, payload, route):
        self.round_ticks.append((self.engine.now, payload.round_index))
        self.readouts.append((payload, route))

    def note_idle(self, operation_id, patch, round_index):
        del operation_id
        self.idle_ticks.append((self.engine.now, patch, round_index))

    def note_completion(self, operation):
        self.completion_ticks.append((self.engine.now, operation.id))


class FinalizingSource(syndrome_devices.TimingOnlyDevice):
    """A timing-only source whose stream finalizer emits one payload."""

    def finalize_stream_round(self, operation, source_round_count):
        readout = round_records.QPUReadout(
            operation.stream_id, 0, source_round_count, bits=(1, 0, 1)
        )
        return [readout]


class RecordingSource(syndrome_devices.TimingOnlyDevice):
    """A timing-only source that keeps every begin_operation call."""

    def __init__(self):
        self.begun = []

    def begin_operation(
        self, operation, segment_round_count, source_round_count
    ):
        self.begun.append(
            (operation.id, segment_round_count, source_round_count)
        )


class SplitSource(syndrome_devices.TimingOnlyDevice):
    """A timing-only source that emits a fixed number of payloads."""

    def __init__(self, payload_count):
        self.payload_count = payload_count

    def round_payloads(self, operation, round_index):
        payload = round_records.QPUReadout(operation.id, 0, round_index)
        return [payload] * self.payload_count


def clocked_qpu(cycle_ticks, source=None):
    engine = decsim.engine.Engine()
    log = ReadoutLog(engine)
    if source is None:
        source = syndrome_devices.TimingOnlyDevice()
    qpu = cycle_clock.QPUDevice(
        engine,
        source,
        cycle_ticks,
        readout_receiver=log,
        completion_receiver=log.note_completion,
        idle_receiver=log.note_idle,
    )
    return engine, qpu, log


def memory_body(operation_id, round_count, cycle_ticks, patch=0, **changes):
    operation = program_records.Operation(
        id=operation_id,
        name="memory",
        qubits=(patch,),
        patches=(patch,),
        **changes,
    )
    return program_records.RunOperationBody(
        operation, cycle_ticks, round_count, round_count
    )


def test_every_round_lands_on_a_boundary_of_the_1100_ns_cycle():
    engine, qpu, log = clocked_qpu(1_100_000)
    body = memory_body(1, 3, 1_100_000)
    qpu.issue(body)
    engine.schedule(3_300_000, qpu.finish)
    engine.run()
    assert log.round_ticks == [(1_100_000, 1), (2_200_000, 2), (3_300_000, 3)]
    assert log.completion_ticks == [(3_300_000, 1)]


def test_a_round_is_logged_as_fired_with_its_place_in_the_body():
    engine, qpu, log = clocked_qpu(1_100_000)
    lines = log_writers.LogWriter()
    engine.line.connect(lines.write)
    body = memory_body(1, 3, 1_100_000)
    qpu.issue(body)
    engine.schedule(3_300_000, qpu.finish)
    engine.run()
    assert lines.lines == [
        "[  1.100 us] QPU: memory fires round 1/3",
        "[  2.200 us] QPU: memory fires round 2/3",
        "[  3.300 us] QPU: memory fires round 3/3",
    ]


def test_a_command_arriving_mid_cycle_starts_on_the_next_boundary():
    engine, qpu, log = clocked_qpu(921_000)
    commands = command_events_module.CommandEvents()
    qpu.command_event.connect(commands.command_event)
    body = memory_body(1, 1, 921_000)

    def issue_body():
        qpu.issue(body)

    engine.schedule(500_000, issue_body)
    engine.schedule(1_842_000, qpu.finish)
    engine.run()
    assert commands.events == [
        cycle_clock.QPUCommandEvent("ARRIVED", 500_000, body),
        cycle_clock.QPUCommandEvent("STARTED", 921_000, body),
    ]
    assert log.round_ticks == [(1_842_000, 1)]


def test_a_command_arriving_on_a_boundary_starts_on_that_boundary():
    engine, qpu, log = clocked_qpu(1_250_000)
    commands = command_events_module.CommandEvents()
    qpu.command_event.connect(commands.command_event)
    body = memory_body(1, 1, 1_250_000)

    def issue_body():
        qpu.issue(body)

    engine.schedule(1_250_000, issue_body)
    engine.schedule(2_500_000, qpu.finish)
    engine.run()
    assert commands.events == [
        cycle_clock.QPUCommandEvent("ARRIVED", 1_250_000, body),
        cycle_clock.QPUCommandEvent("STARTED", 1_250_000, body),
    ]
    assert log.round_ticks == [(2_500_000, 1)]


def test_the_boundary_at_or_after_a_tick_on_a_921_ns_clock():
    engine, qpu, log = clocked_qpu(921_000)
    assert qpu.boundary_at_or_after(0) == 0
    assert qpu.boundary_at_or_after(1) == 921_000
    assert qpu.boundary_at_or_after(921_000) == 921_000
    assert qpu.boundary_at_or_after(921_001) == 1_842_000


def test_the_boundary_at_or_after_a_tick_on_a_1250_ns_clock():
    engine, qpu, log = clocked_qpu(1_250_000)
    assert qpu.boundary_at_or_after(1_300_000) == 2_500_000


def test_a_boundary_query_before_time_zero_is_refused():
    engine, qpu, log = clocked_qpu(921_000)
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


def test_an_operation_starting_on_a_patch_ends_its_idle_rounds():
    engine, qpu, log = clocked_qpu(10)
    first = memory_body(1, 1, 10)
    second = memory_body(2, 2, 10)
    qpu.issue(first)

    def issue_second():
        qpu.issue(second)

    engine.schedule(25, issue_second)
    engine.schedule(45, qpu.finish)
    engine.run()
    assert log.round_ticks == [(10, 1), (40, 1), (50, 2)]
    assert log.idle_ticks == [(20, 0, 1), (30, 0, 2)]
    assert log.completion_ticks == [(10, 1), (50, 2)]


def test_two_operations_on_different_patches_run_in_the_same_cycles():
    engine, qpu, log = clocked_qpu(10)
    on_patch_a = memory_body(1, 2, 10, patch="A")
    on_patch_b = memory_body(2, 1, 10, patch="B")
    qpu.issue(on_patch_a)
    qpu.issue(on_patch_b)
    engine.schedule(15, qpu.finish)
    engine.run()
    assert log.round_ticks == [(10, 1), (10, 1), (20, 2)]
    assert log.idle_ticks == [(20, "B", 1)]
    assert log.completion_ticks == [(10, 2), (20, 1)]


def test_a_running_body_begins_on_the_syndrome_source_when_it_starts():
    source = RecordingSource()
    engine, qpu, log = clocked_qpu(10, source)
    body = memory_body(1, 2, 10)
    qpu.issue(body)
    engine.schedule(20, qpu.finish)
    engine.run()
    assert source.begun == [(1, 2, 2)]
    assert log.round_ticks == [(10, 1), (20, 2)]


def test_a_body_without_detector_data_holds_its_patch_silently():
    engine, qpu, log = clocked_qpu(10)
    operation = program_records.Operation(
        id=1, name="wait", qubits=(0,), patches=(0,)
    )
    body = program_records.RunOperationBody(
        operation, 10, 3, 3, emits_detector_data=False
    )
    qpu.issue(body)
    engine.schedule(30, qpu.finish)
    engine.run()
    assert log.round_ticks == []
    assert log.idle_ticks == []
    assert log.completion_ticks == [(30, 1)]


def test_a_zero_round_finalizer_delivers_the_final_readout_and_completes():
    source = FinalizingSource()
    engine, qpu, log = clocked_qpu(10, source)
    operation = program_records.Operation(
        id=2,
        name="tail",
        qubits=(0,),
        patches=(0,),
        stream_id="s",
        stream_offset=2,
        finalizes_stream_round=True,
    )
    body = program_records.RunOperationBody(
        operation, 10, 0, 3, finalizes_stream_round=True
    )
    qpu.issue(body)
    engine.schedule(5, qpu.finish)
    engine.run()
    payload, route = log.readouts[0]
    assert payload == round_records.QPUReadout("s", 0, 3, bits=(1, 0, 1))
    assert route == round_records.WINDOW_INPUT_ROUTE
    assert log.completion_ticks == [(0, 2)]
    assert log.round_ticks == [(0, 3)]


def test_a_declared_fragment_slot_is_stamped_on_the_one_payload():
    engine, qpu, log = clocked_qpu(10)
    body = memory_body(
        1, 1, 10, syndrome_fragment_index=1, syndrome_fragment_count=3
    )
    qpu.issue(body)
    engine.schedule(10, qpu.finish)
    engine.run()
    payload, route = log.readouts[0]
    assert payload == round_records.QPUReadout(
        1, 0, 1, n_fragments=3, fragment_index=1
    )
    assert route == round_records.WINDOW_INPUT_ROUTE


def test_undeclared_fragments_are_numbered_in_emission_order():
    source = SplitSource(2)
    engine, qpu, log = clocked_qpu(10, source)
    body = memory_body(1, 1, 10)
    qpu.issue(body)
    engine.schedule(10, qpu.finish)
    engine.run()
    first, _ = log.readouts[0]
    second, _ = log.readouts[1]
    assert first == round_records.QPUReadout(
        1, 0, 1, n_fragments=2, fragment_index=0
    )
    assert second == round_records.QPUReadout(
        1, 0, 1, n_fragments=2, fragment_index=1
    )


def test_a_declared_fragment_slot_refuses_two_payloads():
    source = SplitSource(2)
    engine, qpu, log = clocked_qpu(10, source)
    body = memory_body(
        1, 1, 10, syndrome_fragment_index=0, syndrome_fragment_count=2
    )
    qpu.issue(body)
    engine.schedule(10, qpu.finish)
    with pytest.raises(ValueError, match="must emit one payload"):
        engine.run()


def test_a_declared_fragment_count_must_match_the_emitted_payloads():
    source = SplitSource(2)
    engine, qpu, log = clocked_qpu(10, source)
    body = memory_body(1, 1, 10, syndrome_fragment_count=3)
    qpu.issue(body)
    engine.schedule(10, qpu.finish)
    with pytest.raises(ValueError, match="must match emitted readouts"):
        engine.run()


def test_a_detector_emitting_round_must_emit_a_readout():
    source = SplitSource(0)
    engine, qpu, log = clocked_qpu(10, source)
    body = memory_body(1, 1, 10)
    qpu.issue(body)
    engine.schedule(10, qpu.finish)
    with pytest.raises(ValueError, match="at least one readout"):
        engine.run()


def test_an_idle_stream_round_carries_the_sources_bits_to_the_windows():
    code = code_geometry.SurfaceCodeModel(distance=3)
    source = syndrome_devices.SyndromeBitDevice(code, seed=1)
    engine, qpu, log = clocked_qpu(10, source)
    operation = program_records.Operation(
        id=1, name="stream", qubits=(0,), patches=(0,), stream_id="s"
    )
    qpu.emit_idle_stream_round(operation, "s", 4, 0)
    payload, route = log.readouts[0]
    assert payload == round_records.QPUReadout(
        "s",
        0,
        4,
        bits=[0, 0, 1, 0, 1, 1, 1, 1],
        code="rotated surface code (d=3)",
        size_bits=8,
    )
    assert route == round_records.WINDOW_INPUT_ROUTE


def test_a_feedback_memory_round_is_routed_to_its_source_operation():
    engine, qpu, log = clocked_qpu(10)
    qpu.emit_feedback_memory_round(7, "A", 4)
    payload, route = log.readouts[0]
    assert payload == round_records.QPUReadout(("idle", 7, "A"), "A", 4)
    assert route == round_records.SyndromePacketRoute.feedback_memory_round(7)


def test_a_command_with_another_cadence_is_refused():
    engine, qpu, log = clocked_qpu(10)
    body = memory_body(1, 2, 11)
    with pytest.raises(ValueError, match="cadence"):
        qpu.issue(body)


def test_an_instant_emitter_must_finalize_a_stream_round():
    engine, qpu, log = clocked_qpu(10)
    operation = program_records.Operation(
        id=1, name="tail", qubits=(0,), patches=(0,)
    )
    body = program_records.RunOperationBody(operation, 10, 0, 3)
    with pytest.raises(ValueError, match="finalize"):
        qpu.issue(body)
