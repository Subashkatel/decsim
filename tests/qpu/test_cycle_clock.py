"""The QPU cycle clock emits one round per cycle on every live patch.

Sources: Google 2207.06431 and 2408.13687 (every measure qubit is read out
each cycle; cadences of 921 ns and 1.1 us), Krinner 2112.03708 (1.1 us),
Yang 2605.04892 (1.25 us); QubiC
2404.15260 Sec. IV (a command starts on the boundary at or after its
arrival, the boundary itself included); validation matrix row C4. One
microsecond is 1_000_000 ticks.
"""

import random

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
        self.rounds = []
        self.readouts = []
        self.idle_ticks = []
        self.completion_ticks = []

    def accept_qpu_readout(self, payload, route):
        self.round_ticks.append((self.engine.now, payload.round_index))
        self.rounds.append(
            (
                self.engine.now,
                payload.operation_id,
                payload.patch_id,
                payload.round_index,
            )
        )
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
        1, 0, 1, fragment_count=3, fragment_index=1
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
        1, 0, 1, fragment_count=2, fragment_index=0
    )
    assert second == round_records.QPUReadout(
        1, 0, 1, fragment_count=2, fragment_index=1
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


# Round conservation (validation matrix row C4): on every boundary at
# which a patch is live, exactly one round is accounted for, as an
# operation round, an idle round, or a cycle of a body that reads out
# nothing. The failure it guards against is the engine jumping from one
# event tick to a far later one and dropping the boundaries between.


def silent_body(operation_id, round_count, cycle_ticks, patch=0):
    """A body that holds its patch for whole cycles without reading out."""
    operation = program_records.Operation(
        id=operation_id,
        name="wait",
        qubits=(patch,),
        patches=(patch,),
        emits_detector_data=False,
    )
    return program_records.RunOperationBody(
        operation,
        cycle_ticks,
        round_count,
        round_count,
        emits_detector_data=False,
    )


def issuing(qpu, body):
    """An action that issues the body at the tick it is scheduled for."""

    def issue():
        qpu.issue(body)

    return issue


def do_nothing():
    """An unrelated event: it must perturb no boundary."""


def random_schedule(rng):
    """A cycle and, per patch, its (id, start, rounds, emits) operations."""
    cycle = rng.choice([1, 2, 7, 10, 1000])
    patch_count = rng.randint(1, 4)
    schedule = {}
    operation_id = 0
    for patch in range(patch_count):
        operations = []
        start_cycles = rng.randint(0, 6)
        boundary = start_cycles * cycle
        operation_count = rng.randint(1, 4)
        for _ in range(operation_count):
            operation_id += 1
            round_count = rng.randint(1, 6)
            emits = rng.random() > 0.15
            operations.append((operation_id, boundary, round_count, emits))
            gap_cycles = rng.randint(0, 5)
            boundary += (round_count + gap_cycles) * cycle
        schedule[patch] = operations
    return cycle, schedule


def walked_operation(cycle, entry, next_start):
    """One operation's readout ticks and the idle ticks that follow it.

    A body issued in (S - cycle, S] starts at boundary S and reads out
    its round k at S + k*cycle, or occupies that cycle silently. From
    one cycle past its completion the patch emits an idle round on every
    boundary through the next start, that boundary included.
    """
    operation_id, start, round_count, emits = entry
    round_ticks = []
    stop = round_count + 1
    if emits:
        for round_index in range(1, stop):
            tick = start + round_index * cycle
            round_ticks.append((tick, operation_id, round_index))
    completion = start + round_count * cycle
    first_idle = completion + cycle
    last_idle = next_start + 1
    idle_ticks = list(range(first_idle, last_idle, cycle))
    return round_ticks, idle_ticks


def next_start_of(operations, index, end_boundary):
    """The boundary the next operation starts on, else the run's end."""
    next_index = index + 1
    if next_index < len(operations):
        return operations[next_index][1]
    return end_boundary


def walked_patch(cycle, operations, end_boundary):
    """One patch's readout and idle ticks, operation by operation."""
    round_ticks = []
    idle_ticks = []
    for index, entry in enumerate(operations):
        next_start = next_start_of(operations, index, end_boundary)
        entry_rounds, entry_idle = walked_operation(cycle, entry, next_start)
        round_ticks.extend(entry_rounds)
        idle_ticks.extend(entry_idle)
    return round_ticks, idle_ticks


def walked_schedule(cycle, schedule, end_boundary):
    """The reference walker: the ticks the documented rules give.

    Finish ends the run on the first boundary at or after its tick.
    """
    expected_rounds = {}
    expected_idle = {}
    for patch, operations in schedule.items():
        patch_rounds, patch_idle = walked_patch(cycle, operations, end_boundary)
        expected_rounds[patch] = patch_rounds
        expected_idle[patch] = patch_idle
    return expected_rounds, expected_idle


def issued_schedule(engine, qpu, cycle, schedule, rng):
    """Issue every body inside its own last cycle; the last completion."""
    last_completion = 0
    for patch, operations in schedule.items():
        for operation_id, start, round_count, emits in operations:
            operation = program_records.Operation(
                id=operation_id,
                name=f"op{operation_id}",
                qubits=(patch,),
                patches=(patch,),
                emits_detector_data=emits,
            )
            body = program_records.RunOperationBody(
                operation,
                cycle,
                round_count,
                round_count,
                emits_detector_data=emits,
            )
            own_cycle_start = start - cycle + 1
            earliest = max(0, own_cycle_start)
            issue_tick = rng.randint(earliest, start)
            action = issuing(qpu, body)
            engine.schedule(issue_tick, action)
            completion = start + round_count * cycle
            last_completion = max(last_completion, completion)
    return last_completion


def ended_run(engine, qpu, cycle, last_completion, rng):
    """Finish off boundary, with unrelated events before it; the tick."""
    extra_cycles = rng.randint(0, 3)
    offset = 0
    last_offset = cycle - 1
    if cycle > 1:
        offset = rng.randint(1, last_offset)
    finish_tick = last_completion + extra_cycles * cycle + offset
    engine.schedule(finish_tick, qpu.finish)
    noise_count = rng.randint(0, 5)
    before_finish = finish_tick - 1
    latest_noise = max(before_finish, 1)
    for _ in range(noise_count):
        noise_tick = rng.randint(0, latest_noise)
        engine.schedule(noise_tick, do_nothing, label="noise")
    return finish_tick


def run_schedule(cycle, schedule, rng):
    """Run one random schedule; its log and the boundary it ended on."""
    engine, qpu, log = clocked_qpu(cycle)
    last_completion = issued_schedule(engine, qpu, cycle, schedule, rng)
    finish_tick = ended_run(engine, qpu, cycle, last_completion, rng)
    engine.run()
    end_boundary = qpu.boundary_at_or_after(finish_tick)
    return log, end_boundary


def observed_schedule(log, schedule):
    """Per patch, the readouts the run made and the ticks it idled on."""
    observed_rounds = {}
    observed_idle = {}
    for patch in schedule:
        observed_rounds[patch] = []
        observed_idle[patch] = []
    for tick, operation_id, patch, round_index in log.rounds:
        observed_rounds[patch].append((tick, operation_id, round_index))
    for tick, patch, _round_index in log.idle_ticks:
        observed_idle[patch].append(tick)
    return observed_rounds, observed_idle


def covered_boundaries(cycle, operations, round_ticks, idle_ticks):
    """The boundaries one patch accounted for, and how many times."""
    emitted = []
    for tick, _operation_id, _round_index in round_ticks:
        emitted.append(tick)
    emitted.extend(idle_ticks)
    silent = silent_boundaries(cycle, operations)
    accounted = set(emitted) | silent
    accounted_count = len(emitted) + len(silent)
    return sorted(accounted), accounted_count


def strides_of(ticks):
    """The gaps between the boundaries, one gap value per pair."""
    strides = set()
    for earlier, later in zip(ticks, ticks[1:]):
        stride = later - earlier
        strides.add(stride)
    return strides


@pytest.mark.parametrize("seed", range(20))
def test_every_live_boundary_carries_exactly_one_round_property(seed):
    """Every patch's rounds and idle rounds are the walker's, tick for tick.

    The boundaries a patch accounts for run contiguously at the cycle
    stride from its first activity to the end of the run, with none
    counted twice and none missing.
    """
    rng = random.Random(seed)
    cycle, schedule = random_schedule(rng)
    log, end_boundary = run_schedule(cycle, schedule, rng)
    expected_rounds, expected_idle = walked_schedule(
        cycle, schedule, end_boundary
    )
    observed_rounds, observed_idle = observed_schedule(log, schedule)

    for patch, operations in schedule.items():
        assert observed_rounds[patch] == expected_rounds[patch]
        assert observed_idle[patch] == expected_idle[patch]
        covered, accounted_count = covered_boundaries(
            cycle, operations, observed_rounds[patch], observed_idle[patch]
        )
        strides = strides_of(covered)
        assert len(covered) == accounted_count
        assert strides <= {cycle}


def silent_boundaries(cycle, operations):
    """The boundaries a body that reads out nothing occupies."""
    ticks = set()
    for _operation_id, start, round_count, emits in operations:
        if emits:
            continue
        stop = round_count + 1
        for round_index in range(1, stop):
            tick = start + round_index * cycle
            ticks.add(tick)
    return ticks


def test_a_long_jump_between_events_skips_no_boundary():
    """Sparse events far apart: the engine jumps, the boundaries do not.

    One three-round body on a one-millisecond cycle, an unrelated
    mid-cycle event at 7777 and a finish at 10500 leave eight idle
    boundaries between the body and the end of the run.
    """
    engine, qpu, log = clocked_qpu(1000)
    body = memory_body(1, 3, 1000)
    qpu.issue(body)
    engine.schedule(7777, do_nothing, label="noise")
    engine.schedule(10500, qpu.finish)
    engine.run()

    idle_ticks = []
    for tick, _patch, _round_index in log.idle_ticks:
        idle_ticks.append(tick)
    assert log.round_ticks == [(1000, 1), (2000, 2), (3000, 3)]
    assert idle_ticks == [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]


def test_a_silent_body_takes_the_patch_from_the_idle_rounds_and_gives_it_back():
    """Idle extraction stops while a silent body holds the patch.

    The first body reads out rounds 1 and 2, the patch idles once, the
    silent body occupies three cycles, and the last body reads out on the
    boundary after it, with one idle round on each side of the pause.
    """
    engine, qpu, log = clocked_qpu(10)
    first = memory_body(1, 2, 10)
    silent = silent_body(2, 3, 10)
    last = memory_body(3, 1, 10)
    issue_silent = issuing(qpu, silent)
    issue_last = issuing(qpu, last)
    qpu.issue(first)
    engine.schedule(25, issue_silent)
    engine.schedule(65, issue_last)
    engine.schedule(85, qpu.finish)
    engine.run()

    read_out = []
    for tick, operation_id, _patch, _round_index in log.rounds:
        read_out.append((tick, operation_id))
    idle_ticks = []
    for tick, _patch, _round_index in log.idle_ticks:
        idle_ticks.append(tick)
    assert read_out == [(10, 1), (20, 1), (80, 3)]
    assert idle_ticks == [30, 70, 90]
    assert log.completion_ticks == [(20, 1), (60, 2), (80, 3)]


def test_a_two_patch_body_reads_out_on_one_patch_and_idles_on_both():
    """A merge is one round per cycle, named by the operation's first patch.

    The timing-only source merges the patches into one readout, and once
    the body completes each patch idles for itself, so the boundaries
    after it carry two idle rounds.
    """
    engine, qpu, log = clocked_qpu(10)
    merge = program_records.Operation(
        id=1, name="merge", qubits=(0, 1), patches=(0, 1)
    )
    body = program_records.RunOperationBody(merge, 10, 2, 2)
    qpu.issue(body)
    engine.schedule(45, qpu.finish)
    engine.run()

    read_out = []
    for tick, _operation_id, patch, _round_index in log.rounds:
        read_out.append((tick, patch))
    idle_pairs = []
    for tick, patch, _round_index in log.idle_ticks:
        idle_pairs.append((tick, patch))
    assert read_out == [(10, 0), (20, 0)]
    assert sorted(idle_pairs) == [
        (30, 0),
        (30, 1),
        (40, 0),
        (40, 1),
        (50, 0),
        (50, 1),
    ]
