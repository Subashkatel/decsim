"""The flight recorder's ledger: one causal row per hardware transition.

`FlightRecorder.ledger` assembles the run's causal record out of the
owners' own records, named in the module's docstring: the packing
stage's round events, the room-side store's landings, the window
stamps, the frame's corrections and the runtime's release times. Its
`check()` is the accounting proof that makes the ledger evidence: every
emitted window-input round reaches exactly one terminal state, and no
event is stamped before its cause.

The ticks are the declared fabric of tests/declared_run.py, whose
arithmetic is exact. Round r leaves the qpu at r us, its bits are
classified 5 us later (2 us on the qpu path, 3 us of readout
classification), packing is declared free, and from there the
weak-buffer path publishes it 4 us on while the strong-buffer path
lands it in the room-side store 7 us on.

The window trace walk at the end of the file reads one shot of
configs/weak_decoder_baseline.yaml as well, because the four events a
latency claim rests on are read off a shipped config's own run. IBM
arXiv 2510.21600 lines 517-519 name them on the hardware: the decoder
FPGA's trace observes when the decoders start and stop, when the
syndromes and codewords arrive, and when the logical Pauli frame is
produced. gem5 src/sim/probe/probe.hh lines 122 and 272 give the shape
the ledger takes here, a listener attached to a named probe point.
"""

import random
import types

import pytest

import decsim.collect as collect
import decsim.engine as engine_module
import decsim.experiments.experiment as experiment
import decsim.observe.flight_recorder as flight_recorder_module
import decsim.observe.round_events as round_events_module
import decsim.records.rounds as round_records
import tests.declared_run as declared_run
import tests.experiments.yaml_configs as yaml_configs
from decsim.config import microseconds_to_ticks


def kinds_and_ticks(chain):
    """Every event of a chain as (kind, tick), in chain order."""
    rows = []
    for event in chain:
        rows.append((event.kind, event.tick))
    return rows


def chained_ids(chain):
    """The prev_event_id of every event of a chain but the first."""
    causes = []
    for event in chain[1:]:
        causes.append(event.prev_event_id)
    return causes


def leading_ids(chain):
    """The event_id of every event of a chain but the last."""
    ids = []
    for event in chain[:-1]:
        ids.append(event.event_id)
    return ids


def events_by_id(ledger):
    """Every event of the ledger, looked up by its event_id."""
    by_id = {}
    for event in ledger.events:
        by_id[event.event_id] = event
    return by_id


def events_of_kind(ledger, kind):
    """Every event of the ledger of one kind, in ledger order."""
    rows = []
    for event in ledger.events:
        if event.kind == kind:
            rows.append(event)
    return rows


def one_event(ledger, kind, operation_id):
    """The one event of that kind for that operation; a second fails."""
    rows = []
    for event in ledger.events:
        if event.kind != kind:
            continue
        if event.op != operation_id:
            continue
        rows.append(event)
    (found,) = rows
    return found


def terminal_rounds_and_kinds(ledger):
    """(round, kind) of every event the ledger marked terminal."""
    terminals = set()
    for event in ledger.events:
        if event.status != "terminal":
            continue
        terminals.add((event.round, event.kind))
    return terminals


def terminal_kinds_of_operation(ledger, operation_id):
    """The kind of every terminal event of one operation's stream."""
    kinds = []
    for event in ledger.events:
        if event.status != "terminal":
            continue
        if event.op != operation_id:
            continue
        kinds.append(event.kind)
    return kinds


def record_round_event(recorder, kind, operation_id, round_index, patch=None):
    """One transition of one round on the window-input route, at tick 0."""
    route = round_records.WINDOW_INPUT_ROUTE
    event = round_records.RoundEvent.of(
        kind, 0, operation_id, round_index, route, patch
    )
    recorder.record(event)


def ledger_of_rounds_alone(recorder):
    """The ledger of a run whose only listener is the round recorder."""
    windows = types.SimpleNamespace(windows={})
    stamps = types.SimpleNamespace(decode_release={}, result_return={})
    commands = types.SimpleNamespace(events=())
    corrections = flight_recorder_module.FrameCorrections()
    rounds_only = flight_recorder_module.FlightRecorder(
        recorder, windows, stamps, commands, corrections, ()
    )
    return rounds_only.ledger


def test_a_rounds_chain_is_exact_and_each_event_names_its_cause():
    """One round's controller chain, hop by hop on the declared fabric.

    Round 3 is emitted at 3 us, its bits are available at 8, packing is
    free so it is packed and sent at 8, and the weak-buffer path
    publishes it in Buffer 0 at 12. Each row's cause is the row before
    it, which is what lets a reader follow one readout through the
    machine (flight_recorder.py's _round_chains).
    """
    machine = declared_run.weak_only_run(rounds=6)
    ledger = machine.observation.flight_recorder.ledger

    chain = ledger.chain(op=1, round=3)

    rows = kinds_and_ticks(chain)
    assert rows == [
        ("EMITTED", microseconds_to_ticks(3.0)),
        ("BINARY_AVAILABLE", microseconds_to_ticks(8.0)),
        ("PACKED", microseconds_to_ticks(8.0)),
        ("CWB_SENT", microseconds_to_ticks(8.0)),
        ("PUBLISHED", microseconds_to_ticks(12.0)),
    ]
    causes = chained_ids(chain)
    leading = leading_ids(chain)
    assert causes == leading
    last = chain[-1]
    assert last.status == "terminal"


def test_a_windows_chain_is_exact_and_its_cause_is_the_last_round_it_read():
    """One window's chain, and the round whose arrival completed it.

    A six-round window is complete when round 6 is published at 15 us;
    it is queued and assigned in the same instant, the weak decoder
    takes 10 us behind a 5 us input path, and the frame accepts at 32
    and commits its 1 us write at 33. The chain's cause is the last
    round it read, because that publication is the event the window
    manager waited on (flight_recorder.py's _window_chains).
    """
    machine = declared_run.weak_only_run(rounds=6)
    ledger = machine.observation.flight_recorder.ledger

    chain = ledger.chain(op=1, window=0)

    rows = kinds_and_ticks(chain)
    assert rows == [
        ("WINDOW_DATA_COMPLETE", microseconds_to_ticks(15.0)),
        ("DECODE_QUEUED", microseconds_to_ticks(15.0)),
        ("UNIT_ASSIGNED", microseconds_to_ticks(15.0)),
        ("DECODE_STARTED", microseconds_to_ticks(20.0)),
        ("DECODE_DONE", microseconds_to_ticks(30.0)),
        ("FRAME_ACCEPTED", microseconds_to_ticks(32.0)),
        ("FRAME_COMMITTED", microseconds_to_ticks(33.0)),
    ]
    by_id = events_by_id(ledger)
    first = chain[0]
    cause = by_id[first.prev_event_id]
    assert cause.kind == "PUBLISHED"
    assert cause.round == 6
    assert cause.tick == microseconds_to_ticks(15.0)


def test_every_emitted_round_reaches_exactly_one_terminal_state():
    """Conservation over a whole run, which is the ledger's own claim.

    A weak-only run of six rounds publishes all six, so the check finds
    one terminal state per emitted round and nothing lost on the way
    (flight_recorder.py's RunLedgerView docstring).
    """
    machine = declared_run.weak_only_run(rounds=6)
    ledger = machine.observation.flight_recorder.ledger

    ledger.check()

    emitted = events_of_kind(ledger, "EMITTED")
    published = events_of_kind(ledger, "PUBLISHED")
    assert len(emitted) == 6
    assert len(published) == 6


def test_a_strong_primary_run_records_the_room_side_landing():
    """A strong-primary round travels once, and its journey ends there.

    Readiness listens to the room-side store, so nothing crosses the
    weak-buffer path: the round is packed at 8 us and lands at 15, and
    that landing is its terminal state rather than a publication
    (flight_recorder.py's _store_landings). The window then waits the
    6 us strong input path and the 30 us strong decoder, and the frame
    records the tier that served it.
    """
    machine = declared_run.strong_only_run(rounds=6)
    ledger = machine.observation.flight_recorder.ledger

    ledger.check()

    round_rows = ledger.chain(op=1, round=3)
    assert kinds_and_ticks(round_rows) == [
        ("EMITTED", microseconds_to_ticks(3.0)),
        ("BINARY_AVAILABLE", microseconds_to_ticks(8.0)),
        ("PACKED", microseconds_to_ticks(8.0)),
        ("STORED_SB1", microseconds_to_ticks(15.0)),
    ]
    window_rows = ledger.chain(op=1, window=0)
    assert kinds_and_ticks(window_rows) == [
        ("WINDOW_DATA_COMPLETE", microseconds_to_ticks(18.0)),
        ("DECODE_QUEUED", microseconds_to_ticks(18.0)),
        ("UNIT_ASSIGNED", microseconds_to_ticks(18.0)),
        ("DECODE_STARTED", microseconds_to_ticks(24.0)),
        ("DECODE_DONE", microseconds_to_ticks(54.0)),
        ("FRAME_ACCEPTED", microseconds_to_ticks(58.0)),
        ("FRAME_COMMITTED", microseconds_to_ticks(59.0)),
    ]
    last = window_rows[-1]
    assert last.route == "strong"


def test_a_full_escalation_runs_ledger_passes_its_check():
    """Every window escalating exercises both tiers of one chain.

    An escalated window is decoded twice and committed once, so the
    accounting has to hold across the second decode as well as the
    first.
    """
    machine = declared_run.switching_run(rounds=6, escalation_probability=1.0)
    ledger = machine.observation.flight_recorder.ledger

    ledger.check()


def test_a_release_is_caused_by_the_blocking_operations_commit():
    """The feedback path of a blocked operation, from commit to pulse.

    Operation 2 waits on operation 1, whose last correction commits at
    33 us; the frame's decision reaches the controller 2 us later, so
    the decision is available and the release and the pulse command are
    issued at 35, and the command link puts the command at the qpu at
    37. The cause of the decision is the blocking operation's commit,
    not the blocked operation's own work
    (flight_recorder.py's _output_row and _releases).
    """
    first = declared_run.memory_operation(1)
    second = declared_run.memory_operation(2, blocked_by=1)
    operations = [first, second]
    machine = declared_run.weak_only_run(rounds=6, operations=operations)
    ledger = machine.observation.flight_recorder.ledger

    ledger.check()

    decision = one_event(ledger, "DECISION_AVAILABLE", 2)
    released = one_event(ledger, "DECODE_RELEASED", 2)
    issued = one_event(ledger, "CONTROL_PULSE_COMMAND_ISSUED", 2)
    arrived = one_event(ledger, "QPU_COMMAND_ARRIVED", 2)
    started = one_event(ledger, "QPU_COMMAND_STARTED", 2)
    by_id = events_by_id(ledger)
    cause = by_id[decision.prev_event_id]
    assert cause.kind == "FRAME_COMMITTED"
    assert cause.op == 1
    assert cause.tick == microseconds_to_ticks(33.0)
    assert decision.tick == microseconds_to_ticks(35.0)
    assert released.tick == microseconds_to_ticks(35.0)
    assert issued.tick == microseconds_to_ticks(35.0)
    assert arrived.tick == microseconds_to_ticks(37.0)
    assert started.tick == microseconds_to_ticks(37.0)
    chain = [decision, released, issued, arrived, started]
    causes = chained_ids(chain)
    leading = leading_ids(chain)
    assert causes == leading


def test_the_check_refuses_an_effect_stamped_before_its_cause():
    """The causal half of the check has teeth.

    A ledger whose rows are ordered by tick can still name a cause that
    is later than its effect, and that is a modeling bug rather than a
    slow run, so the check raises instead of reporting a negative
    latency (flight_recorder.py's _causes_before_effects).
    """
    cause_tick = microseconds_to_ticks(5.0)
    effect_tick = microseconds_to_ticks(4.0)
    cause = flight_recorder_module.LedgerEvent(
        event_id=0, kind="EMITTED", tick=cause_tick, op=1, round=1
    )
    effect = flight_recorder_module.LedgerEvent(
        event_id=1,
        kind="PUBLISHED",
        tick=effect_tick,
        op=1,
        round=1,
        prev_event_id=0,
        status="terminal",
    )
    view = flight_recorder_module.RunLedgerView(events=(cause, effect))

    with pytest.raises(RuntimeError, match="precedes its cause"):
        view.check()


def test_the_check_refuses_a_round_that_disappeared():
    """The conservation half of the check has teeth.

    A round emitted and never terminal is a packet that vanished. The
    ledger is used as evidence, so it must fail rather than read as a
    complete run (flight_recorder.py's _one_terminal_per_round).
    """
    orphan_tick = microseconds_to_ticks(1.0)
    orphan = flight_recorder_module.LedgerEvent(
        event_id=0, kind="EMITTED", tick=orphan_tick, op=1, round=1
    )
    view = flight_recorder_module.RunLedgerView(events=(orphan,))

    with pytest.raises(RuntimeError, match="terminal states"):
        view.check()


def test_a_round_dropped_for_want_of_store_room_ends_in_dropped():
    """A loss at Buffer 0 admission is accounted, not silent.

    DROPPED is one of the round terminals, so the round that the writer
    refused closes its chain there and the conservation check still
    passes: the ledger says the round was lost rather than saying
    nothing (flight_recorder.py's _ROUND_TERMINALS).
    """
    engine = engine_module.Engine()
    recorder = round_events_module.RoundEventRecorder(engine)
    record_round_event(recorder, "EMITTED", 1, 1, patch=0)
    record_round_event(recorder, "PACKED", 1, 1)
    record_round_event(recorder, "PUBLISHED", 1, 1)
    record_round_event(recorder, "EMITTED", 1, 2, patch=0)
    record_round_event(recorder, "PACKED", 1, 2)
    record_round_event(recorder, "DROPPED", 1, 2)
    ledger = ledger_of_rounds_alone(recorder)

    ledger.check()

    terminals = terminal_rounds_and_kinds(ledger)
    assert terminals == {(1, "PUBLISHED"), (2, "DROPPED")}
    assert recorder.packing_drops == 1


def test_a_round_refused_a_reassembly_context_ends_in_dropped():
    """The same accounting holds for a loss before packing.

    The assembler's workspace is bounded too, so a round refused a
    context there is dropped with no PACKED row of its own; the chain
    closes on DROPPED and the check still passes
    (flight_recorder.py's _ROUND_TERMINALS).
    """
    engine = engine_module.Engine()
    recorder = round_events_module.RoundEventRecorder(engine)
    record_round_event(recorder, "EMITTED", 1, 1, patch=0)
    record_round_event(recorder, "EMITTED", 1, 2, patch=0)
    record_round_event(recorder, "DROPPED", 1, 2, patch=0)
    record_round_event(recorder, "PACKED", 1, 1)
    record_round_event(recorder, "PUBLISHED", 1, 1)
    ledger = ledger_of_rounds_alone(recorder)

    ledger.check()

    terminals = terminal_rounds_and_kinds(ledger)
    assert terminals == {(1, "PUBLISHED"), (2, "DROPPED")}
    assert recorder.packing_drops == 1


def weak_mode(generator, rounds):
    """Weak only, over one to three independent memory patches."""
    operation_count = generator.randint(1, 3)
    last_operation = operation_count + 1
    operations = []
    for operation_id in range(1, last_operation):
        operation = declared_run.memory_operation(operation_id)
        operations.append(operation)
    return declared_run.weak_only_run(rounds=rounds, operations=operations)


def weak_blocked_mode(_generator, rounds):
    """Weak only, with a successor that waits on the first operation."""
    first = declared_run.memory_operation(1)
    second = declared_run.memory_operation(2, blocked_by=1)
    operations = [first, second]
    return declared_run.weak_only_run(rounds=rounds, operations=operations)


def strong_mode(_generator, rounds):
    """Strong primary: readiness listens to the room-side store."""
    return declared_run.strong_only_run(rounds=rounds)


def switching_keep_mode(_generator, rounds):
    """Weak primary, every window keeping its weak result."""
    return declared_run.switching_run(rounds=rounds, escalation_probability=0.0)


def switching_escalate_mode(_generator, rounds):
    """Weak primary, every window escalating to the strong tier."""
    return declared_run.switching_run(rounds=rounds, escalation_probability=1.0)


def switching_parallel_mode(_generator, rounds):
    """Both tiers started at once, on a faster strong-buffer path."""
    return declared_run.switching_run(
        rounds=rounds,
        escalation_probability=1.0,
        run_both_at_once=True,
        strong_buffer_microseconds=2.0,
    )


def switching_forward_mode(_generator, rounds):
    """Escalation under the forward window, which holds no boundary."""
    return declared_run.switching_run(
        rounds=rounds, escalation_probability=1.0, strong_window="forward"
    )


SWEEP_MODES = (
    ("weak", weak_mode),
    ("weak_blocked", weak_blocked_mode),
    ("strong", strong_mode),
    ("switching_keep", switching_keep_mode),
    ("switching_escalate", switching_escalate_mode),
    ("switching_parallel", switching_parallel_mode),
    ("switching_forward", switching_forward_mode),
)
WINDOW_STAMP_NAMES = (
    "t_first_round",
    "t_data_complete",
    "t_queued",
    "t_dispatch",
    "t_done",
)


def accounted_round_keys(ledger):
    """(op, round) of every round a store ever accounted for."""
    keys = set()
    for event in ledger.events:
        if event.kind not in ("PUBLISHED", "STORED_SB1"):
            continue
        keys.add((event.op, event.round))
    return keys


def highest_emitted_round(ledger):
    """The last round each operation emitted, by operation."""
    highest = {}
    for event in ledger.events:
        if event.kind != "EMITTED":
            continue
        seen = highest.get(event.op, 0)
        highest[event.op] = max(seen, event.round)
    return highest


def unaccounted_rounds_of_window(window, accounted, highest):
    """The rounds one window read that no store ever accounted for.

    A sliding window's lookahead range is clamped to the operation's
    actual rounds, exactly as the retention reads it
    (decsim/windows/round_retention.py, read_keys_for_bounds).
    """
    emitted_high = highest[window.operation_id]
    input_high = min(window.buffer_hi, emitted_high)
    stop_round = input_high + 1
    missing = []
    for round_index in range(window.start_round, stop_round):
        key = (window.operation_id, round_index)
        if key not in accounted:
            missing.append(key)
    return missing


def unaccounted_window_rounds(machine):
    """Every decoded window's rounds that no store accounted for."""
    ledger = machine.observation.flight_recorder.ledger
    accounted = accounted_round_keys(ledger)
    highest = highest_emitted_round(ledger)
    windows = machine.observation.windows.windows
    decoded = windows.values()
    missing = []
    for window in decoded:
        if window.t_done is None:
            continue
        found = unaccounted_rounds_of_window(window, accounted, highest)
        missing.extend(found)
    return sorted(missing)


def window_stamps(window):
    """The stamps one window carries, in pipeline order, skipping None."""
    stamps = []
    for name in WINDOW_STAMP_NAMES:
        tick = getattr(window, name)
        if tick is None:
            continue
        stamps.append(tick)
    return stamps


def windows_with_unordered_stamps(machine):
    """The key of every decoded window whose stamps go backwards."""
    windows = machine.observation.windows.windows
    items = windows.items()
    unordered = []
    for key, window in items:
        if window.t_done is None:
            continue
        stamps = window_stamps(window)
        ordered = sorted(stamps)
        if stamps != ordered:
            unordered.append(key)
    return sorted(unordered)


@pytest.mark.parametrize("seed", range(24))
def test_the_ledger_holds_over_random_runs_of_every_mode(seed):
    """The property test of the ledger, over the run modes and 24 seeds.

    Each seed picks one of the seven shapes the declared builders offer
    and a round count from 6 to 9. Whatever the shape, the ledger passes
    its causal and conservation checks, every round a decoded window
    read is accounted for by a publication or a room-side landing, and
    each window's five stamps run forwards. The staged-pipeline mode of
    the older sweep is left out because the builders declare one preset
    decoder per tier and cannot express it.
    """
    generator = random.Random(seed)
    mode_index = seed % len(SWEEP_MODES)
    mode_name, build_mode = SWEEP_MODES[mode_index]
    rounds = generator.randint(6, 9)
    machine = build_mode(generator, rounds)
    ledger = machine.observation.flight_recorder.ledger

    ledger.check()

    unaccounted = unaccounted_window_rounds(machine)
    unordered = windows_with_unordered_stamps(machine)
    assert unaccounted == [], mode_name
    assert unordered == [], mode_name


def test_an_idle_feedback_memory_round_reaches_one_terminal_state():
    """An idle patch's rounds end in the feedback store, and only there.

    An idle stream's rounds are routed as feedback memory rather than as
    window input, so each must carry its FEEDBACK_MEMORY_DELIVERED
    terminal and no publication beside it (flight_recorder.py's
    _ROUND_TERMINALS).
    """
    first = declared_run.memory_operation(1)
    second = declared_run.memory_operation(2, blocked_by=1)
    operations = [first, second]
    machine = declared_run.weak_only_run(rounds=6, operations=operations)
    ledger = machine.observation.flight_recorder.ledger

    ledger.check()

    idle_stream = ("idle", 1, 1)
    kinds = terminal_kinds_of_operation(ledger, idle_stream)
    assert kinds
    assert set(kinds) == {"FEEDBACK_MEMORY_DELIVERED"}


# the four events a window's latency claim is read from: the window
# complete in its store, the decode started on a unit, the decode done,
# and the correction written into the frame
WINDOW_TRACE_KINDS = (
    "WINDOW_DATA_COMPLETE",
    "DECODE_STARTED",
    "DECODE_DONE",
    "FRAME_COMMITTED",
)


def window_keys(ledger):
    """(op, window) of every window the ledger carries, in a fixed order."""
    keys = set()
    for event in ledger.events:
        if event.window is None:
            continue
        keys.add((event.op, event.window))
    return sorted(keys, key=repr)


def trace_ticks_of_window(ledger, operation_id, window_id):
    """The tick of each of one window's trace events, by kind."""
    ticks = {}
    for event in ledger.events:
        if event.op != operation_id:
            continue
        if event.window != window_id:
            continue
        if event.kind not in WINDOW_TRACE_KINDS:
            continue
        ticks[event.kind] = event.tick
    return ticks


def missing_trace_kinds(ticks):
    """The trace kinds one window has no event of, in pipeline order."""
    missing = []
    for kind in WINDOW_TRACE_KINDS:
        if kind not in ticks:
            missing.append(kind)
    return missing


def trace_problem_of_window(ledger, operation_id, window_id):
    """What is wrong with one window's four trace events, or None.

    A kind the window has no event of, or a tick earlier than the tick
    of the kind before it, names the window it was found on.
    """
    ticks = trace_ticks_of_window(ledger, operation_id, window_id)
    missing = missing_trace_kinds(ticks)
    named = f"window {window_id} of operation {operation_id}"
    if missing:
        listed = ", ".join(missing)
        return f"{named} has no {listed}"
    walked = []
    for kind in WINDOW_TRACE_KINDS:
        tick = ticks[kind]
        walked.append(tick)
    forwards = sorted(walked)
    if walked != forwards:
        return f"{named} traces {walked} backwards"
    return None


def windows_with_a_broken_trace(ledger):
    """Every window of the ledger whose four trace events do not hold."""
    problems = []
    for operation_id, window_id in window_keys(ledger):
        problem = trace_problem_of_window(ledger, operation_id, window_id)
        if problem is None:
            continue
        problems.append(problem)
    return problems


def ledger_without(ledger, kind, window_id):
    """The same ledger with one window's event of that kind removed."""
    kept = []
    for event in ledger.events:
        if event.kind == kind and event.window == window_id:
            continue
        kept.append(event)
    return flight_recorder_module.RunLedgerView(events=tuple(kept))


def test_every_window_of_a_shipped_run_records_its_four_trace_events():
    """The shipped weak baseline, window by window, in pipeline order.

    One shot of configs/weak_decoder_baseline.yaml at p = 0.001,
    distance 3 and a 10 us round period decodes nine sliding windows,
    and each carries the four events a latency claim is read from, with
    non-decreasing ticks: the window complete in Buffer 0, its decode
    started on a unit, its decode done, and its correction written into
    the frame. Those are the four the hardware trace of a decoder FPGA
    reports (IBM arXiv 2510.21600 lines 517-519: when the decoders start
    and stop, when the syndromes and codewords arrive, and when the
    logical Pauli frame is produced), each one a named probe point a
    listener reads here (gem5 src/sim/probe/probe.hh lines 122 and 272).
    """
    config_path = yaml_configs.CONFIGS_DIR / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    task = config.point_task(
        physical_error_probability=0.001,
        distance=3,
        round_period_us=10.0,
        shots=1,
    )
    shot = collect.run_shot(task, 0)
    ledger = shot.machine.observation.flight_recorder.ledger

    problems = windows_with_a_broken_trace(ledger)

    assert problems == []
    keys = window_keys(ledger)
    assert keys == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
    ]


def test_a_window_whose_frame_write_is_missing_fails_the_trace_walk():
    """The walk has teeth, on the declared fabric where every tick is known.

    A window decoded and never written into the frame is a correction
    that vanished, so the walk names that window rather than reading as
    a complete run. The other five windows of the run still hold their
    four events, which is what makes the failure point at one window.
    """
    machine = declared_run.weak_only_run(rounds=6)
    ledger = machine.observation.flight_recorder.ledger
    intact = windows_with_a_broken_trace(ledger)
    broken = ledger_without(ledger, "FRAME_COMMITTED", 0)

    problems = windows_with_a_broken_trace(broken)

    assert intact == []
    assert problems == ["window 0 of operation 1 has no FRAME_COMMITTED"]
