"""The Chrome trace of gate point 1: its shape, its chains, its counts.

The event model and its worked example (weak_decoder_baseline d 3
p 0.003 seed 0, the frozen
suite's first strict point); the copy, reference and move counts are
data_path.md's hop table. The trace never moves a tick: a run with it on
narrates the same log and returns the same results as one with it off.
"""

import dataclasses
import gzip
import hashlib
import json

import pytest

import decsim.decoders.decoders as decoders
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.qpu.settings as qpu_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import tests.declared_run as declared_run
import tests.experiments.test_measure as measure_tests
import tests.observe.gate_point as gate_point

SEED = gate_point.SEED
POINT_LOG_SHA256 = gate_point.POINT_LOG_SHA256

PHASES = ("M", "X", "i", "C", "s", "t", "f")
_METADATA_NAMES = ("process_name", "thread_name", "thread_sort_index")
# the switching run's one-microsecond weak card
ONE_MICROSECOND_WEAK = {
    "kind": 1.0,
    "unit_memory": {"bits": None},
    "engine": {
        "clock": "fridge",
        "fetch_cycles_per_round": 1,
        "fetch_cycles_per_job": 0,
        "release_cycles_per_job": 10,
        "release_cycles_per_round": 0,
    },
}
# the withdrawal run of test_measure: a five-microsecond weak card under
# double_window, which takes back a queued request when it re-slices
RE_SLICED_WINDOWS = {
    "escalation": {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "double_window",
    },
    "weak_decoder": {
        "kind": 5.0,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": measure_tests.ONE_CYCLE_FETCH_TEN_CYCLE_RELEASE,
    },
}


def _settings(trace_path=None, data_movement=False):
    trace = "off"
    if trace_path is not None:
        trace = str(trace_path)
    return gate_point.settings(trace=trace, data_movement=data_movement)


def _run(trace_path=None, data_movement=False):
    point = _settings(trace_path, data_movement)
    machine = machine_module.Machine.build(point, SEED)
    result = machine.run()
    return machine, result


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    """One traced run of gate point 1, with its file already written."""
    directory = tmp_path_factory.mktemp("trace")
    path = directory / "point1.trace.json"
    machine, result = _run(path, data_movement=True)
    machine.observation.trace_writer.write(str(path))
    text = path.read_text()
    document = json.loads(text)
    return machine, result, document


def _log_sha(machine) -> str:
    text = "\n".join(machine.observation.log.lines)
    encoded = text.encode()
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def _by_phase(document, phase) -> list:
    return [row for row in document if row["ph"] == phase]


def _rows_with(rows, key: str, value) -> list:
    """The rows whose key holds that value, in document order."""
    matching = []
    for row in rows:
        if row.get(key) == value:
            matching.append(row)
    return matching


def _rows_with_arg(rows, key: str, value) -> list:
    """The rows whose args hold that value under key, in document order."""
    matching = []
    for row in rows:
        if row["args"].get(key) == value:
            matching.append(row)
    return matching


def _flow_rows(document) -> list:
    flows = []
    for row in document:
        if row["ph"] in ("s", "t", "f"):
            flows.append(row)
    return flows


def _chains_by_id(flows) -> dict:
    """Each flow id: the (category, round, window) chains that use it."""
    chains_by_id = {}
    for row in flows:
        args = row["args"]
        followed = (row["cat"], args.get("round"), args.get("window"))
        chains = chains_by_id.setdefault(row["id"], set())
        chains.add(followed)
    return chains_by_id


def _recorded_steps_of(events, round_key: tuple) -> list:
    """(kind, tick) of every recorded event of one round, in order."""
    steps = []
    for event in events:
        if (event.operation_id, event.round_index) == round_key:
            steps.append((event.kind, event.tick))
    return steps


def _flow_of(document, kind, key_text) -> list:
    """The events of the chain that follows that round or that window."""
    rows = []
    for row in document:
        if row["ph"] not in ("s", "t", "f"):
            continue
        if row["args"].get(kind) == key_text:
            rows.append(row)
    return rows


def test_the_trace_moves_no_tick_and_narrates_the_same_log(tmp_path):
    """Rule: the writer is a listener, so the run is the same run."""
    trace_path = tmp_path / "on.trace.json"
    plain, plain_result = _run()
    traced, traced_result = _run(trace_path)
    plain_sha = _log_sha(plain)
    plain_frame = plain.control.pauli_frame.snapshot()
    traced_frame = traced.control.pauli_frame.snapshot()

    assert plain_sha == _log_sha(traced)
    assert plain_sha.startswith(POINT_LOG_SHA256)
    assert plain.engine.now == traced.engine.now
    assert plain_result.link_traffic == traced_result.link_traffic
    assert plain_result.operation_results == traced_result.operation_results
    assert plain_frame.records == traced_frame.records
    assert (
        plain.observation.queue_depth.samples
        == traced.observation.queue_depth.samples
    )


def test_a_run_with_no_trace_builds_no_writer():
    """The section did not ask, so nothing is connected."""
    point = _settings()
    machine = machine_module.Machine.build(point, SEED)

    assert machine.observation.trace_writer is None
    assert machine.observation.data_movement is None


def test_the_trace_is_not_a_reason_to_build_the_counters(tmp_path):
    """Every listener is built because the section asked, and only then.

    The law of the slice is that a run with the trace off is the same
    run as one with it on, in the results as well as the log, so the
    RunResult's data movement counters are None in both when the
    section asked for the trace alone.
    """
    trace_path = tmp_path / "on.trace.json"
    plain_machine, plain_result = _run()
    traced_machine, traced_result = _run(trace_path)

    assert traced_machine.observation.trace_writer is not None
    assert traced_machine.observation.data_movement is None
    assert plain_machine.observation.data_movement is None
    plain_fields = dataclasses.asdict(plain_result)
    traced_fields = dataclasses.asdict(traced_result)
    assert traced_fields == plain_fields
    assert traced_result.data_movement is None


def test_a_path_ending_in_gz_holds_the_same_trace_compressed(traced, tmp_path):
    """The reference yaml's own promise: .gz compresses."""
    machine, _result, document = traced
    path = tmp_path / "point1.trace.json.gz"

    machine.observation.trace_writer.write(str(path))

    with gzip.open(path, "rt") as handle:
        compressed = json.load(handle)
    assert compressed == document


def test_every_event_carries_the_fields_its_phase_declares(traced):
    """The JSON schema of the trace note's section 2, checked row by row."""
    _machine, _result, document = traced
    event_rows = _event_rows(document)
    tids_with_events = {row["tid"] for row in event_rows}

    _check_every_row(document)

    named_tids = _thread_names(document)
    assert tids_with_events <= set(named_tids)


def _event_rows(document) -> list:
    """Every row but the metadata ones."""
    events = []
    for row in document:
        if row["ph"] != "M":
            events.append(row)
    return events


def _check_every_row(document) -> None:
    for row in document:
        _check_row(row)


def test_every_flow_event_lies_inside_a_complete_event_on_its_thread(traced):
    """A flow's binding point is its enclosing slice, so it must have one."""
    _machine, _result, document = traced
    complete_rows = _by_phase(document, "X")
    spans_by_tid = _tick_spans_by_tid(complete_rows)
    flows = _flow_rows(document)

    uncovered = _rows_outside_every_span(flows, spans_by_tid)

    assert uncovered == []


def _tick_spans_by_tid(complete_rows) -> dict:
    """Each lane's complete events as (start tick, end tick)."""
    spans_by_tid = {}
    for row in complete_rows:
        start = row["args"]["tick"]
        microseconds = row["dur"] * 1_000_000
        duration = round(microseconds)
        end = start + int(duration)
        spans = spans_by_tid.setdefault(row["tid"], [])
        spans.append((start, end))
    return spans_by_tid


def _lanes_with_overlapping_spans(spans_by_tid: dict) -> list:
    """The lanes on which a span starts before the one before it ends."""
    overlapping = []
    for thread_id in sorted(spans_by_tid):
        ordered = sorted(spans_by_tid[thread_id])
        pairs = zip(ordered, ordered[1:], strict=False)
        if any(later[0] < earlier[1] for earlier, later in pairs):
            overlapping.append(thread_id)
    return overlapping


def _rows_outside_every_span(rows, spans_by_tid: dict) -> list:
    """The rows whose tick no span on their own lane covers."""
    outside = []
    for row in rows:
        tick = row["args"]["tick"]
        spans = spans_by_tid.get(row["tid"], ())
        covered = any(start <= tick <= end for start, end in spans)
        if not covered:
            outside.append(row)
    return outside


@pytest.mark.parametrize("run_both_at_once", [False, True])
def test_each_decodes_stages_are_on_the_lane_of_the_unit_that_ran_it(
    tmp_path, run_both_at_once
):
    """Two decodes of one window on two units keep two lanes.

    The forced-class pair of a complementary gap runs on both weak
    units at once, and under run_both_at_once the speculative strong
    decode does too. A slice's tid is its lane, and slices on one lane must nest
    (Perfetto, "Other trace formats": overlapping, non-nested events
    are out of spec), so every lane that served a decode carries that
    decode's stages and no two of its stages overlap.
    """
    trace_path = tmp_path / "pair.trace.json"
    observation = {"trace": str(trace_path)}
    weak_decoder = {**ONE_MICROSECOND_WEAK, "units": 2}
    sections = {"weak_decoder": weak_decoder, "observation": observation}
    shot = measure_tests.switching_run(
        tmp_path,
        20.0,
        run_both_at_once=run_both_at_once,
        strong_units=2,
        sections=sections,
    )
    document = shot.machine.observation.trace_writer.document()
    complete_rows = _by_phase(document, "X")
    stages = _rows_with(complete_rows, "cat", "stage")
    services = _rows_with(complete_rows, "cat", "window,service")

    stage_spans_by_tid = _tick_spans_by_tid(stages)
    service_lanes = {row["tid"] for row in services}
    assert len(service_lanes) > 2
    assert set(stage_spans_by_tid) == service_lanes
    assert _lanes_with_overlapping_spans(stage_spans_by_tid) == []


def test_a_withdrawn_request_leaves_the_queue_when_it_is_withdrawn(tmp_path):
    """A queued request taken back ends its queue slice there.

    double_window withdraws window 3's queued request 1:3:weak:6 when it
    re-slices the window, 1.132 us after it joined the queue, the wait
    measure books as that window's admission_wait. Ciw ends a reneging
    customer's record at the reneging (ciw/node.py
    write_reneging_record), so the slice ends there too rather than at
    the end of the run.
    """
    trace_path = tmp_path / "withdrawn.trace.json"
    observation = {"trace": str(trace_path)}
    sections = {**RE_SLICED_WINDOWS, "observation": observation}
    shot = measure_tests.switching_run(tmp_path, 20.0, sections=sections)
    document = shot.machine.observation.trace_writer.document()
    complete_rows = _by_phase(document, "X")
    queued = _rows_with(complete_rows, "cat", "window,queue")
    withdrawn_rows = _rows_with_arg(queued, "request", "1:3:weak:6")
    withdrawn = withdrawn_rows[0]
    reasons = {row["args"].get("freed_reason") for row in queued}

    assert withdrawn["args"]["freed_reason"] == "withdrawn"
    assert withdrawn["ts"] == 15.008
    assert withdrawn["dur"] == 1.132
    assert "end of run" not in reasons


def test_two_decodes_with_no_request_keep_the_lanes_of_their_units():
    """A factory's two correction decodes at once run on two lanes.

    A correction decode serves no request, so the window and its run
    ordinals are the same for both. The stage record names the unit
    that ran it, as each LLVM XRay record carries its thread, and each
    decode's stages sit on its own service's lane.
    """
    factory = qpu_settings.factory_from_yaml(
        {
            "kind": "distillation",
            "unit_count": 1,
            "attempt_ticks": 100,
            "correction_round_count": 3,
            "correction_decode_count": 2,
            "production_mode": "continuous",
            "buffer_capacity": 1,
        }
    )
    point = gate_point.settings(trace="chrome")
    weak_decoder = dataclasses.replace(point.weak_decoder, unit_count=2)
    point = dataclasses.replace(
        point, weak_decoder=weak_decoder, magic_state_factory=factory
    )
    machine = machine_module.Machine.build(point, SEED)
    machine.run()

    document = machine.observation.trace_writer.document()
    complete_rows = _by_phase(document, "X")
    stages = _rows_with(complete_rows, "cat", "stage")
    services = _rows_with(complete_rows, "cat", "window,service")
    stage_spans_by_tid = _tick_spans_by_tid(stages)
    service_lanes = {row["tid"] for row in services}
    assert len(service_lanes) == 2
    assert set(stage_spans_by_tid) == service_lanes
    assert _lanes_with_overlapping_spans(stage_spans_by_tid) == []


def test_every_flow_chain_has_a_number_of_its_own(traced):
    """A flow's id is a number and one chain's events share it.

    The spec's ids are numbers (Trace Event Format, Async Events, which
    flow events follow), and Perfetto's trace processor drops a flow
    whose id does not read as one (its stat flow_invalid_id), so the
    round or window a chain follows is in its args.
    """
    _machine, _result, document = traced
    flows = _flow_rows(document)
    chains_by_id = _chains_by_id(flows)

    starts = _by_phase(document, "s")
    assert all(isinstance(flow_id, int) for flow_id in chains_by_id)
    assert len(chains_by_id) == len(starts)
    assert all(len(chains) == 1 for chains in chains_by_id.values())


def test_round_ones_first_hops_are_the_notes_worked_example(traced):
    """The note's example, hop by hop, for the first round of the point."""
    _machine, _result, document = traced
    instants = _by_phase(document, "i")
    emitted = _rows_with(instants, "name", "emitted round 1")
    (emitted_row,) = emitted
    assert emitted_row["args"]["tick"] == 1_000_000
    assert emitted_row["args"]["bits"] == 8

    complete_rows = _by_phase(document, "X")
    round_moves = _rows_with(complete_rows, "cat", "round,link")
    moves = _rows_with_arg(round_moves, "rounds_by_operation", {"1": "1..1"})
    assert [row["args"]["tick"] for row in moves] == [1_000_000, 1_004_000]
    assert [row["args"]["delivery_tick"] for row in moves] == [
        1_004_000,
        1_008_000,
    ]
    # the readout hop carries the eight measurement outcomes; the store
    # hop carries what left the controller, the four detection events
    # this card's row formed there
    assert [row["args"]["bits"] for row in moves] == [8, 4]

    # the round takes its weak syndrome buffer slot where its bits are, at
    # the landing of the hop that carried them, which is also when it
    # becomes readable
    residence = _one(document, "X", "round 1")
    assert residence["cat"] == "round,residence"
    assert residence["args"]["tick"] == 1_008_000
    assert residence["dur"] == 5.004
    # the store holds the four detection events of round 1, which is
    # what the hop into it carried
    assert residence["args"]["bits"] == 4
    assert residence["args"]["data_ready"] == 1_008_000
    assert residence["args"]["freed"] == 6_012_000
    assert residence["args"]["capacity_bits"] is None


def test_window_zeros_service_and_stages_are_the_notes_worked_example(traced):
    """W0 queued, moved, resident, staged, returned and committed."""
    _machine, _result, document = traced
    queued = _one(document, "X", "W0 queued")
    assert queued["args"]["tick"] == 6_008_000

    complete_rows = _by_phase(document, "X")
    window_moves = _rows_with(complete_rows, "cat", "window,link")
    input_moves = _rows_with_arg(
        window_moves, "channel", "weak_buffer_to_weak_decoder"
    )
    input_move = input_moves[0]
    assert input_move["args"]["tick"] == 6_008_000
    assert input_move["args"]["bits"] == 44
    assert input_move["args"]["rounds_by_operation"] == {"1": "1..6"}

    resident = _one(document, "X", "W0 input in memory")
    assert resident["args"]["tick"] == 6_012_000
    assert resident["dur"] == 0.092
    assert resident["args"]["bits"] == 44
    assert resident["args"]["capacity_bits"] is None
    assert resident["args"]["freed"] == 6_104_000

    stages = _rows_with(complete_rows, "cat", "stage")
    first_three = stages[:3]
    assert [row["name"] for row in first_three] == [
        "fetch",
        "algorithm",
        "release",
    ]
    assert [row["args"]["tick"] for row in first_three] == [
        6_012_000,
        6_036_000,
        6_064_000,
    ]
    assert [row["args"]["cycles"] for row in first_three] == [6, None, 10]

    service = _one(document, "X", "W0 service")
    assert service["args"]["tick"] == 6_012_000
    assert service["dur"] == 0.092

    correction = _one(document, "X", "1:0 correction")
    assert correction["args"]["tick"] == 6_108_000
    assert correction["args"]["tier"] == "weak"


def test_one_rounds_flow_chain_equals_the_round_events_recorded(traced):
    """The file's chain for round 1 is the recorder's chain for round 1."""
    machine, _result, document = traced
    events = machine.observation.round_events.events
    recorded = _recorded_steps_of(events, (1, 1))

    flow = _flow_of(document, "round", "1:1")
    assert [row["ph"] for row in flow] == ["s", "t", "t", "t", "f"]
    flow_ticks = [row["args"]["tick"] for row in flow]
    assert flow_ticks == [1_000_000, 1_004_000, 1_008_000, 6_008_000, 6_012_000]
    # the recorder's own chain for the same round; the store step of the
    # flow is the landing that takes the slot, the tick the recorder
    # calls PUBLISHED, and the flow ends one hop later, in the unit's
    # memory
    assert recorded == [
        ("EMITTED", 1_000_000),
        ("BINARY_AVAILABLE", 1_004_000),
        ("PACKED", 1_004_000),
        ("CWB_SENT", 1_004_000),
        ("PUBLISHED", 1_008_000),
    ]


def test_the_data_movement_counts_are_the_hop_tables(traced):
    """data_path.md sections 3 and 4, counted per round as the table does.

    Predicted before the run: 30 rounds each copied at the controller's
    intake, at its assembler and into the weak syndrome buffer (hops 1 to 3),
    one unit-memory copy per window of the six rounds it reads (hop 5, nine
    windows of six), and the masked view each window past the first
    folds its predecessor's boundary into (eight of six): 30 x 3 + 54 +
    48 = 192 copied rounds. The nine window input holds reference 54
    rounds where they sit; each hold is registered, transferred to its
    request and released, so 27 hold events. The links carry 86 moves.
    """
    machine, _result, _document = traced
    counts = machine.observation.data_movement.json_value()

    assert counts["rounds"] == 30
    assert counts["copied_rounds"] == 192
    assert counts["referenced_rounds"] == 54
    assert counts["hold_events"] == 27
    assert counts["moves"] == 86
    assert counts["copies"] == 107
    by_path = counts["copies_by_path"]
    assert by_path["readout -> controller intake"]["rounds"] == 30
    assert by_path["controller intake -> controller assembler"]["rounds"] == 30
    intake = by_path["controller assembler -> weak syndrome buffer"]
    to_unit = by_path["weak syndrome buffer -> unit default#0 memory"]
    assert intake["rounds"] == 30
    assert to_unit["rounds"] == 54
    assert by_path["unit default#0 memory -> masked view"]["rounds"] == 48


def test_the_movement_counts_group_by_the_memory_class_they_cross(traced):
    """Horowitz ISSCC 2014 lines 232-247, Dally CACM 2020 lines 231-234.

    The class is the cost, so the report keeps the classes apart and
    every class row sums back to the run's total. On this point the
    readout's own hop is the only off-board move, the write into Buffer
    0, the window's read out of it and the correction to the frame are
    on board, and the boundary handoff stays on chip; every copy but the
    write into the weak syndrome buffer lands in a register or a unit's own
    memory.
    """
    machine, _result, _document = traced
    counts = machine.observation.data_movement.json_value()

    copies = counts["copies_by_memory_class"]
    moves = counts["moves_by_memory_class"]
    assert list(copies) == ["on_chip", "on_board"]
    assert list(moves) == ["on_chip", "on_board", "off_board"]
    assert copies["on_board"]["rounds"] == 30
    assert moves["off_board"]["rounds"] == 30
    assert moves["on_board"]["rounds"] == 138
    copied_events = [row["events"] for row in copies.values()]
    moved_events = [row["events"] for row in moves.values()]
    assert sum(copied_events) == counts["copies"]
    assert sum(moved_events) == counts["moves"]
    copied_bits = [row["bits"] for row in copies.values()]
    assert sum(copied_bits) == counts["copy_bits"]


def test_a_windows_flow_joins_its_queue_its_moves_its_unit_and_the_frame(
    traced,
):
    """The second flow of the trace note's section 2, for W0.

    A window's chain is its own, apart from the round flows: it begins
    where the request joins the ready queue, steps over the input move,
    the unit's memory and the result move (and over the boundary move
    out to the next window), and ends where the frame holds the
    correction.
    """
    _machine, _result, document = traced
    threads = _thread_names(document)
    flow = _flow_of(document, "window", "1:0")
    lanes = [
        (row["ph"], threads[row["tid"]], row["args"]["tick"]) for row in flow
    ]
    categories = {row["cat"] for row in flow}
    names = {row["name"] for row in flow}

    assert lanes == [
        ("s", "Window planner", 6_008_000),
        ("t", "weak_buffer_to_weak_decoder", 6_008_000),
        ("t", "Decoder unit default#0", 6_012_000),
        ("t", "decoder_to_decoder", 6_104_000),
        ("t", "weak_decoder_to_frame", 6_104_000),
        ("f", "Frame", 6_108_000),
    ]
    assert categories == {"window"}
    assert names == {"W0"}


def test_every_window_is_ready_when_its_last_round_is_readable(traced):
    """One W ready instant per window, at that round's own data_ready.

    The moment is the input gate's: the window's data is complete when
    the last round it reads is published in the store, which is the tick
    the store residence records as data_ready for that round.
    """
    machine, _result, document = traced
    instants = _by_phase(document, "i")
    ready = _ready_rows(instants)
    windows = machine.observation.windows.windows
    ready_ticks = [row["args"]["tick"] for row in ready]
    last_round_ticks = _last_rounds_data_ready(document, ready)

    assert len(ready) == len(windows)
    assert len(ready) == 9
    assert ready_ticks == last_round_ticks


def _ready_rows(instants) -> list:
    ready = []
    for row in instants:
        if row["name"].endswith(" ready"):
            ready.append(row)
    return ready


def _last_rounds_data_ready(document, ready) -> list:
    """For each ready window, the data_ready tick of the last round it reads."""
    ticks = []
    for row in ready:
        (read_rounds,) = row["args"]["rounds_by_operation"].values()
        first_and_last = read_rounds.split("..")
        last_round = first_and_last[1]
        residence = _one(document, "X", f"round {last_round}")
        ticks.append(residence["args"]["data_ready"])
    return ticks


class _HeldBits:
    """The memory's own occupied bits, read at every deposit and take."""

    def __init__(self, memory) -> None:
        self.memory = memory
        self.values: list = []

    def changed(self, _job, _decoder_input) -> None:
        """One deposit or take: the bits the memory holds after it."""
        self.values.append(self.memory.occupied_bits)


def test_the_unit_memory_counter_is_the_memorys_own_occupancy(tmp_path):
    """The C track of the unit's memory steps with the memory's own count."""
    path = tmp_path / "point1.trace.json"
    point = _settings(path, data_movement=True)
    machine = machine_module.Machine.build(point, SEED)
    (unit,) = machine.decoders.decoder_manager.pool.units
    held = _HeldBits(unit.memory)
    unit.memory.trace.deposited.connect(held.changed)
    unit.memory.trace.taken.connect(held.changed)
    machine.run()
    machine.observation.trace_writer.write(str(path))
    text = path.read_text()
    document = json.loads(text)
    name = f"{unit.memory.name} bits"

    values = _counter_values(document, name, "bits")

    assert values
    assert values == held.values


def test_the_assembler_workspace_holds_round_one_until_it_is_packed(traced):
    """The one bounded controller-side structure, as a residence.

    A round enters the packing workspace at its first fragment and
    leaves when it is packed; the point prices no packing time, so round
    1 is in and out at 1.004 us. The counter carries the occupancy and
    the residence carries the bound the yaml sets, unbounded here.
    """
    _machine, _result, document = traced
    residence = _one(document, "X", "assemble round 1")

    assert residence["cat"] == "round,residence"
    assert residence["args"]["tick"] == 1_004_000
    assert residence["dur"] == 0.0
    assert residence["args"]["slot_taken"] == 1_004_000
    assert residence["args"]["freed"] == 1_004_000
    assert residence["args"]["freed_reason"] == "packed"
    assert residence["args"]["capacity"] is None
    steps = _counter_values(document, "controller assembler rounds", "rounds")
    assert max(steps) == 1
    assert steps[-1] == 0


def test_a_round_waits_for_a_packing_place_before_it_takes_one():
    """The stage never holds more rounds than its bound, in the trace too.

    On the declared card with a bound of one, round 1 takes the place at
    its emission, 1 us, and is published at 10 us. Round 2 is emitted at
    2 us, waits in front of the stage from then, and takes round 1's
    place at 10 us (controller/round_assembly.py).
    """
    controller = declared_run.declared_controller(packing_rounds_in_flight=1)
    observation = observe_settings.ObservationSettings(trace="chrome")
    machine = declared_run.weak_only_run(
        rounds=3, controller=controller, observation=observation
    )
    document = machine.observation.trace_writer.document()

    waiting = _one(document, "X", "wait round 2 for a packing place")
    assembly = _one(document, "X", "assemble round 2")
    in_stage = _counter_values(
        document, "controller assembler rounds", "rounds"
    )
    assert waiting["args"]["held_from"] == 2_000_000
    assert waiting["args"]["freed"] == 10_000_000
    assert assembly["args"]["slot_taken"] == 10_000_000
    assert assembly["args"]["capacity"] == 1
    assert max(in_stage) == 1


def _counter_values(document, name: str, series: str) -> list:
    """The values one named counter plotted, in document order."""
    counters = _by_phase(document, "C")
    named = _rows_with(counters, "name", name)
    return [row["args"][series] for row in named]


def _check_row(row) -> None:
    """One event of the trace against the fields its phase declares."""
    assert row["ph"] in PHASES, row
    assert row["pid"] == 1
    assert isinstance(row["tid"], int)
    if row["ph"] == "M":
        assert row["name"] in _METADATA_NAMES
        return
    assert isinstance(row["ts"], float)
    _check_tick(row)
    _check_phase_fields(row)


def _check_tick(row) -> None:
    """A counter keeps its tick in ts alone; every other event states it."""
    if row["ph"] == "C":
        # catapult makes one series per key of a counter's args, so a
        # counter carries its value and nothing else; ts holds the tick
        assert "tick" not in row["args"]
        return
    assert isinstance(row["args"]["tick"], int)
    assert row["args"]["tick"] / 1_000_000 == row["ts"]


def _check_phase_fields(row) -> None:
    """The fields a complete, instant or flow event adds."""
    if row["ph"] == "X":
        assert isinstance(row["dur"], float)
        assert row["dur"] >= 0
    if row["ph"] == "i":
        assert row["s"] == "t"
    if row["ph"] in ("s", "t", "f"):
        assert isinstance(row["id"], int)
    if row["ph"] == "f":
        assert row["bp"] == "e"


def _thread_names(document) -> dict:
    """Every named lane of the trace, by its tid."""
    names = {}
    for row in document:
        if row["name"] == "thread_name":
            names[row["tid"]] = row["args"]["name"]
    return names


def _one(document, phase, name) -> dict:
    """The one event of that phase and name; a second is a failure."""
    rows = []
    for row in document:
        if row["ph"] == phase and row["name"] == name:
            rows.append(row)
    (found,) = rows
    return found


def test_the_observation_section_names_the_log_and_the_trace_apart():
    """The narrator is observation.log; observation.trace is the trace."""
    import decsim.observe.settings as observe_settings

    default = observe_settings.ObservationSettings.from_yaml({})
    assert default.log == "off"
    assert default.trace == "off"
    assert default.writes_trace is False
    assert default.trace_path is None
    assert default.trace_shots == (0,)

    narrating = observe_settings.ObservationSettings.from_yaml({"log": "both"})
    assert narrating.prints_log is True
    assert narrating.writes_log is True
    assert narrating.writes_trace is False

    chrome = observe_settings.ObservationSettings.from_yaml(
        {"trace": "chrome", "trace_shots": [0, 3]}
    )
    assert chrome.writes_trace is True
    assert chrome.trace_path is None
    assert chrome.trace_shots == (0, 3)

    at_a_path = observe_settings.ObservationSettings.from_yaml(
        {"trace": "/tmp/run.trace.json"}
    )
    assert at_a_path.trace_path == "/tmp/run.trace.json"

    with pytest.raises(ValueError, match="observation.log must be one of"):
        observe_settings.ObservationSettings.from_yaml({"log": "chrome"})


def _corrections(document) -> list:
    corrections = []
    for row in document:
        if _is_a_correction(row):
            corrections.append(row)
    return corrections


def _is_a_correction(row) -> bool:
    """The residence of one window's correction on the frame's lane."""
    is_complete = row["ph"] == "X"
    return is_complete and row["name"].endswith(" correction")


def test_a_write_with_no_prediction_is_traced_without_observables(tmp_path):
    """A result may carry no observables, and the residence then names none.

    decsim/ports.py's Decoder.decode returns a result whose correction
    and logical observables are optional, and PresetLatencyDecoder
    leaves both unset; the frame carries that None into its record.
    So the writer names the bits of a write only when the write has
    some, and the trace of a timing-only run is written rather than
    raising on the frame's commit.
    """
    trace_path = tmp_path / "timing_only.trace.json"
    point = _settings(trace_path)
    timing_only = decoders.PresetLatencyDecoder.Settings(1.0)
    weak_decoder = dataclasses.replace(
        point.weak_decoder, algorithm=timing_only
    )
    point = dataclasses.replace(point, weak_decoder=weak_decoder)
    machine = machine_module.Machine.build(point, SEED)
    result = machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    text = trace_path.read_text()
    document = json.loads(text)
    corrections = _corrections(document)
    names_committed = {"committed" in row["args"] for row in corrections}
    names_observables = {"observables" in row["args"] for row in corrections}

    assert result.terminal_status == "complete"
    assert names_committed == {True}
    assert names_observables == {False}


def test_a_counter_row_carries_one_series_and_the_tick_stays_in_ts(traced):
    """Catapult's importer makes one series per args key.

    trace_event_importer 402-431 reads a counter's args as the series to
    plot, so a tick beside the value would plot a second series five
    orders of magnitude larger and flatten the occupancy the reader
    came for.
    """
    _machine, _result, document = traced
    counters = _by_phase(document, "C")
    series_counts = {len(row["args"]) for row in counters}
    series_names = {name for row in counters for name in row["args"]}
    values = [value for row in counters for value in row["args"].values()]
    value_is_whole = {isinstance(value, int) for value in values}
    assert series_counts == {1}
    assert "tick" not in series_names
    assert value_is_whole == {True}


def test_each_port_access_is_one_span_on_its_ports_lane(tmp_path):
    """A ported store's port serves one access at a time, in arrival order.

    gem5's SimpleMemory is busy for an access's duration and refuses the
    next until it frees (src/mem/simple_mem.cc:136-161), so the spans on
    one port's lane never overlap.
    """
    path = tmp_path / "ported.trace.json"
    point = _settings(path)
    weak_store = point.weak_syndrome_buffer
    ported = ported_syndrome_buffer.PortedSyndromeBufferSettings(
        bits=weak_store.bits, clock=weak_store.clock
    )
    point = dataclasses.replace(point, weak_syndrome_buffer=ported)
    machine = machine_module.Machine.build(point, SEED)
    machine.run()
    machine.observation.trace_writer.write(str(path))
    text = path.read_text()
    document = json.loads(text)

    accesses = _rows_with(document, "cat", "access")
    names = {row["name"] for row in accesses}
    tids = {row["tid"] for row in accesses}
    first_port, second_port = sorted(tids)
    assert names == {"read", "write"}
    assert _lane_spans_overlap(accesses, first_port) is False
    assert _lane_spans_overlap(accesses, second_port) is False


def _lane_spans_overlap(accesses, thread_id) -> bool:
    """Whether a span on this lane starts before the one before it ends."""
    spans = [row for row in accesses if row["tid"] == thread_id]
    ends = [row["ts"] + row["dur"] for row in spans[:-1]]
    starts = [row["ts"] for row in spans[1:]]
    return any(start < end for start, end in zip(starts, ends, strict=True))


def test_each_transfer_is_one_frame_on_its_channels_frame_lane(traced):
    """An ideal channel moves each transfer as one frame, one at a time.

    ns-3's point-to-point device starts a packet only when its
    transmitter is READY (point-to-point-net-device.cc, TransmitStart),
    so the frames on one channel's lane never overlap, one per move.
    """
    _machine, _result, document = traced
    frames = _rows_with(document, "cat", "frame")
    moves = _moves(document)
    frame_lanes = {row["tid"] for row in frames}
    overlapping = _overlapping_lanes(frames, frame_lanes)

    assert len(frames) == len(moves)
    assert overlapping == []


def _moves(document) -> list:
    moves = []
    for row in document:
        if _is_a_move(row):
            moves.append(row)
    return moves


def _overlapping_lanes(rows, lanes) -> list:
    overlapping = []
    for thread_id in sorted(lanes):
        if _lane_spans_overlap(rows, thread_id):
            overlapping.append(thread_id)
    return overlapping


def _is_a_move(row) -> bool:
    """Whether the row is a move on a link path's lane."""
    category = row.get("cat", "")
    return category.endswith(",link")
