"""The Chrome trace of gate point 1: its shape, its chains, its counts.

The event model and its worked example (weak_decoder_baseline d 3
p 0.003 seed 0, the frozen
suite's first strict point); the copy, reference and move counts are
data_path.md's hop table. The trace never moves a tick: a run with it on
narrates the same log and returns the same results as one with it off.
"""

import dataclasses
import hashlib
import json

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import tests.observe.gate_point as gate_point

SEED = gate_point.SEED
POINT_LOG_SHA256 = gate_point.POINT_LOG_SHA256

PHASES = ("M", "X", "i", "C", "s", "t", "f")
_METADATA_NAMES = ("process_name", "thread_name", "thread_sort_index")


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


def _flow_of(document, flow_id) -> list:
    rows = []
    for row in document:
        if row["ph"] in ("s", "t", "f") and row["id"] == flow_id:
            rows.append(row)
    return rows


def test_the_trace_moves_no_tick_and_narrates_the_same_log(tmp_path):
    """Rule: the writer is a listener, so the run is the same run."""
    trace_path = tmp_path / "on.trace.json"
    plain, plain_result = _run()
    traced, traced_result = _run(trace_path)
    plain_sha = _log_sha(plain)
    plain_frame = plain.pauli_frame.snapshot()
    traced_frame = traced.pauli_frame.snapshot()

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


def test_every_event_carries_the_fields_its_phase_declares(traced):
    """The JSON schema of the trace note's section 2, checked row by row."""
    _machine, _result, document = traced
    tids_with_events = set()

    for row in document:
        _check_row(row)
        if row["ph"] != "M":
            tids_with_events.add(row["tid"])

    named_tids = _thread_names(document)
    assert tids_with_events <= set(named_tids)


def test_every_flow_event_lies_inside_a_complete_event_on_its_thread(traced):
    """A flow's binding point is its enclosing slice, so it must have one."""
    _machine, _result, document = traced
    spans_by_tid = {}
    for row in _by_phase(document, "X"):
        start = row["args"]["tick"]
        microseconds = row["dur"] * 1_000_000
        duration = round(microseconds)
        end = start + int(duration)
        spans = spans_by_tid.setdefault(row["tid"], [])
        spans.append((start, end))

    for row in document:
        if row["ph"] not in ("s", "t", "f"):
            continue
        tick = row["args"]["tick"]
        spans = spans_by_tid.get(row["tid"], ())
        covered = any(start <= tick <= end for start, end in spans)
        assert covered, row


def test_round_ones_first_hops_are_the_notes_worked_example(traced):
    """The note's example, hop by hop, for the first round of the point."""
    _machine, _result, document = traced
    emitted = [
        row
        for row in document
        if row["ph"] == "i" and row["name"] == "emitted round 1"
    ]
    (emitted_row,) = emitted
    assert emitted_row["args"]["tick"] == 1_000_000
    assert emitted_row["args"]["bits"] == 8

    moves = [
        row
        for row in _by_phase(document, "X")
        if row["cat"] == "round,link" and row["args"]["rounds"] == "1..1"
    ]
    assert [row["args"]["tick"] for row in moves] == [1_000_000, 1_004_000]
    assert [row["args"]["delivery_tick"] for row in moves] == [
        1_004_000,
        1_008_000,
    ]
    assert [row["args"]["bits"] for row in moves] == [8, 8]

    residence = _one(document, "X", "round 1")
    assert residence["cat"] == "round,residence"
    assert residence["args"]["tick"] == 1_004_000
    assert residence["dur"] == 5.008
    # the store holds detection events, four per round at d 3 memory z;
    # the eight of the note's example are the raw measurement bits the
    # links carry, which stop at the assembler (round_assembly.py)
    assert residence["args"]["bits"] == 4
    assert residence["args"]["data_ready"] == 1_008_000
    assert residence["args"]["freed"] == 6_012_000
    assert residence["args"]["capacity"] is None


def test_window_zeros_service_and_stages_are_the_notes_worked_example(traced):
    """W0 queued, moved, resident, staged, returned and committed."""
    _machine, _result, document = traced
    queued = _one(document, "X", "W0 queued")
    assert queued["args"]["tick"] == 6_008_000

    input_move = [
        row
        for row in _by_phase(document, "X")
        if row["cat"] == "window,link"
        and row["args"]["channel"] == "weak_buffer_to_weak_decoder"
    ][0]
    assert input_move["args"]["tick"] == 6_008_000
    assert input_move["args"]["bits"] == 44
    assert input_move["args"]["rounds"] == "1..6"

    resident = _one(document, "X", "W0 input in memory")
    assert resident["args"]["tick"] == 6_012_000
    assert resident["dur"] == 0.092
    assert resident["args"]["bits"] == 44
    assert resident["args"]["freed"] == 6_104_000

    stages = [row for row in _by_phase(document, "X") if row["cat"] == "stage"]
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
    recorded = []
    for event in machine.observation.round_events.events:
        if (event.operation_id, event.round_index) != (1, 1):
            continue
        recorded.append((event.kind, event.tick))

    flow = _flow_of(document, "1:1")
    assert [row["ph"] for row in flow] == ["s", "t", "t", "t", "f"]
    flow_ticks = [row["args"]["tick"] for row in flow]
    assert flow_ticks == [1_000_000, 1_004_000, 1_004_000, 6_008_000, 6_012_000]
    # the recorder's own chain for the same round, same ticks up to the
    # store landing; the flow ends one hop later, in the unit's memory
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
    intake, at its assembler and into Buffer 0 (hops 1 to 3), and one
    unit-memory copy per window of the six rounds it reads (hop 5, nine
    windows of six): 30 x 3 + 54 = 144 copied rounds. The nine window
    input holds reference 54 rounds where they sit; each hold is
    registered, transferred to its request and released, so 27 hold
    events. The links carry 86 moves.
    """
    machine, _result, _document = traced
    counts = machine.observation.data_movement.json_value()

    assert counts["rounds"] == 30
    assert counts["copied_rounds"] == 144
    assert counts["referenced_rounds"] == 54
    assert counts["hold_events"] == 27
    assert counts["moves"] == 86
    assert counts["copies"] == 99
    by_path = counts["copies_by_path"]
    assert by_path["readout -> controller intake"]["rounds"] == 30
    assert by_path["controller intake -> controller assembler"]["rounds"] == 30
    assert by_path["controller assembler -> Buffer 0"]["rounds"] == 30
    assert by_path["Buffer 0 -> unit default#0 memory"]["rounds"] == 54


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
    flow = _flow_of(document, "window 1:0")
    lanes = []
    for row in flow:
        lanes.append((row["ph"], threads[row["tid"]], row["args"]["tick"]))

    assert lanes == [
        ("s", "Window planner", 6_008_000),
        ("t", "weak_buffer_to_weak_decoder", 6_008_000),
        ("t", "Decoder unit default#0", 6_012_000),
        ("t", "decoder_to_decoder", 6_104_000),
        ("t", "weak_decoder_to_frame", 6_104_000),
        ("f", "Frame", 6_108_000),
    ]
    for row in flow:
        assert row["cat"] == "window"
        assert row["name"] == "W0"


def test_every_window_is_ready_when_its_last_round_is_readable(traced):
    """One W ready instant per window, at that round's own data_ready.

    The moment is the input gate's: the window's data is complete when
    the last round it reads is published in the store, which is the tick
    the store residence records as data_ready for that round.
    """
    machine, _result, document = traced
    ready = []
    for row in _by_phase(document, "i"):
        if row["name"].endswith(" ready"):
            ready.append(row)
    windows = machine.observation.windows.windows

    assert len(ready) == len(windows)
    assert len(ready) == 9
    for row in ready:
        read_rounds = row["args"]["rounds"]
        first_and_last = read_rounds.split("..")
        last_round = first_and_last[1]
        residence = _one(document, "X", f"round {last_round}")
        assert row["args"]["tick"] == residence["args"]["data_ready"]


def test_the_unit_memory_counter_peaks_at_the_memorys_high_water_mark(traced):
    """The C track of the unit's memory is the memory's own occupancy."""
    machine, _result, document = traced
    (unit,) = machine.decoder_manager.pool.units()
    name = f"{unit.memory.name} rounds"
    values = []
    for row in _by_phase(document, "C"):
        if row["name"] == name:
            values.append(row["args"]["rounds"])

    assert values
    assert max(values) == unit.memory.statistics.peak_occupied_rounds
    assert values[-1] == unit.memory.occupied_rounds


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
    steps = []
    for row in _by_phase(document, "C"):
        if row["name"] == "controller assembler rounds":
            steps.append(row["args"]["rounds"])
    assert max(steps) == 1
    assert steps[-1] == 0


def _check_row(row) -> None:
    """One event of the trace against the fields its phase declares."""
    assert row["ph"] in PHASES, row
    assert row["pid"] == 1
    assert isinstance(row["tid"], int)
    if row["ph"] == "M":
        assert row["name"] in _METADATA_NAMES
        return
    assert isinstance(row["ts"], float)
    assert isinstance(row["args"]["tick"], int)
    assert row["args"]["tick"] / 1_000_000 == row["ts"]
    if row["ph"] == "X":
        assert isinstance(row["dur"], float)
        assert row["dur"] >= 0
    if row["ph"] == "i":
        assert row["s"] == "t"
    if row["ph"] in ("s", "t", "f"):
        assert isinstance(row["id"], str)
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


def _timing_only_row(latency_model=None):
    """A DECODERS row on the shipped timing-only decoder."""
    del latency_model
    return decoders.PresetLatencyDecoder(1.0)


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
    weak_decoder = dataclasses.replace(point.weak_decoder, kind="timing_only")
    point = dataclasses.replace(point, weak_decoder=weak_decoder)
    decoder_settings.DECODERS["timing_only"] = _timing_only_row
    try:
        machine = machine_module.Machine.build(point, SEED)
        result = machine.run()
    finally:
        del decoder_settings.DECODERS["timing_only"]
    machine.observation.trace_writer.write(str(trace_path))
    text = trace_path.read_text()
    document = json.loads(text)
    corrections = [row for row in document if _is_a_correction(row)]

    assert result.terminal_status == "complete"
    assert corrections
    for row in corrections:
        assert "committed" in row["args"]
        assert "observables" not in row["args"]
