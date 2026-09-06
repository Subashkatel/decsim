"""The Chrome trace of gate point 1: its shape, its chains, its counts.

The event model is docs/rewrite/notes/trace_and_viewer.md section 2 and
its worked example (weak_decoder_baseline d 3 p 0.003 seed 0, the frozen
suite's first strict point); the copy, reference and move counts are
data_path.md's hop table. The trace never moves a tick: a run with it on
narrates the same log and returns the same results as one with it off.
"""

import dataclasses
import hashlib
import json
import pathlib

import pytest

import decsim.machine as machine_module

SUITE = pathlib.Path(
    "/scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/validation/"
    "responsibility_audit_2026_08_30/frozen_suite"
)
POINT = {
    "physical_error_probability": 0.003,
    "distance": 3,
    "round_period_us": 1.0,
}
SEED = 0
# the golden log hash of the point, from the frozen suite
POINT_LOG_SHA256 = "74c2e7aee37a"

PHASES = ("M", "X", "i", "C", "s", "t", "f")


def _settings(trace_path=None, data_movement=False):
    from experiments.experiment_config import load_experiment

    config_path = SUITE / "weak_decoder_baseline.yaml"
    config = load_experiment(config_path)
    settings = config.point_settings(**POINT)
    if trace_path is None:
        trace = "off"
    else:
        trace = str(trace_path)
    observation = dataclasses.replace(
        settings.observation, trace=trace, data_movement=data_movement
    )
    return dataclasses.replace(settings, observation=observation)


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
    document = json.loads(path.read_text())
    return machine, result, document


def _log_sha(machine) -> str:
    text = "\n".join(machine.observation.log.lines)
    return hashlib.sha256(text.encode()).hexdigest()


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
    plain, plain_result = _run()
    traced, traced_result = _run(tmp_path / "on.trace.json")

    assert _log_sha(plain) == _log_sha(traced)
    assert _log_sha(plain).startswith(POINT_LOG_SHA256)
    assert plain.engine.now == traced.engine.now
    assert plain_result.link_traffic == traced_result.link_traffic
    assert plain_result.operation_results == traced_result.operation_results
    assert (
        plain.pauli_frame.snapshot().records
        == traced.pauli_frame.snapshot().records
    )
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
    named_tids = {}

    for row in document:
        assert row["ph"] in PHASES, row
        assert row["pid"] == 1
        assert isinstance(row["tid"], int)
        if row["ph"] == "M":
            assert row["name"] in (
                "process_name",
                "thread_name",
                "thread_sort_index",
            )
            if row["name"] == "thread_name":
                named_tids[row["tid"]] = row["args"]["name"]
            continue
        assert isinstance(row["ts"], float)
        assert isinstance(row["args"]["tick"], int)
        assert row["args"]["tick"] / 1_000_000 == row["ts"]
        tids_with_events.add(row["tid"])
        if row["ph"] == "X":
            assert isinstance(row["dur"], float)
            assert row["dur"] >= 0
        if row["ph"] == "i":
            assert row["s"] == "t"
        if row["ph"] in ("s", "t", "f"):
            assert isinstance(row["id"], str)
        if row["ph"] == "f":
            assert row["bp"] == "e"

    assert tids_with_events <= set(named_tids)


def test_every_flow_event_lies_inside_a_complete_event_on_its_thread(traced):
    """A flow's binding point is its enclosing slice, so it must have one."""
    _machine, _result, document = traced
    spans_by_tid = {}
    for row in _by_phase(document, "X"):
        start = row["args"]["tick"]
        end = start + int(round(row["dur"] * 1_000_000))
        spans_by_tid.setdefault(row["tid"], []).append((start, end))

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

    residences = [
        row
        for row in _by_phase(document, "X")
        if row["cat"] == "round,residence" and row["args"]["round"] == "1:1"
    ]
    (residence,) = residences
    assert residence["args"]["tick"] == 1_004_000
    assert residence["dur"] == 5.008
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
    for event in machine.round_events.events:
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
