"""The timeline figure, drawn from a shot's Chrome trace file alone.

The figure's rule: plots never build machines; the timeline reads a
trace file. Gate
point 1 is the shot: every span the figure draws is compared against the
same run's own listeners and its RunResult, so the file is proven to
carry the figure.
"""

import dataclasses
import json

import matplotlib
import matplotlib.pyplot
import pytest

import decsim.experiments.experiment as experiment
import decsim.experiments.plots as plots
import decsim.experiments.trace_file as trace_file
import decsim.machine as machine_module
import decsim.records.identity as identity_records
import tests.experiments.test_measure as measure_tests
import tests.experiments.yaml_configs as yaml_configs
import tests.observe.gate_point as gate_point

matplotlib.use("Agg")


TICKS_PER_MICROSECOND = 1000000


def _traced_run(trace_path):
    settings = gate_point.settings(trace=str(trace_path))
    machine = machine_module.Machine.build(settings, gate_point.SEED)
    result = machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    return machine, result


def _microseconds(ticks):
    return ticks / TICKS_PER_MICROSECOND


def _ledger_windows(ledger: dict) -> dict:
    """Window key -> (commit lo, commit hi, read hi, dispatch us).

    A key is the operation as the trace writes it and the index.
    """
    windows = {}
    for (operation_id, window_id), window in ledger.items():
        dispatch = _microseconds(window.t_dispatch)
        window_key = (str(operation_id), window_id)
        windows[window_key] = (
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
            dispatch,
        )
    return windows


def _drawn_windows(shot, expected: dict) -> dict:
    """The same fields of the windows the figure drew, for those keys."""
    windows = {}
    for window_key in expected:
        drawn = shot.windows[window_key]
        windows[window_key] = (
            drawn.commit_lo,
            drawn.commit_hi,
            drawn.read_hi,
            drawn.dispatch_us,
        )
    return windows


def _recorded_stages(machine) -> list:
    """(window key, stage), start us, end us: one row per stage record.

    Sorted, since two decodes of one window may record in either order.
    """
    stages = machine.observation.stages
    rows = []
    for operation_id, window_id in machine.observation.windows.windows:
        window_key = (str(operation_id), window_id)
        for record in stages.records_for(operation_id, window_id):
            start = _microseconds(record.start_ticks)
            end = _microseconds(record.end_ticks)
            rows.append(((window_key, record.stage), start, end))
    return sorted(rows)


def _committed_frame_spans(committed) -> list:
    """Window key, accepted us, committed us: one row per frame record."""
    rows = []
    for record in committed:
        operation_id, window_id = record.window_key
        window_key = (str(operation_id), window_id)
        start = _microseconds(record.accepted_ticks)
        end = _microseconds(record.committed_ticks)
        rows.append((window_key, start, end))
    return rows


def _transfer_spans(transfers) -> tuple:
    """(by round, by window): (path, key), sent us, delivered us per row.

    A key is the operation as the trace writes it, and the round or the
    window index.
    """
    by_round = []
    by_window = []
    for transfer in transfers:
        path = transfer["path"]
        attribution = transfer["attribution"]
        window_id = attribution["window_id"]
        start = _microseconds(transfer["send_ticks"])
        end = _microseconds(transfer["delivery_ticks"])
        if window_id is None:
            (rounds,) = attribution["rounds_by_operation"]
            operation = _operation_text(rounds["operation_id"])
            round_key = (operation, rounds["round_lo"])
            by_round.append(((path, round_key), start, end))
        else:
            operation = _operation_text(attribution["operation_id"])
            window_key = (operation, window_id)
            by_window.append(((path, window_key), start, end))
    return by_round, by_window


def _operation_text(recorded: dict) -> str:
    """A recorded operation identity as the trace writes it."""
    operation_id = identity_records.stable_identity_from_json(recorded)
    return str(operation_id)


def _drawn_spans(drawn_by_key: dict, expected: list) -> list:
    """The figure's start and end us under each expected row's key.

    One row per expected row, duplicates kept, so every record is held
    against the one span the figure draws for its key.
    """
    rows = []
    for key, _start, _end in expected:
        drawn = drawn_by_key[key]
        rows.append((key, drawn.start_us, drawn.end_us))
    return rows


def _traced_switching_run(tmp_path, trace_path):
    """One switching shot of nine rounds at d 3, traced.

    Switching keeps the lookahead terminal policy, so the last window
    commits rounds 7 to 9 and names rounds 10 to 12 in its buffer, rounds
    the stream never has.
    """
    workload = yaml_configs.memory_workload(9)
    sweep_point = {
        "axes": {
            "workload.arguments.physical_error_probability": [0.008],
            "qpu.distance": [3],
            "qpu.round_period_microseconds": [1.0],
        },
        "collection": {"max_shots": 1},
    }
    card = {
        "escalation": {"kind": "switching", "gap_threshold_db": 20.0},
        "workload": workload,
        **yaml_configs.strong_unit("belief_matching"),
        "sweep": [sweep_point],
    }
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    shipped = point.settings
    observation = dataclasses.replace(
        shipped.observation, trace=str(trace_path)
    )
    settings = dataclasses.replace(shipped, observation=observation)
    machine = machine_module.Machine.build(settings, gate_point.SEED)
    result = machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    return machine, result


def test_the_timeline_windows_are_the_runs_own_windows(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    ledger = machine.observation.windows.windows
    expected = _ledger_windows(ledger)
    drawn = _drawn_windows(shot, expected)
    assert drawn == expected


def test_the_timeline_stages_are_the_runs_own_stage_records(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    expected = _recorded_stages(machine)
    drawn = _every_drawn_span(shot.stages)
    assert drawn == expected


def test_the_timeline_keeps_every_decode_of_a_window_decoded_again(tmp_path):
    """Each forced-class solve and the speculative decode keep their stages.

    Under run_both_at_once a window's weak solves and its speculative
    strong decode all run, so one window and stage name has several
    spans, and the figure holds every one the run recorded.
    """
    shot_run, shot = _traced_speculative_run(tmp_path)
    expected = _recorded_stages(shot_run.machine)

    drawn = _every_drawn_span(shot.stages)

    assert len(expected) > len(shot.stages)
    assert drawn == expected


def test_the_timeline_keeps_every_move_of_a_window_moved_twice(tmp_path):
    """Both boundary handoffs of a window decoded twice keep their bars.

    Under run_both_at_once a window's two decodes each hand a boundary
    on, so one link and window has two moves, and the figure holds every
    transfer the run recorded.
    """
    shot_run, shot = _traced_speculative_run(tmp_path)
    transfers = shot_run.result.link_traffic["transfers"]
    _by_round, by_window = _transfer_spans(transfers)
    expected = sorted(by_window)

    drawn = _every_drawn_span(shot.moves_by_window)

    assert len(expected) > len(shot.moves_by_window)
    assert drawn == expected


def _traced_speculative_run(tmp_path) -> tuple:
    """A switching shot under run_both_at_once, traced; (run, timeline)."""
    trace_path = tmp_path / "speculative.trace.json"
    observation = {"trace": str(trace_path)}
    shot_run = measure_tests.switching_run(
        tmp_path,
        20.0,
        run_both_at_once=True,
        sections={"observation": observation},
    )
    shot_run.machine.observation.trace_writer.write(str(trace_path))
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    return shot_run, shot


def _every_drawn_span(spans_by_key: dict) -> list:
    """(key, start us, end us) for every span the figure draws, sorted."""
    rows = []
    for key, spans in spans_by_key.items():
        for span in spans:
            rows.append((key, span.start_us, span.end_us))
    return sorted(rows)


def test_the_timeline_frame_bars_are_the_frames_own_corrections(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    committed = machine.observation.frame_corrections.committed
    expected = _committed_frame_spans(committed)
    drawn = _drawn_spans(shot.frame, expected)
    assert len(committed) == len(shot.frame)
    assert drawn == expected


def test_the_timeline_moves_are_the_results_own_transfers(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    _machine, result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    transfers = result.link_traffic["transfers"]
    by_round, by_window = _transfer_spans(transfers)
    drawn_by_round = _drawn_spans(shot.moves_by_round, by_round)
    drawn_by_window = _every_drawn_span(shot.moves_by_window)
    assert drawn_by_round == by_round
    assert drawn_by_window == sorted(by_window)


def test_the_timeline_reads_the_lanes_and_the_period_off_the_file(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    _machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    lanes = plots._timeline_lanes(document)
    shot = plots.timeline_shot(document)
    assert lanes.store_path == "controller_to_weak_buffer"
    assert lanes.store_name == "weak syndrome buffer"
    assert lanes.input_path == "weak_buffer_to_weak_decoder"
    assert lanes.output_path == "weak_decoder_to_frame"
    assert shot.round_period_microseconds == 1.0


def test_the_timeline_figure_is_written_from_the_file(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    _traced_run(trace_path)
    figure_path = tmp_path / "timeline.png"
    plots.timeline_plot(trace_path, figure_path)
    status = figure_path.stat()
    assert status.st_size > 0


@pytest.mark.parametrize(
    ("escalation_kind", "title"),
    [
        ("weak_baseline", "Weak baseline path timeline"),
        ("a_row_no_table_names", "A row no table names path timeline"),
    ],
)
def test_the_timeline_is_titled_by_the_escalation_kind_it_recorded(
    tmp_path, monkeypatch, escalation_kind, title
):
    """The title reads the trace's recorded kind, so any row has one."""
    trace_path = tmp_path / "point1.trace.json"
    _traced_run(trace_path)
    _rename_the_escalation(trace_path, escalation_kind)
    figure_path = tmp_path / "timeline.png"
    drawn = []
    monkeypatch.setattr(matplotlib.pyplot, "close", drawn.append)

    plots.timeline_plot(trace_path, figure_path)

    (figure,) = drawn
    (axis,) = figure.axes
    assert axis.get_title() == title


def test_a_last_window_reading_past_the_stream_is_drawn_to_the_last_round(
    tmp_path,
):
    """The figure reads to the last stored round, as the decode did."""
    trace_path = tmp_path / "switching.trace.json"
    _traced_switching_run(tmp_path, trace_path)
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    lanes = plots._timeline_lanes(document)
    stored = plots._stored_rounds(lanes, shot)
    last_window = shot.windows[max(shot.windows)]
    _operation, last_stored_round = max(stored)
    assert last_window.read_hi == last_stored_round
    figure_path = tmp_path / "timeline.png"
    plots.timeline_plot(trace_path, figure_path)
    status = figure_path.stat()
    assert status.st_size > 0


def test_two_streams_draw_every_window_of_both(tmp_path):
    """Two streams each have windows 0 to 8, and the figure keeps all 18.

    A window is its operation and its index (records/windows.py
    Window.key), the "op:index" the trace writes, so the figure files
    each stream's window 3 apart and draws the stages and the frame
    write its own decode made.
    """
    trace_path = tmp_path / "streams.trace.json"
    machine = _traced_streams_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots.timeline_shot(document)
    ledger = machine.observation.windows.windows
    expected_windows = _ledger_windows(ledger)
    expected_stages = _recorded_stages(machine)
    committed = machine.observation.frame_corrections.committed
    expected_frame = _committed_frame_spans(committed)

    assert len(expected_windows) == 18
    assert _drawn_windows(shot, expected_windows) == expected_windows
    assert _every_drawn_span(shot.stages) == expected_stages
    assert len(shot.frame) == 18
    assert _drawn_spans(shot.frame, expected_frame) == expected_frame
    figure_path = tmp_path / "timeline.png"
    plots.timeline_plot(trace_path, figure_path)
    status = figure_path.stat()
    assert status.st_size > 0


def _traced_streams_run(trace_path):
    """The two side-by-side streams of test_measure, traced."""
    shipped = measure_tests.seam_streams_settings(2)
    observation = dataclasses.replace(
        shipped.observation, trace=str(trace_path)
    )
    settings = dataclasses.replace(shipped, observation=observation)
    machine = machine_module.Machine.build(settings, gate_point.SEED)
    machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    return machine


def test_a_run_folder_without_a_trace_has_no_file_to_draw_from(tmp_path):
    assert plots.first_trace_file(tmp_path) is None


def test_the_figures_shot_is_the_first_points_lowest_traced_seed(tmp_path):
    """The sweep's order, from the manifest, not the ids' name order.

    Point ids are hashes, so the first point's file can sort last.
    """
    manifest = {"points": ["ffff", "0000"]}
    manifest_path = tmp_path / "manifest.json"
    manifest_text = json.dumps(manifest)
    manifest_path.write_text(manifest_text)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    second_point = trace_dir / "0000_seed0.trace.json"
    second_point.write_text("[]")
    later_seed = trace_dir / "ffff_seed10.trace.json"
    later_seed.write_text("[]")
    first = trace_dir / "ffff_seed2.trace.json"
    first.write_text("[]")

    assert plots.first_trace_file(tmp_path) == first


def _rename_the_escalation(trace_path, escalation_kind: str) -> None:
    """The trace's process name with its escalation kind replaced.

    The file is one JSON array of events (observe/trace_writer.py).
    """
    trace_text = trace_path.read_text()
    events = json.loads(trace_text)
    for event in events:
        if event["name"] == "process_name":
            words = event["args"]["name"].split()
            words[1] = escalation_kind
            event["args"]["name"] = " ".join(words)
    renamed_text = json.dumps(events)
    trace_path.write_text(renamed_text)
