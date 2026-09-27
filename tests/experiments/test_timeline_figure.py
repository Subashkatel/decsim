"""The timeline figure, drawn from a shot's Chrome trace file alone.

The figure's rule: plots never build machines; the timeline reads a
trace file. Gate
point 1 is the shot: every span the figure draws is compared against the
same run's own listeners and its RunResult, so the file is proven to
carry the figure.
"""

import dataclasses

import matplotlib

import decsim.experiments.experiment as experiment
import decsim.experiments.plots as plots
import decsim.experiments.trace_file as trace_file
import decsim.machine as machine_module
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
    """Window id -> (commit lo, commit hi, read hi, dispatch us)."""
    windows = {}
    for (_operation_id, window_id), window in ledger.items():
        dispatch = _microseconds(window.t_dispatch)
        windows[window_id] = (
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
            dispatch,
        )
    return windows


def _drawn_windows(shot, expected: dict) -> dict:
    """The same fields of the windows the figure drew, for those ids."""
    windows = {}
    for window_id in expected:
        drawn = shot.windows[window_id]
        windows[window_id] = (
            drawn.commit_lo,
            drawn.commit_hi,
            drawn.read_hi,
            drawn.dispatch_us,
        )
    return windows


def _recorded_stages(machine) -> list:
    """(window id, stage), start us, end us: one row per stage record."""
    stages = machine.observation.stages
    rows = []
    for operation_id, window_id in machine.observation.windows.windows:
        for record in stages.records_for(operation_id, window_id):
            start = _microseconds(record.start_ticks)
            end = _microseconds(record.end_ticks)
            rows.append(((window_id, record.stage), start, end))
    return rows


def _committed_frame_spans(committed) -> list:
    """Window id, accepted us, committed us: one row per frame record."""
    rows = []
    for record in committed:
        window_id = record.window_key[1]
        start = _microseconds(record.accepted_ticks)
        end = _microseconds(record.committed_ticks)
        rows.append((window_id, start, end))
    return rows


def _transfer_spans(transfers) -> tuple:
    """(by round, by window): (path, key), sent us, delivered us per row."""
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
            by_round.append(((path, rounds["round_lo"]), start, end))
        else:
            by_window.append(((path, window_id), start, end))
    return by_round, by_window


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
        "shots": 1,
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
        1,
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
    drawn = _drawn_spans(shot.stages, expected)
    assert drawn == expected


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
    drawn_by_window = _drawn_spans(shot.moves_by_window, by_window)
    assert drawn_by_round == by_round
    assert drawn_by_window == by_window


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
    assert last_window.read_hi == max(stored)
    figure_path = tmp_path / "timeline.png"
    plots.timeline_plot(trace_path, figure_path)
    status = figure_path.stat()
    assert status.st_size > 0


def test_a_run_folder_without_a_trace_has_no_file_to_draw_from(tmp_path):
    assert plots.first_trace_file(tmp_path) is None


def test_the_first_traced_shot_of_a_run_folder_is_the_figures_shot(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    second = trace_dir / "p0.005_d5_seed0.trace.json"
    second.write_text("[]")
    first = trace_dir / "p0.003_d3_seed0.trace.json"
    first.write_text("[]")
    assert plots.first_trace_file(tmp_path) == first
