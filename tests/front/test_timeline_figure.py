"""The timeline figure, drawn from a shot's Chrome trace file alone.

The figure's source is docs/rewrite/notes/slice_11_front.md section 10
("plots never build machines; the timeline reads a trace file"). Gate
point 1 is the shot: every span the figure draws is compared against the
same run's own listeners and its RunResult, so the file is proven to
carry the figure.
"""

import matplotlib

import decsim.front.plots as plots
import decsim.front.trace_file as trace_file
import decsim.machine as machine_module
import tests.observe.gate_point as gate_point

matplotlib.use("Agg")

pytestmark = gate_point.needs_the_frozen_suite

TICKS_PER_MICROSECOND = 1000000


def _traced_run(trace_path):
    settings = gate_point.settings(trace=str(trace_path))
    machine = machine_module.Machine.build(settings, gate_point.SEED)
    result = machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    return machine, result


def _microseconds(ticks):
    return ticks / TICKS_PER_MICROSECOND


def test_the_timeline_windows_are_the_runs_own_windows(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots._timeline_shot(document)
    ledger = machine.observation.windows.windows
    for (_op_id, window_id), window in ledger.items():
        drawn = shot.windows[window_id]
        assert drawn.commit_lo == window.commit_lo
        assert drawn.commit_hi == window.commit_hi
        assert drawn.read_hi == window.buffer_hi
        assert drawn.dispatch_us == _microseconds(window.t_dispatch)


def test_the_timeline_stages_are_the_runs_own_stage_records(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots._timeline_shot(document)
    stages = machine.observation.stages
    for op_id, window_id in machine.observation.windows.windows:
        for record in stages.records_for(op_id, window_id):
            drawn = shot.stages[(window_id, record.stage)]
            assert drawn.start_us == _microseconds(record.start_ticks)
            assert drawn.end_us == _microseconds(record.end_ticks)


def test_the_timeline_frame_bars_are_the_frames_own_corrections(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots._timeline_shot(document)
    committed = machine.observation.frame_corrections.committed
    assert len(committed) == len(shot.frame)
    for record in committed:
        window_id = record.window_key[1]
        drawn = shot.frame[window_id]
        assert drawn.start_us == _microseconds(record.accepted_ticks)
        assert drawn.end_us == _microseconds(record.committed_ticks)


def test_the_timeline_moves_are_the_results_own_transfers(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    _machine, result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    shot = plots._timeline_shot(document)
    for transfer in result.link_traffic["transfers"]:
        path = transfer["path"]
        window_id = transfer["attribution"]["window_id"]
        key = (path, window_id)
        if window_id is None:
            key = (path, transfer["attribution"]["round_lo"])
            drawn = shot.moves_by_round[key]
        else:
            drawn = shot.moves_by_window[key]
        assert drawn.start_us == _microseconds(transfer["send_ticks"])
        assert drawn.end_us == _microseconds(transfer["delivery_ticks"])


def test_the_timeline_reads_the_lanes_and_the_period_off_the_file(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    _machine, _result = _traced_run(trace_path)
    document = trace_file.load(trace_path)
    lanes = plots._timeline_lanes(document)
    shot = plots._timeline_shot(document)
    assert lanes.store_path == "controller_to_weak_buffer"
    assert lanes.store_name == "buffer 0"
    assert lanes.input_path == "weak_buffer_to_weak_decoder"
    assert lanes.output_path == "weak_decoder_to_frame"
    assert shot.round_period_us == 1.0


def test_the_timeline_figure_is_written_from_the_file(tmp_path):
    trace_path = tmp_path / "point1.trace.json"
    _traced_run(trace_path)
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
