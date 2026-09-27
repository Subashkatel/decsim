"""The experiment figures.

timeline.png          one traced shot, read from its Chrome trace file:
                      every stage of every window on its own row, in
                      real time, the weak baseline's figure style
                      (commit reads solid, buffer reads lighter,
                      geometry in the subtitle). `decsim run --trace`,
                      or `trace: chrome` in the observation section,
                      writes the file this reads, and `decsim collect`
                      draws it for the first traced shot
ler.png               logical error rate against a swept setting, Wilson
                      95% bars, a curve per run folder and per value of
                      another, from each run's sweep.csv
latency_combined.png  decode wall clock per window against a swept
                      setting, violins, from each run's
                      latency_samples.csv
stage_breakdown.png   where a window's time goes, a stacked bar per
                      value of a swept setting, from shots.csv
data_movement.png     bits copied and moved per shot, by the memory
                      class the hop crosses, against a swept setting, a
                      panel per run folder, from data_movement.csv

Every figure but the timeline reads its points by their metadata, the
values the sweep set, each named by its yaml path, as sinter's plot
reads a point's json_metadata through --x_func, --group_func and
--filter_func and names no axis itself
(sinter/_command/_main_plot.py:21-35): `decsim plot --x qpu.distance
--where workload.arguments.physical_error_probability=0.001`. Every
time is in microseconds.
"""

import csv
import dataclasses
import json
import math
import pathlib
import statistics
from collections.abc import Mapping
from typing import Optional

import decsim.config as config_module
import decsim.escalation.settings as escalation_settings
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.experiments.trace_file as trace_file
import decsim.tables as tables

WINDOW_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
)
MAX_LEGEND_WINDOWS = 8
TIMELINE_TITLES = {
    "weak_baseline": "Weak only path timeline",
    "strong_only": "Strong only path timeline",
    "switching": "Switching path timeline",
}
BREAKDOWN_TITLES = {
    "pymatching": "Time breakdown: Weak decoder (pymatching)",
    "belief_matching": "Time breakdown: Strong decoder (belief matching)",
}
# The measured window chain from syndrome arrival to frame commit, in
# pipeline order; each name is a per-shot mean column of shots.csv.
STAGE_BREAKDOWN_STAGES = (
    ("buffer_fill_mean_us", "buffer fill"),
    ("queue_wait_mean_us", "queue wait"),
    ("input_link_per_window_mean_us", "input link"),
    ("fetch_mean_us", "fetch"),
    ("algorithm_mean_us", "algorithm"),
    ("release_mean_us", "release"),
    ("dd_per_window_mean_us", "boundary handoff"),
    ("output_link_per_window_mean_us", "output link"),
    ("frame_commit_mean_us", "frame commit"),
)
# every figure `decsim plot` draws, and the file each one writes
FIGURES = {
    "timeline": "timeline.png",
    "stage_breakdown": "stage_breakdown.png",
    "latency": "latency_combined.png",
    "ler": "ler.png",
    "data_movement": "data_movement.png",
}
# the two data-movement quantities that have bits, and the line each one
# is drawn with; a reference books the rounds a hold keeps where they
# are and copies no bits at all (observe/data_movement.py), so it has no
# series on an axis of bits
MOVEMENT_SERIES = (
    ("copy_bits_per_shot", "copied", "o", "-"),
    ("move_bits_per_shot", "moved", "s", "--"),
)
# where collect_command leaves the traces of the shots it traced
TRACE_DIR = "trace"
# an axis whose values span this ratio or more is drawn logarithmic, as
# an error rate's decade is; a narrower one, a distance's, is linear
LOG_AXIS_SPAN = 10.0


@dataclasses.dataclass(frozen=True)
class Selection:
    """Which points a figure draws and how: `decsim plot`'s three options.

    x_path is the swept setting on the x axis, group_path the one a
    curve is drawn per value of, and where the values the kept points'
    metadata holds; each names a setting by its yaml path, sinter's
    --x_func, --group_func and --filter_func on json_metadata
    (sinter/_command/_main_plot.py:21-35).
    """

    x_path: Optional[str] = None
    group_path: Optional[str] = None
    where: Mapping = dataclasses.field(default_factory=dict)


def timeline_plot(trace_path, path: pathlib.Path) -> None:
    """One traced shot's hops and stages, in the time they happened.

    Drawn from that shot's Chrome trace file alone (`decsim run --trace`,
    or `trace: chrome` in the yaml's observation section): the trace
    records where every round and window sat and for how long, so this
    builds no machine and runs nothing.
    """
    import matplotlib.pyplot as plt

    document = trace_file.load(trace_path)
    shot = timeline_shot(document)
    lanes = _timeline_lanes(document)
    rows = _timeline_rows(lanes, shot)
    lane_count = max(len(shot.windows), 3)
    height = 0.45 * len(rows) + 1.6
    figure, axis = plt.subplots(figsize=(11, height))
    timeline = _TimelineAxes(axis, rows, lane_count)
    _draw_rounds(timeline, lanes, shot)
    window_ranges = _draw_windows(timeline, lanes, shot)
    _label_timeline(timeline, document, shot)
    _add_timeline_legend(timeline, shot.windows, window_ranges)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def timeline_shot(document) -> "_TimelineShot":
    """Every span the timeline draws, indexed by round, window and stage.

    What timeline_plot reads off one trace document before it draws.
    """
    moves_by_round = {}
    moves_by_window = {}
    for event in document.of_phase("X"):
        _index_move(event, moves_by_round, moves_by_window)
    windows = _timeline_windows(document)
    stages = _timeline_stages(document)
    frame = _frame_spans(document)
    round_period_microseconds = _round_period_microseconds(moves_by_round)
    return _TimelineShot(
        round_period_microseconds=round_period_microseconds,
        moves_by_round=moves_by_round,
        moves_by_window=moves_by_window,
        windows=windows,
        stages=stages,
        frame=frame,
    )


def first_trace_file(run_dir) -> Optional[pathlib.Path]:
    """The first trace file of a run folder, None when nothing traced.

    A sweep writes one file per traced shot under trace/, named by the
    shot's point id and seed (experiments/measure.py shot_label); the
    first in name order is the shot the timeline draws.
    """
    trace_dir = pathlib.Path(run_dir) / TRACE_DIR
    if not trace_dir.is_dir():
        return None
    entries = trace_dir.iterdir()
    found = sorted(entries)
    for path in found:
        if path.is_file():
            return path
    return None


def ler_plot(run_dirs: list, selection: Selection, path: pathlib.Path) -> None:
    """The logical error rate against the x setting, from each sweep.csv.

    One curve per run folder and per value of the group setting, the
    points the where values keep. Measured points carry Wilson 95%
    bars; a zero-failure point cannot sit on a log axis, so its curve
    simply leaves it out. The x axis is logarithmic when its values span
    a decade or more (an error rate's) and linear otherwise (a
    distance's).

        decsim plot <run_dir>... --figure ler --x <path> [--group <path>]
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    swept = []
    for run_dir in run_dirs:
        rows = _sweep_rows(run_dir, selection)
        tier_label = _run_tier_label(run_dir, rows[0]["algorithm"])
        values = _draw_ler_curves(axis, rows, selection, tier_label)
        swept.extend(values)
    _scale_x_axis(axis, swept)
    axis.set_yscale("log")
    axis.set_xlabel(selection.x_path)
    axis.set_ylabel("Logical error rate per shot")
    title = _titled("Logical error rate", selection)
    axis.set_title(title, fontsize=9)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def memory_class_series(run_dir, selection: Selection) -> dict:
    """One run's bits per shot by memory class, then by copied and moved.

    memory class -> word -> (x values, bits), read from that folder's
    data_movement.csv memory-class rows. A value whose bits are zero is
    left out: an off-board hop of this machine moves and never copies,
    and zero has no place on a log axis.
    """
    rows = _memory_class_rows(run_dir, selection)
    series = {}
    for memory_class in _classes_in_order(rows):
        at_class = _rows_of_class(rows, memory_class)
        by_word = {}
        for column, word, _marker, _style in MOVEMENT_SERIES:
            by_word[word] = _series_of(at_class, column)
        series[memory_class] = by_word
    return series


def data_movement_plot(
    run_dirs: list, selection: Selection, path: pathlib.Path
) -> None:
    """Bits copied and moved per shot by memory class, against x.

    One panel per study config, read from each run folder's
    data_movement.csv memory-class rows. The classes are kept apart
    rather than summed because the classical sources make the class the
    cost: a DRAM access is "a couple of orders-of-magnitude higher than
    the cost of an internal cache access" (Horowitz, ISSCC 2014 lines
    232-247) and an accelerator's access costs what the memory it reads
    costs (Dally, CACM 2020 lines 231-234).

        decsim plot <run_dir>... --figure data_movement --x <path>
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = len(run_dirs)
    width = 4.8 * panels
    figure, axes = plt.subplots(
        1, panels, figsize=(width, 3.6), sharey=True, squeeze=False
    )
    for panel_index, run_dir in enumerate(run_dirs):
        axis = axes[0][panel_index]
        series = memory_class_series(run_dir, selection)
        _draw_movement_panel(axis, series, selection.x_path)
        title = _study_config_name(run_dir)
        axis.set_title(title, fontsize=9)
    first_axis = axes[0][0]
    first_axis.set_ylabel("Bits per shot")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def stage_breakdown_plot(
    run_dir, selection: Selection, path: pathlib.Path
) -> None:
    """One stacked bar per x value: where a window's time goes.

    From syndrome arrival in the buffer to the Pauli-frame commit.

        decsim plot <run_dir> --figure stage_breakdown --x <path>
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _shot_rows(run_dir, selection)
    medians_by_value = _median_stage_us_by_value(rows, selection.x_path)
    values = list(medians_by_value)
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    bar_positions = range(len(values))
    stacked_left = _draw_stage_bars(axis, medians_by_value, values)
    _label_stage_totals(axis, bar_positions, stacked_left)
    axis.set_yticks(list(bar_positions))
    tick_labels = []
    for value in values:
        tick_labels.append(f"{selection.x_path}={value}")
    axis.set_yticklabels(tick_labels)
    axis.invert_yaxis()
    widest = max(stacked_left)
    right_edge = widest * 1.12
    axis.set_xlim(0, right_edge)
    # every breakdown is drawn in ms so the two tiers' figures share
    # one unit; the axis stays linear with plain tick numbers
    axis.set_xlabel("median time per window (ms)")
    algorithm = rows[0]["algorithm"]
    title = _breakdown_title(algorithm)
    axis.set_title(title)
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def combined_latency_plot(
    sample_files: list, selection: Selection, path: pathlib.Path
) -> None:
    """Every run's decode wall clock per window, against x, on one axes.

    One violin per x value per run (median marked, worst window
    flagged), microsecond log axis, from each run's
    latency_samples.csv, with the window-generation deadline drawn as
    the throughput boundary: a new window every commit rounds times the
    round period, the window_period_us each row carries. The violin
    and deadline shape follows Helios 2301.08419 Fig. 7 and Google
    2408.13687 Fig. 4d; the log axis is what makes two tiers legible,
    decades apart.

        decsim plot <run_dir>... --figure latency --x <path>
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    all_log_values = []
    deadlines = {}
    for file_index, sample_file in enumerate(sample_files):
        rows = _selected_rows(sample_file, selection)
        pooled = _samples_by_value(rows, selection.x_path)
        _deadlines_by_value(deadlines, rows, selection.x_path)
        algorithm = rows[0]["algorithm"]
        color = f"C{file_index}"
        positions = list(pooled)
        _latency_violins(axis, pooled, positions, 1.4, color, algorithm)
        log_values = _log_values_of(pooled)
        all_log_values.extend(log_values)
    _deadline_line(axis, deadlines)
    _log_decade_axis(axis, all_log_values)
    axis.set_xticks(sorted(deadlines))
    axis.set_xlabel(selection.x_path)
    axis.set_ylabel("Decode wall clock per window (µs)")
    axis.set_title("Decode latency, weak and strong tiers")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plots(report_dir: pathlib.Path) -> None:
    """The figure `decsim collect` draws itself: the first traced shot.

    timeline.png needs a traced shot, so it is drawn only when the
    observation section asked for a trace. Every other figure plots a
    swept setting the run cannot guess, so `decsim plot` names it.
    """
    import matplotlib

    matplotlib.use("Agg")
    trace_path = first_trace_file(report_dir)
    if trace_path is None:
        return
    timeline_path = report_dir / "timeline.png"
    timeline_plot(trace_path, timeline_path)


def figure(
    name: str,
    run_dirs: list,
    out_path=None,
    selection: Optional[Selection] = None,
):
    """Draw one named figure from run folders, and return where it went.

    Every figure here reads files: a run folder's csv rows, or the
    Chrome trace of one of its shots. The names are the rows of
    FIGURES: what each one needs is what its run folders must hold, and
    every figure but the timeline needs the x setting.
    """
    import matplotlib

    matplotlib.use("Agg")
    if name not in FIGURES:
        listed = ", ".join(FIGURES)
        raise refusal.RefusalError(
            f"no figure named {name}; the figures are {listed}"
        )
    if selection is None:
        selection = Selection()
    first_dir = pathlib.Path(run_dirs[0])
    if out_path is None:
        out_path = first_dir / FIGURES[name]
    out_path = pathlib.Path(out_path)
    if name == "timeline":
        trace_path = _timeline_source(first_dir)
        timeline_plot(trace_path, out_path)
        return out_path
    _refuse_a_figure_without_x(name, selection)
    _draw_named_figure(name, run_dirs, out_path, selection)
    return out_path


def _card_label(algorithm) -> str:
    """A named algorithm capitalized, a latency card as its microseconds."""
    if isinstance(algorithm, str):
        return algorithm.capitalize()
    return f"{algorithm:g} µs"


def _refuse_a_figure_without_x(name: str, selection: Selection) -> None:
    """A figure against a swept setting needs that setting named."""
    if selection.x_path is not None:
        return
    raise refusal.RefusalError(
        f"the {name} figure is drawn against a swept setting; name it "
        "with --x and its yaml path, as in --x qpu.distance"
    )


def _draw_named_figure(
    name: str, run_dirs: list, out_path: pathlib.Path, selection: Selection
) -> None:
    """The one figure the name asks for, from the folders it was given."""
    first_dir = pathlib.Path(run_dirs[0])
    if name == "stage_breakdown":
        stage_breakdown_plot(first_dir, selection, out_path)
        return
    if name == "latency":
        sample_files = _sample_files(run_dirs)
        combined_latency_plot(sample_files, selection, out_path)
        return
    if name == "data_movement":
        data_movement_plot(run_dirs, selection, out_path)
        return
    ler_plot(run_dirs, selection, out_path)


def _timeline_source(run_dir: pathlib.Path) -> pathlib.Path:
    """The trace the timeline draws: a folder's first, or the file named."""
    if run_dir.is_file():
        return run_dir
    trace_path = first_trace_file(run_dir)
    if trace_path is None:
        raise refusal.RefusalError(
            f"{run_dir} has no trace/ folder, so no shot of it was traced; "
            "run it again with --trace, or `trace: chrome` in the yaml"
        )
    return trace_path


@dataclasses.dataclass(frozen=True)
class _TimelineLanes:
    """Which link each hop of the traced shot crossed, and its store.

    Read off the channels the trace names: a weak run moves windows on
    weak_buffer_to_weak_decoder, a strong-only run on
    strong_buffer_to_strong_decoder.
    """

    input_path: str
    output_path: str
    store_path: str
    store_name: str


@dataclasses.dataclass(frozen=True)
class _Span:
    """One bar of the figure: when it started and when it ended, in us."""

    start_us: float
    end_us: float


@dataclasses.dataclass(frozen=True)
class _TimelineWindow:
    """One window of the traced shot: its rounds and its two instants."""

    window_id: int
    read_lo: int
    read_hi: int
    commit_lo: int
    commit_hi: int
    dispatch_us: float


@dataclasses.dataclass(frozen=True)
class _TimelineShot:
    """Everything the timeline draws, read off one trace document."""

    round_period_microseconds: float
    moves_by_round: dict  # (channel, round number) -> _Span
    moves_by_window: dict  # (channel, window id) -> _Span
    windows: dict  # window id -> _TimelineWindow
    stages: dict  # (window id, stage name) -> _Span
    frame: dict  # window id -> _Span, accepted to committed


class _TimelineAxes:
    """The timeline's lanes: a bar for a round, and a bar for a window.

    A window's bars sit on their own sub-lane inside the row, so several
    windows share a row without overlapping.
    """

    def __init__(self, axis, rows: list, lane_count: int):
        self.axis = axis
        self.rows = tuple(rows)
        self.lane_count = lane_count
        self.row_index = _row_index(rows)

    def round_bar(self, row: str, left: float, width: float, shade) -> None:
        """One round's bar, full height, on the row's centre line."""
        lane = self.row_index[row]
        self.axis.barh(lane, width, left=left, color=shade, height=0.55)

    def window_bar(
        self,
        row: str,
        start: float,
        end: float,
        window_id: int,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        """One window's bar on its own sub-lane inside the row."""
        centre_offset = window_id - (self.lane_count - 1) / 2
        lane_pitch = 0.5 / self.lane_count
        lane = self.row_index[row] + centre_offset * lane_pitch
        width = end - start
        self.axis.barh(
            lane, width, left=start, color=color, height=0.17, alpha=alpha
        )


def _row_index(rows: list) -> dict:
    """Row label -> its position from the top of the timeline."""
    index_by_row = {}
    for index, name in enumerate(rows):
        index_by_row[name] = index
    return index_by_row


def _timeline_lanes(document) -> _TimelineLanes:
    """The links this shot moved data on, and the store it filled."""
    channels = document.channels()
    store_path = "controller_to_weak_buffer"
    store_name = "weak syndrome buffer"
    input_path = "weak_buffer_to_weak_decoder"
    output_path = "weak_decoder_to_frame"
    if store_path not in channels:
        store_path = "controller_to_strong_buffer"
        store_name = "strong syndrome buffer"
    if input_path not in channels:
        input_path = "strong_buffer_to_strong_decoder"
    if output_path not in channels:
        output_path = "strong_decoder_to_frame"
    return _TimelineLanes(
        input_path=input_path,
        output_path=output_path,
        store_path=store_path,
        store_name=store_name,
    )


def _index_move(event: dict, by_round: dict, by_window: dict) -> None:
    """One link move filed under the round or the window it carried."""
    channel = event["args"].get("channel")
    if channel is None:
        return
    span = _span_of(event)
    window_id = trace_file.window_id_of(event)
    if window_id is not None:
        by_window[(channel, window_id)] = span
        return
    rounds = event["args"].get("rounds_by_operation")
    if rounds is None:
        return
    # a move that names no window carries one round of one operation
    (rounds_text,) = rounds.values()
    round_lo, _round_hi = trace_file.range_of(rounds_text)
    by_round[(channel, round_lo)] = span


def _timeline_windows(document) -> dict:
    """Window id -> the rounds it reads and the tick its unit took it."""
    dispatch_microseconds = {}
    for event in document.of_phase("X"):
        if not event["name"].endswith(" queued"):
            continue
        window_id = trace_file.window_id_of(event)
        dispatch_ticks = trace_file.end_tick_of(event)
        dispatch_microseconds[window_id] = config_module.ticks_to_microseconds(
            dispatch_ticks
        )
    windows = {}
    for event in document.of_phase("i"):
        if not event["name"].endswith(" ready"):
            continue
        window = _timeline_window(event, dispatch_microseconds)
        windows[window.window_id] = window
    return windows


def _timeline_window(
    event: dict, dispatch_microseconds: dict
) -> _TimelineWindow:
    """One window's rounds and the moment its unit was assigned."""
    window_id = trace_file.window_id_of(event)
    read_lo, read_hi = _own_read_range(event["args"])
    commit_lo, commit_hi = trace_file.range_of(event["args"]["commit"])
    ready_ticks = trace_file.tick_of(event)
    ready_microseconds = config_module.ticks_to_microseconds(ready_ticks)
    dispatch = dispatch_microseconds.get(window_id, ready_microseconds)
    return _TimelineWindow(
        window_id=window_id,
        read_lo=read_lo,
        read_hi=read_hi,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        dispatch_us=dispatch,
    )


def _own_read_range(args: dict) -> tuple:
    """The rounds a window reads of its own operation, the figure's stream.

    The ready event names the rounds the stream has, so a lookahead
    window's buffer past the stream's end is not among them
    (windows/round_retention.py, read_keys_for_bounds).
    """
    window = args["window"]
    operation, _window_index = window.split(":")
    rounds_by_operation = args["rounds_by_operation"]
    rounds_text = rounds_by_operation[operation]
    return trace_file.range_of(rounds_text)


def _timeline_stages(document) -> dict:
    """(window id, stage) -> the span the decoder engine spent in it."""
    stages = {}
    for event in document.of_phase("X"):
        if event["cat"] != "stage":
            continue
        window_id = trace_file.window_id_of(event)
        stages[(window_id, event["name"])] = _span_of(event)
    return stages


def _frame_spans(document) -> dict:
    """Window id -> the span from the frame accepting a write to landing."""
    accepted_microseconds = {}
    for event in document.of_phase("X"):
        if not event["name"].endswith(" correction"):
            continue
        window_id = trace_file.window_id_of(event)
        accepted_ticks = trace_file.tick_of(event)
        accepted_microseconds[window_id] = config_module.ticks_to_microseconds(
            accepted_ticks
        )
    spans = {}
    for event in document.of_phase("i"):
        if not event["name"].endswith(" committed"):
            continue
        window_id = trace_file.window_id_of(event)
        committed_ticks = trace_file.tick_of(event)
        committed = config_module.ticks_to_microseconds(committed_ticks)
        accepted = accepted_microseconds.get(window_id, committed)
        spans[window_id] = _Span(start_us=accepted, end_us=committed)
    return spans


def _span_of(event: dict) -> _Span:
    """A complete event's bar, from its own ticks and not its float ts."""
    start_ticks = trace_file.tick_of(event)
    end_ticks = trace_file.end_tick_of(event)
    start_microseconds = config_module.ticks_to_microseconds(start_ticks)
    end_microseconds = config_module.ticks_to_microseconds(end_ticks)
    return _Span(start_us=start_microseconds, end_us=end_microseconds)


def _round_period_microseconds(moves_by_round: dict) -> float:
    """The shot's round period, as the median gap between QPU sends.

    The trace records when each round left the QPU, not the period the
    yaml set; on a shot whose rounds are evenly spaced the two are the
    same number, and the bar is drawn one period wide as before.
    """
    sends = []
    for (channel, _round_number), span in moves_by_round.items():
        if channel != "qpu_to_controller":
            continue
        sends.append(span.start_us)
    ordered = sorted(sends)
    gaps = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        gap = later - earlier
        gaps.append(gap)
    if not gaps:
        return 0.0
    return statistics.median(gaps)


def _timeline_rows(lanes: _TimelineLanes, shot: _TimelineShot) -> list:
    """The timeline's row labels, top to bottom, one per hop."""
    rows = ["qpu round", "qc link"]
    if _stored_rounds(lanes, shot):
        rows.append(f"{lanes.store_name} link")
    rows.append(f"{lanes.store_name} fill")
    rows.append("wait")
    rows.append(f"transfer ({lanes.input_path})")
    rows.append("fetch")
    rows.append("algorithm")
    rows.append("release")
    rows.append("dd handoff")
    rows.append(f"{lanes.output_path} link")
    rows.append("frame commit")
    return rows


def _stored_rounds(lanes: _TimelineLanes, shot: _TimelineShot) -> dict:
    """Round number -> its move into the store, empty when none was made."""
    return _moves_on(shot.moves_by_round, lanes.store_path)


def _moves_on(moves_by_key: dict, channel: str) -> dict:
    """One channel's moves, keyed by the round or window they carried."""
    found = {}
    for (path, key), span in moves_by_key.items():
        if path == channel:
            found[key] = span
    return found


def _draw_rounds(
    timeline: _TimelineAxes, lanes: _TimelineLanes, shot: _TimelineShot
) -> None:
    """Every round's QPU time, its QC hop, and its hop into the store."""
    qpu_link = _moves_on(shot.moves_by_round, "qpu_to_controller")
    store = _stored_rounds(lanes, shot)
    for round_number in sorted(qpu_link):
        shade = "0.75"
        if round_number % 2:
            shade = "0.55"
        sent = qpu_link[round_number].start_us
        round_start = sent - shot.round_period_microseconds
        timeline.round_bar(
            "qpu round", round_start, shot.round_period_microseconds, shade
        )
        link_time = qpu_link[round_number].end_us - sent
        timeline.round_bar("qc link", sent, link_time, shade)
        if round_number not in store:
            continue
        store_span = store[round_number]
        store_time = store_span.end_us - store_span.start_us
        timeline.round_bar(
            f"{lanes.store_name} link", store_span.start_us, store_time, shade
        )


def _draw_windows(
    timeline: _TimelineAxes, lanes: _TimelineLanes, shot: _TimelineShot
) -> list:
    """Every decoded window's stages, and the legend text for each."""
    stored = _stored_rounds(lanes, shot)
    if not stored:
        stored = _moves_on(shot.moves_by_round, "qpu_to_controller")
    window_ranges = []
    windows = shot.windows.items()
    for window_id, window in sorted(windows):
        if window_id not in shot.frame:
            continue
        color = WINDOW_COLORS[window_id % len(WINDOW_COLORS)]
        range_text = _window_range_text(window)
        window_ranges.append(range_text)
        _draw_store_fill(timeline, lanes, window, color, stored)
        stored_tick = stored[window.read_hi].end_us
        timeline.window_bar(
            "wait", stored_tick, window.dispatch_us, window_id, color
        )
        _draw_window_transfers(timeline, lanes, shot, window_id, color)
        _draw_window_stages(timeline, shot, window_id, color)
    return window_ranges


def _window_range_text(window: _TimelineWindow) -> str:
    """The legend line for one window: what it commits and what it reads."""
    return (
        f"window {window.window_id}: "
        f"commits {window.commit_lo}-{window.commit_hi}, "
        f"reads {window.read_lo}-{window.read_hi}"
    )


def _draw_store_fill(
    timeline: _TimelineAxes,
    lanes: _TimelineLanes,
    window: _TimelineWindow,
    color: str,
    stored: dict,
) -> None:
    """The window's rounds landing in the store it reads.

    Commit rounds land solid; the trailing buffer reads land lighter.
    """
    row = f"{lanes.store_name} fill"
    window_id = window.window_id
    first_round = stored[window.read_lo].end_us
    commit_stored = stored[window.commit_hi].end_us
    timeline.window_bar(row, first_round, commit_stored, window_id, color)
    if window.read_hi <= window.commit_hi:
        return
    stored_tick = stored[window.read_hi].end_us
    timeline.window_bar(
        row, commit_stored, stored_tick, window_id, color, alpha=0.45
    )


def _draw_window_transfers(
    timeline: _TimelineAxes,
    lanes: _TimelineLanes,
    shot: _TimelineShot,
    window_id: int,
    color: str,
) -> None:
    """The window's input link, boundary handoff and output link bars."""
    _draw_transfer_bar(
        timeline,
        f"transfer ({lanes.input_path})",
        shot,
        lanes.input_path,
        window_id,
        color,
    )
    _draw_transfer_bar(
        timeline, "dd handoff", shot, "decoder_to_decoder", window_id, color
    )
    _draw_transfer_bar(
        timeline,
        f"{lanes.output_path} link",
        shot,
        lanes.output_path,
        window_id,
        color,
    )
    commit = shot.frame[window_id]
    timeline.window_bar(
        "frame commit", commit.start_us, commit.end_us, window_id, color
    )


def _draw_transfer_bar(
    timeline: _TimelineAxes,
    row: str,
    shot: _TimelineShot,
    channel: str,
    window_id: int,
    color: str,
) -> None:
    """One link hop's bar, when the window crossed that link."""
    span = shot.moves_by_window.get((channel, window_id))
    if span is None:
        return
    timeline.window_bar(row, span.start_us, span.end_us, window_id, color)


def _draw_window_stages(
    timeline: _TimelineAxes,
    shot: _TimelineShot,
    window_id: int,
    color: str,
) -> None:
    """The decoder engine's fetch, algorithm and release bars."""
    for stage in ("fetch", "algorithm", "release"):
        span = shot.stages.get((window_id, stage))
        if span is None:
            continue
        timeline.window_bar(stage, span.start_us, span.end_us, window_id, color)


def _label_timeline(
    timeline: _TimelineAxes, document, shot: _TimelineShot
) -> None:
    """The timeline's axes labels, title and geometry subtitle."""
    axis = timeline.axis
    axis.set_yticks(range(len(timeline.rows)))
    axis.set_yticklabels(timeline.rows, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("time from shot start (µs)")
    escalation_kind = _escalation_kind(document)
    title = TIMELINE_TITLES.get(escalation_kind, document.process_name)
    axis.set_title(title, pad=22)
    subtitle = _timeline_subtitle(document, shot)
    axis.text(
        0.5,
        1.005,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )


def _escalation_kind(document) -> str:
    """The escalation kind the traced run used, from its process name."""
    words = document.process_name.split()
    if len(words) < 2:
        return ""
    return words[1]


def _timeline_subtitle(document, shot: _TimelineShot) -> str:
    """The geometry line above the timeline: the point and its windows."""
    words = document.process_name.split()
    point_text = " ".join(words[2:])
    if not shot.windows:
        return point_text
    first_id = min(shot.windows)
    first = shot.windows[first_id]
    commit_rounds = first.commit_hi - first.commit_lo + 1
    buffer_rounds = first.read_hi - first.commit_hi
    return (
        f"{point_text}"
        f" · windows: commit {commit_rounds}, buffer {buffer_rounds} rounds"
        f" · rounds every {shot.round_period_microseconds:g} µs"
    )


def _add_timeline_legend(
    timeline: _TimelineAxes, windows: dict, window_ranges: list
) -> None:
    """The legend: the rounds, one entry per window, and the lighter fill."""
    import matplotlib.pyplot as plt

    handles = [plt.Rectangle((0, 0), 1, 1, color="0.6")]
    labels = ["rounds"]
    window_ids = sorted(windows)
    legend_windows = window_ids[:MAX_LEGEND_WINDOWS]
    for window_id in legend_windows:
        color = WINDOW_COLORS[window_id % len(WINDOW_COLORS)]
        patch = plt.Rectangle((0, 0), 1, 1, color=color)
        handles.append(patch)
        label = _legend_label(window_ranges, window_id)
        labels.append(label)
    if len(windows) > MAX_LEGEND_WINDOWS:
        labels[-1] += "  (…)"
    buffer_patch = plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.45)
    handles.append(buffer_patch)
    labels.append("lighter fill = buffer reads (not committed)")
    axis = timeline.axis
    axis.legend(handles, labels, loc="lower left", fontsize=7.5)
    axis.grid(alpha=0.5, linewidth=0.8)
    axis.set_axisbelow(True)


def _legend_label(window_ranges: list, window_id: int) -> str:
    """One window's legend text, its id alone when it never decoded."""
    if window_id < len(window_ranges):
        return window_ranges[window_id]
    return f"window {window_id}"


def _metadata_value(row: dict, path: str, source):
    """The value a row's point set at a path, which it must have set.

    source names the folder or file the row came from, for the refusal.
    """
    metadata = json.loads(row["metadata"])
    if path in metadata:
        return metadata[path]
    listed = sorted(metadata)
    raise refusal.RefusalError(
        f"the points of {source} set no {path}; their sweep sets {listed}"
    )


def _selected_rows(path: pathlib.Path, selection: Selection) -> list:
    """A csv file's rows whose points hold every value where names."""
    rows = _csv_rows(path)
    kept = []
    for row in rows:
        if _holds(row, selection.where):
            kept.append(row)
    if kept:
        return kept
    where_text = _where_text(selection.where)
    raise refusal.RefusalError(f"{path} holds no point where {where_text}")


def _holds(row: dict, where: Mapping) -> bool:
    """Whether a row's point set every path where names to its value."""
    metadata = json.loads(row["metadata"])
    for path, value in where.items():
        if metadata.get(path) != value:
            return False
    return True


def _where_text(where: Mapping) -> str:
    """The where values as the command line wrote them."""
    pairs = []
    for path, value in where.items():
        pairs.append(f"{path}={value}")
    return ", ".join(pairs)


def _titled(text: str, selection: Selection) -> str:
    """A figure's title, the values its points were kept at after it."""
    if not selection.where:
        return f"{text} vs {selection.x_path}"
    where_text = _where_text(selection.where)
    return f"{text} vs {selection.x_path}, {where_text}"


def _sweep_rows(run_dir, selection: Selection) -> list:
    """A run's sweep.csv rows at the where values; one without is refused."""
    sweep_path = pathlib.Path(run_dir) / "sweep.csv"
    if not sweep_path.is_file():
        raise refusal.RefusalError(
            f"{run_dir} has no sweep.csv; the ler figure reads the "
            "logical_error_rate, ler_wilson_low and ler_wilson_high "
            "columns of a `decsim collect` run folder"
        )
    return _selected_rows(sweep_path, selection)


def _draw_ler_curves(
    axis, rows: list, selection: Selection, tier_label: str
) -> list:
    """One run's curves, one per group value; returns every x value."""
    curves = {}
    x_values = []
    for row in rows:
        x_value = _metadata_value(row, selection.x_path, "sweep.csv")
        x_values.append(x_value)
        group_value = _group_value(row, selection.group_path)
        curve = curves.setdefault(group_value, [])
        curve.append((x_value, row))
    for group_value, curve in curves.items():
        label = _curve_label(tier_label, selection.group_path, group_value)
        _draw_measured_ler_points(axis, curve, label)
    return x_values


def _group_value(row: dict, group_path: Optional[str]):
    """The value of the group setting at a row's point; None for no group."""
    if group_path is None:
        return None
    return _metadata_value(row, group_path, "sweep.csv")


def _curve_label(tier_label: str, group_path, group_value) -> str:
    """A curve's legend: the run's tier, and its group value when grouped."""
    if group_path is None:
        return tier_label
    return f"{tier_label}, {group_path}={group_value}"


def _draw_measured_ler_points(axis, curve: list, label: str) -> None:
    """A curve's failures > 0 points: a connected line with Wilson bars."""
    curve.sort(key=_by_x_value)
    x_values = []
    rates = []
    bars_below = []
    bars_above = []
    for x_value, row in curve:
        if int(row["logical_failures"]) == 0:
            continue
        rate = float(row["logical_error_rate"])
        x_values.append(x_value)
        rates.append(rate)
        low = float(row["ler_wilson_low"])
        high = float(row["ler_wilson_high"])
        below = rate - low
        bars_below.append(below)
        above = high - rate
        bars_above.append(above)
    axis.errorbar(
        x_values,
        rates,
        yerr=[bars_below, bars_above],
        fmt="o-",
        capsize=3,
        label=label,
    )


def _by_x_value(point: tuple):
    return point[0]


def _scale_x_axis(axis, values: list) -> None:
    """Log for values spanning a decade, as an error rate's; else linear.

    A log axis ticks every value in the y axis's power-of-ten notation,
    since a decades-only axis labels two of seven swept error rates.
    """
    import matplotlib.ticker as ticker

    swept = sorted(set(values))
    if not _spans_a_decade(swept):
        axis.set_xticks(swept)
        return
    axis.set_xscale("log")
    axis.set_xticks(swept)
    tick_labels = []
    for value in swept:
        tick_label = _power_of_ten_label(value)
        tick_labels.append(tick_label)
    axis.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
    no_minor_labels = ticker.NullFormatter()
    axis.xaxis.set_minor_formatter(no_minor_labels)


def _spans_a_decade(swept: list) -> bool:
    """Whether sorted positive values reach LOG_AXIS_SPAN times their least."""
    if swept[0] <= 0:
        return False
    return swept[-1] >= LOG_AXIS_SPAN * swept[0]


def _power_of_ten_label(value: float) -> str:
    r"""5e-4 -> $5{\times}10^{-4}$, 1e-3 -> $10^{-3}$: the axis's notation."""
    logarithm = math.log10(value)
    exponent = math.floor(logarithm)
    mantissa = value / 10.0**exponent
    if math.isclose(mantissa, 1.0):
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}{{\\times}}10^{{{exponent}}}$"


def _run_tier_label(run_dir, algorithm_field: str) -> str:
    """The run's decoder and tier, read back from what the run recorded.

    "pymatching (weak)" or "relay bp (strong)". The algorithm column
    names the card of the tier that decodes the plan's windows, and that
    tier is the escalation row's primary tier, read from the settings
    the run's first point recorded in resolved/. A numeric card reads as
    pymatching: the card prices latency but its corrections come from
    the same MWPM path.
    """
    try:
        float(algorithm_field)
        algorithm_name = "pymatching"
    except ValueError:
        algorithm_name = algorithm_field
    records = run_folder.resolved_by_point(run_dir)
    record_values = records.values()
    record = next(iter(record_values))
    escalation_kind = record["settings"]["escalation"]["kind"]
    escalation_row = tables.row(
        escalation_settings.ESCALATIONS, "escalation.kind", escalation_kind
    )
    tier = escalation_row.primary_tier.value
    display_name = algorithm_name.replace("_", " ")
    return f"{display_name} ({tier})"


def _memory_class_rows(run_dir, selection: Selection) -> list:
    """One run folder's per-class data-movement rows, by rising x value.

    A folder whose run had observation.data_movement off wrote no file
    and is refused, because the figure has nothing to draw for it.
    """
    movement_path = pathlib.Path(run_dir) / "data_movement.csv"
    if not movement_path.is_file():
        raise refusal.RefusalError(
            f"{run_dir} has no data_movement.csv; the data movement figure "
            "reads the copy and move bits per memory class, which a run "
            "records when its observation section says data_movement: true"
        )
    all_rows = _selected_rows(movement_path, selection)
    class_rows = []
    for row in all_rows:
        if row["grouping"] == "memory_class":
            x_value = _metadata_value(row, selection.x_path, run_dir)
            class_rows.append((x_value, row))
    class_rows.sort(key=_by_x_value)
    return class_rows


def _classes_in_order(rows: list) -> list:
    """The memory classes these rows carry, in the order they appear."""
    classes = []
    for _x_value, row in rows:
        if row["name"] not in classes:
            classes.append(row["name"])
    return classes


def _rows_of_class(rows: list, memory_class: str) -> list:
    """Every row of one memory class, in the order they were sorted."""
    found = []
    for x_value, row in rows:
        if row["name"] == memory_class:
            found.append((x_value, row))
    return found


def _draw_movement_panel(axis, series: dict, x_path: str) -> None:
    """One config's classes, copied solid and moved dashed, on a log axis."""
    swept = _swept_values_of(series)
    class_index = 0
    for memory_class, by_word in series.items():
        color = f"C{class_index}"
        _draw_class_series(axis, by_word, memory_class, color)
        class_index += 1
    axis.set_yscale("log")
    axis.set_xticks(swept)
    axis.set_xlabel(x_path)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)


def _draw_class_series(axis, by_word: dict, memory_class: str, color) -> None:
    """One memory class's two lines; a line of only zeros is not drawn."""
    for _column, word, marker, style in MOVEMENT_SERIES:
        drawn_values, drawn_bits = by_word[word]
        if not drawn_bits:
            continue
        axis.plot(
            drawn_values,
            drawn_bits,
            marker=marker,
            linestyle=style,
            color=color,
            label=f"{memory_class} {word}",
        )


def _swept_values_of(series: dict) -> list:
    """Every x value any class of one run carries, rising."""
    swept = set()
    for by_word in series.values():
        for x_values, _bits in by_word.values():
            swept.update(x_values)
    return sorted(swept)


def _series_of(rows: list, column: str) -> tuple:
    """The x values and bits of one column, zeros left off the log axis."""
    x_values = []
    bits = []
    for x_value, row in rows:
        value = float(row[column])
        if value <= 0:
            continue
        x_values.append(x_value)
        bits.append(value)
    return x_values, bits


def _study_config_name(run_dir) -> str:
    """The yaml a run folder ran, off the manifest it recorded."""
    folder = pathlib.Path(run_dir)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        return folder.name
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    config_files = manifest["config_files"]
    named = pathlib.Path(config_files[0])
    return named.stem


def _csv_rows(path) -> list:
    """Every row of one csv file, as dicts of text."""
    with open(path) as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _shot_rows(run_dir, selection: Selection) -> list:
    """The run's shots.csv rows the where values keep; none is refused."""
    shots_path = pathlib.Path(run_dir) / "shots.csv"
    if not shots_path.exists():
        raise refusal.RefusalError(
            f"{run_dir} has no shots.csv; the stage breakdown reads the "
            f"per-shot stage means a closed-loop run records"
        )
    return _selected_rows(shots_path, selection)


def _median_stage_us_by_value(rows: list, x_path: str) -> dict:
    """The median us per stage, in breakdown order, keyed by x value.

    The median is over shots of each shot's per-window mean, so one
    slow shot cannot move the bar the way a mean of means would let it.
    """
    samples_by_value = {}
    for row in rows:
        x_value = _metadata_value(row, x_path, "shots.csv")
        empty = _empty_stage_samples()
        per_stage = samples_by_value.setdefault(x_value, empty)
        _collect_stage_samples(per_stage, row)
    medians = {}
    items = samples_by_value.items()
    for x_value, per_stage in sorted(items):
        medians[x_value] = _stage_medians(per_stage)
    return medians


def _empty_stage_samples() -> list:
    """One empty sample list per stage of the breakdown."""
    per_stage = []
    for _column, _label in STAGE_BREAKDOWN_STAGES:
        per_stage.append([])
    return per_stage


def _collect_stage_samples(per_stage: list, row: dict) -> None:
    """One shot's per-stage means appended to its x value's samples."""
    for stage_index, stage in enumerate(STAGE_BREAKDOWN_STAGES):
        column = stage[0]
        per_stage[stage_index].append(float(row[column]))


def _stage_medians(per_stage: list) -> list:
    """The median of each stage's samples, in breakdown order."""
    medians = []
    for values in per_stage:
        median = statistics.median(values)
        medians.append(median)
    return medians


def _draw_stage_bars(axis, medians_by_value: dict, values: list) -> list:
    """One stacked segment per stage; returns each bar's running total."""
    stacked_left = [0.0] * len(values)
    bar_positions = range(len(values))
    for stage_index, stage in enumerate(STAGE_BREAKDOWN_STAGES):
        stage_label = stage[1]
        stage_widths = _stage_widths(medians_by_value, values, stage_index)
        axis.barh(
            bar_positions,
            stage_widths,
            left=stacked_left,
            height=0.6,
            label=stage_label,
        )
        stacked_left = _stacked(stacked_left, stage_widths)
    return stacked_left


def _stage_widths(
    medians_by_value: dict, values: list, stage_index: int
) -> list:
    """One stage's bar width per x value, in milliseconds."""
    widths = []
    for value in values:
        median_us = medians_by_value[value][stage_index]
        median_ms = median_us / 1000.0
        widths.append(median_ms)
    return widths


def _stacked(stacked_left: list, stage_widths: list) -> list:
    """The running totals after one stage's segment is laid down."""
    totals = []
    for left, width in zip(stacked_left, stage_widths, strict=True):
        total = left + width
        totals.append(total)
    return totals


def _label_stage_totals(axis, bar_positions, stacked_left: list) -> None:
    """The total beside each stacked bar."""
    for position, total in zip(bar_positions, stacked_left, strict=True):
        label = f"{total:,.0f}"
        if total < 100:
            label = f"{total:.3g}"
        axis.text(total, position, f"  {label}", va="center", fontsize=8)


def _breakdown_title(algorithm) -> str:
    """The breakdown figure's title for the tier that ran."""
    if algorithm in BREAKDOWN_TITLES:
        return BREAKDOWN_TITLES[algorithm]
    label = _card_label(algorithm)
    return f"Time breakdown: {label}"


def _log_values_of(pooled: dict) -> list:
    """log10 of every sample, one list per x value."""
    log_values = []
    for samples in pooled.values():
        log_samples = []
        for sample in samples:
            log_sample = math.log10(sample)
            log_samples.append(log_sample)
        log_values.append(log_samples)
    return log_values


def _log_decade_axis(axis, log_values: list) -> None:
    """Label a log10-transformed time axis in plain microseconds.

    The violins are drawn on log10(us) values so their density is
    estimated in log space, where wall-clock latency is roughly
    symmetric; a raw linear KDE under a log axis would smear the tails.
    """
    minima = []
    maxima = []
    for values in log_values:
        minima.append(min(values))
        maxima.append(max(values))
    lowest = math.floor(min(minima))
    highest = math.ceil(max(maxima))
    past_highest = highest + 1
    ticks = list(range(lowest, past_highest))
    axis.set_yticks(ticks)
    tick_labels = []
    for tick in ticks:
        microseconds = 10.0**tick
        tick_labels.append(f"{microseconds:g}")
    axis.set_yticklabels(tick_labels)


def _latency_violins(
    axis, pooled: dict, positions: list, width: float, color: str, label: str
) -> None:
    """One violin per x value on log10(us) values, median marked."""
    log_samples = _log_values_of(pooled)
    parts = axis.violinplot(
        log_samples,
        positions=positions,
        widths=width,
        showmedians=True,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.6)
    parts["cmedians"].set_color(color)
    maxima = []
    for samples in log_samples:
        maxima.append(max(samples))
    axis.plot(
        positions,
        maxima,
        "v",
        color=color,
        markersize=4,
        label=f"{label} (worst window marked)",
    )


def _deadline_line(axis, deadlines: dict) -> None:
    """The deadline: a new window every commit rounds times the round period.

    Decode must beat the window's inter-arrival to keep up; each x
    value's is its rows' window_period_us.
    """
    x_values = sorted(deadlines)
    deadline_log_us = []
    for x_value in x_values:
        log_deadline = math.log10(deadlines[x_value])
        deadline_log_us.append(log_deadline)
    axis.plot(
        x_values,
        deadline_log_us,
        "--",
        color="grey",
        label="window generation",
    )


def _deadlines_by_value(deadlines: dict, rows: list, x_path: str) -> None:
    """Each x value's window inter-arrival, off the rows that carry it."""
    for row in rows:
        x_value = _metadata_value(row, x_path, "latency_samples.csv")
        deadlines[x_value] = float(row["window_period_us"])


def _samples_by_value(rows: list, x_path: str) -> dict:
    """The algorithm wall clock of every row, keyed by x value."""
    pooled = {}
    for row in rows:
        x_value = _metadata_value(row, x_path, "latency_samples.csv")
        samples = pooled.setdefault(x_value, [])
        samples.append(float(row["algorithm_us"]))
    items = pooled.items()
    ordered = sorted(items)
    return dict(ordered)


def _sample_files(run_dirs) -> list:
    """Each folder's latency_samples.csv; one without it is refused."""
    paths = []
    for run_dir in run_dirs:
        samples_path = pathlib.Path(run_dir) / "latency_samples.csv"
        if not samples_path.is_file():
            raise refusal.RefusalError(
                f"{run_dir} has no latency_samples.csv; the latency figure "
                "reads the decode wall clock per window, which a run whose "
                "decoder is named and not a latency card records"
            )
        paths.append(samples_path)
    return paths
