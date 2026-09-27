"""The figures that read decsim's own records, not a sweep's numbers.

timeline.png         one traced shot, read from its Chrome trace file:
                     every stage of every window on its own row, in real
                     time, the weak baseline's figure style (commit
                     reads solid, buffer reads lighter, geometry in the
                     subtitle). `decsim run --trace`, or `trace: chrome`
                     in the observation section, writes the file this
                     reads, and `decsim collect` draws it for the first
                     traced shot of the sweep's first traced point
stage_breakdown.png  where a window's time goes, one stacked bar per
                     sweep point, the stages in the order the pipeline
                     runs them, from shots.csv

A figure of a sweep's numbers against a setting is the reader's to
draw: the run folder holds every point's values by path beside the
counts and times a figure is drawn from (docs/reference/run_folder.md);
what a figure computes from them, such as a bar's length or a median,
is computed when it is drawn and not stored. sinter keeps its figures in
a separate `sinter plot` over the csv it wrote
(sinter/_command/_main_plot.py). Both figures read times in
microseconds, as the records hold them; the timeline draws them in
microseconds and the stage breakdown in milliseconds.
"""

import csv
import dataclasses
import json
import pathlib
import statistics
from typing import Optional

import decsim.config as config_module
import decsim.experiments.fold as fold
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.experiments.trace_file as trace_file

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
}
# where collect_command leaves the traces of the shots it traced
TRACE_DIR = "trace"


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
    """The first traced shot of a run folder, None when nothing traced.

    A sweep writes one file per traced shot under trace/, named by the
    shot's point id and seed (experiments/measure.py shot_label). The
    ids are hashes, so their name order is no order of the sweep's; the
    manifest lists the points in task order, and the figure draws the
    lowest traced seed of the first point that traced one.
    """
    trace_dir = pathlib.Path(run_dir) / TRACE_DIR
    if not trace_dir.is_dir():
        return None
    for point_id in _recorded_points(run_dir):
        traced = _traced_shots_of(trace_dir, point_id)
        if traced:
            return traced[0]
    return None


def stage_breakdown_plot(run_dir, path: pathlib.Path) -> None:
    """One stacked bar per sweep point: where a window's time goes.

    From syndrome arrival in the buffer to the Pauli-frame commit. Each
    bar is one point, labelled by the values its sweep set and its
    decoder, so no two points' shots are pooled into one bar.

        decsim plot <run_dir> --figure stage_breakdown
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _shot_rows(run_dir)
    medians_by_point = _median_stage_us_by_point(rows)
    point_ids = list(medians_by_point)
    labels = _point_labels(run_dir, rows)
    height = _breakdown_height(labels)
    # constrained layout keeps the axis clear of every label, and the
    # key below it, rather than shrinking what does not fit off the edge
    figure, axis = plt.subplots(figsize=(8.0, height), layout="constrained")
    bar_positions = range(len(point_ids))
    stacked_left = _draw_stage_bars(axis, medians_by_point, point_ids)
    _label_stage_totals(axis, bar_positions, stacked_left)
    axis.set_yticks(list(bar_positions))
    tick_labels = []
    for point_id in point_ids:
        tick_labels.append(labels[point_id])
    axis.set_yticklabels(tick_labels, fontsize=7)
    axis.invert_yaxis()
    widest = max(stacked_left)
    right_edge = widest * 1.2
    axis.set_xlim(0, right_edge)
    # every breakdown is drawn in ms so two runs' figures share one
    # unit; the axis stays linear with plain tick numbers
    axis.set_xlabel("median time per window (ms)")
    axis.set_title("Time breakdown per window")
    figure.legend(loc="outside lower center", fontsize=7, ncol=3)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plots(report_dir: pathlib.Path) -> None:
    """The figure `decsim collect` draws itself: the first traced shot.

    timeline.png needs a traced shot, so it is drawn only when the
    observation section asked for a trace.
    """
    import matplotlib

    matplotlib.use("Agg")
    trace_path = first_trace_file(report_dir)
    if trace_path is None:
        return
    timeline_path = report_dir / "timeline.png"
    timeline_plot(trace_path, timeline_path)


def figure(name: str, run_dir, out_path=None) -> pathlib.Path:
    """Draw one named figure from a run folder, and return where it went.

    The names are the rows of FIGURES. The timeline also reads a trace
    file named in place of the folder.
    """
    import matplotlib

    matplotlib.use("Agg")
    if name not in FIGURES:
        listed = ", ".join(FIGURES)
        raise refusal.RefusalError(
            f"no figure named {name}; the figures are {listed}"
        )
    run_dir = pathlib.Path(run_dir)
    if out_path is None:
        out_path = run_dir / FIGURES[name]
    out_path = pathlib.Path(out_path)
    if name == "timeline":
        trace_path = _timeline_source(run_dir)
        timeline_plot(trace_path, out_path)
        return out_path
    stage_breakdown_plot(run_dir, out_path)
    return out_path


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


def _recorded_points(run_dir) -> list:
    """The point ids a run folder's manifest lists, in task order."""
    manifest_path = pathlib.Path(run_dir) / "manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    return manifest["points"]


def _traced_shots_of(trace_dir: pathlib.Path, point_id: str) -> list:
    """One point's trace files, lowest seed first."""
    found = trace_dir.glob(f"{point_id}_seed*.trace.json")
    return sorted(found, key=_seed_of_trace)


def _seed_of_trace(path: pathlib.Path) -> int:
    """The seed a trace file's name carries: <id>_seed<seed>.trace.json."""
    label, _, _suffixes = path.name.partition(".")
    _point_id, _, seed_text = label.rpartition("_seed")
    return int(seed_text)


def _csv_rows(path) -> list:
    """Every row of one csv file, as dicts of text."""
    with open(path) as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _shot_rows(run_dir) -> list:
    """The run's shots.csv rows; a folder without them is refused."""
    shots_path = pathlib.Path(run_dir) / "shots.csv"
    if not shots_path.exists():
        raise refusal.RefusalError(
            f"{run_dir} has no shots.csv; the stage breakdown reads the "
            "per-shot stage means a closed-loop run records"
        )
    return _csv_rows(shots_path)


def _point_labels(run_dir, rows: list) -> dict:
    """Each point's bar label: the values its sweep set, and its decoder.

    The values are the point's cells in the csv files
    (run_folder.swept_values), the value it ran with at each swept yaml
    path, each on a line of its own, since a path is too long to share
    one.
    """
    algorithms = {row["point_id"]: row["algorithm"] for row in rows}
    point_ids = list(algorithms)
    swept = run_folder.swept_values(run_dir, point_ids)
    labels = {}
    for point_id, cells in swept.items():
        lines = [f"{path}={cell}" for path, cell in cells.items()]
        algorithm_line = _algorithm_line(algorithms[point_id])
        lines.append(algorithm_line)
        labels[point_id] = "\n".join(lines)
    return labels


def _algorithm_line(algorithm_cell: str) -> str:
    """The csv's algorithm cell on a label: a latency card in microseconds."""
    algorithm = fold.number_of(algorithm_cell)
    if isinstance(algorithm, str):
        return f"algorithm {algorithm}"
    return f"algorithm {algorithm:g} us"


def _breakdown_height(labels: dict) -> float:
    """The figure's inches: a bar as tall as its label, then the rest.

    A line of 7 pt text takes 0.12 in; the title, the axis and a three
    row key below it take 2 in.
    """
    newline_counts = [label.count("\n") for label in labels.values()]
    most_newlines = max(newline_counts)
    most_lines = most_newlines + 1
    bar_inches = 0.12 * most_lines + 0.2
    bars = len(labels)
    return bar_inches * bars + 2.0


def _median_stage_us_by_point(rows: list) -> dict:
    """The median us per stage, in breakdown order, keyed by point id.

    The median is over shots of each shot's per-window mean, so one
    slow shot cannot move the bar the way a mean of means would let it.
    The points come in the order shots.csv holds them, the sweep's.
    """
    samples_by_point = {}
    for row in rows:
        empty = _empty_stage_samples()
        per_stage = samples_by_point.setdefault(row["point_id"], empty)
        _collect_stage_samples(per_stage, row)
    medians = {}
    for point_id, per_stage in samples_by_point.items():
        medians[point_id] = _stage_medians(per_stage)
    return medians


def _empty_stage_samples() -> list:
    """One empty sample list per stage of the breakdown."""
    per_stage = []
    for _column, _label in STAGE_BREAKDOWN_STAGES:
        per_stage.append([])
    return per_stage


def _collect_stage_samples(per_stage: list, row: dict) -> None:
    """One shot's per-stage means appended to its point's samples."""
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


def _draw_stage_bars(axis, medians_by_point: dict, point_ids: list) -> list:
    """One stacked segment per stage; returns each bar's running total."""
    stacked_left = [0.0] * len(point_ids)
    bar_positions = range(len(point_ids))
    for stage_index, stage in enumerate(STAGE_BREAKDOWN_STAGES):
        stage_label = stage[1]
        stage_widths = _stage_widths(medians_by_point, point_ids, stage_index)
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
    medians_by_point: dict, point_ids: list, stage_index: int
) -> list:
    """One stage's bar width per point, in milliseconds."""
    widths = []
    for point_id in point_ids:
        median_us = medians_by_point[point_id][stage_index]
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
