"""The experiment figures.

timeline.png   one traced shot, read from its Chrome trace file: every
               stage of every window on its own row, in real time, the
               weak baseline's figure style (commit reads solid, buffer
               reads lighter, geometry in the subtitle). `decsim run
               --trace`, or `trace: chrome` in the observation section,
               writes the file this reads
ler.png        logical error rate vs physical error rate, Wilson 95%
               bars, drawn when more than one p was swept
latency.png    decode wall clock per window vs code distance, violins,
               drawn when a wall-clock algorithm swept more than one d;
               the cross-tier combined figure comes from `decsim plot
               <run_dir> <run_dir> --figure latency`, reading each run's
               latency_samples.csv
ler vs d       both tiers' logical error rate against code distance at
               one physical error rate, from each run's sweep.csv:
               `decsim plot <run_dir> <run_dir> --figure ler_vs_d
               --probability <p>`
data_movement  bits copied and moved per shot, by the memory class the
               hop crosses, against code distance, one panel per study
               config, from each run's data_movement.csv

Every time is in microseconds.
"""

import csv
import dataclasses
import json
import math
import statistics
from pathlib import Path
from typing import Optional

import decsim.config as config_module
import decsim.front.refusal as refusal
import decsim.front.trace_file as trace_file

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
    "ler_vs_d": "ler_vs_distance.png",
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


def ticks_to_microseconds(ticks) -> float:
    """A tick count as a float of microseconds, the figures' time unit."""
    return ticks / config_module.TICKS_PER_MICROSECOND


def card_label(algorithm) -> str:
    """A named algorithm capitalized, a latency card as its microseconds."""
    if isinstance(algorithm, str):
        return algorithm.capitalize()
    return f"{algorithm:g} µs"


def timeline_plot(trace_path, path: Path) -> None:
    """One traced shot's hops and stages, in the time they happened.

    Drawn from that shot's Chrome trace file alone (`decsim run --trace`,
    or `trace: chrome` in the yaml's observation section): the trace
    records where every round and window sat and for how long, so this
    builds no machine and runs nothing.
    """
    import matplotlib.pyplot as plt

    document = trace_file.load(trace_path)
    shot = _timeline_shot(document)
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


def first_trace_file(run_dir) -> Optional[Path]:
    """The first trace file of a run folder, None when nothing traced.

    A sweep writes one file per traced shot under trace/, named after the
    point (front/measure.py); the first in name order is the first
    point's, which is the shot the timeline draws.
    """
    trace_dir = Path(run_dir) / TRACE_DIR
    if not trace_dir.is_dir():
        return None
    entries = trace_dir.iterdir()
    found = sorted(entries)
    for path in found:
        if path.is_file():
            return path
    return None


def ler_groups(rows: list) -> list:
    """(distance, round time, rows sorted by p) per curve of the LER figure.

    One curve per distance and round time that swept more than one
    physical error rate: the papers' convention, one LER curve per code
    distance.
    """
    distances = _column_values(rows, "distance")
    periods = _column_values(rows, "round_period_us")
    groups = []
    for distance in distances:
        at_distance = _curves_at_distance(rows, distance, periods)
        groups.extend(at_distance)
    return groups


def decoder_title(config) -> str:
    """The figure's decoder title, tier included.

    "pymatching decoder (weak)" or "belief matching decoder (strong)".
    A numeric card reads as pymatching: the card prices latency but its
    corrections come from the same MWPM path.
    """
    algorithm = config.active_decoder.kind
    name = "pymatching"
    if isinstance(algorithm, str):
        name = algorithm
    spelled = name.replace("_", " ")
    tier = config.active_tier
    return f"{spelled} decoder ({tier})"


def ler_plot(rows: list, path: Path, title: str = "Logical error rate") -> None:
    """Logical error rate against physical error rate, Wilson 95% bars.

    One line per card and round time that swept more than one p.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    groups = ler_groups(rows)
    plotted_periods = set()
    for _distance, period, _group in groups:
        plotted_periods.add(period)
    for distance, period, group in groups:
        label = f"d={distance}"
        if len(plotted_periods) > 1:
            label = f"d={distance}, {period:g} µs"
        _draw_ler_curve(axis, group, label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    _label_swept_probabilities(axis, rows)
    axis.set_xlabel("Physical error rate")
    axis.set_ylabel("Logical error rate")
    axis.set_title(title)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def ler_vs_distance_plot(
    run_dirs: list, probability: float, path: Path
) -> None:
    """Both tiers' logical error rate against code distance at one p.

    Read from each run's sweep.csv. Measured points carry Wilson 95% bars;
    a zero-failure point cannot sit on a log axis, so its curve simply
    ends at the last distance that saw failures.

        python -m decsim.front.plots ler_vs_d <run_dir> <run_dir> <p>
        <out.png>
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    swept_distances = set()
    for run_index, run_dir in enumerate(run_dirs):
        rows = _ler_rows_at_probability(run_dir, probability)
        color = f"C{run_index}"
        tier_label = _csv_tier_label(rows[0]["algorithm"])
        measured_rows = _rows_with_failures(rows, swept_distances)
        _draw_measured_ler_points(axis, measured_rows, color, tier_label)
    axis.set_yscale("log")
    axis.set_xticks(sorted(swept_distances))
    axis.set_xlabel("Code distance")
    axis.set_ylabel("Logical error rate per shot (10d rounds)")
    probability_label = _power_of_ten_label(probability)
    axis.set_title(f"Logical error rate vs distance, p={probability_label}")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def memory_class_series(run_dir) -> dict:
    """One run's bits per shot by memory class, then by copied and moved.

    memory class -> word -> (distances, bits), read from that folder's
    data_movement.csv memory-class rows. A distance whose bits are zero
    is left out: an off-board hop of this machine moves and never
    copies, and zero has no place on a log axis.
    """
    rows = _memory_class_rows(run_dir)
    series = {}
    for memory_class in _classes_in_order(rows):
        at_class = _rows_of_class(rows, memory_class)
        by_word = {}
        for column, word, _marker, _style in MOVEMENT_SERIES:
            by_word[word] = _series_of(at_class, column)
        series[memory_class] = by_word
    return series


def data_movement_plot(run_dirs: list, path: Path) -> None:
    """Bits copied and moved per shot by memory class, against distance.

    One panel per study config, read from each run folder's
    data_movement.csv memory-class rows. The classes are kept apart
    rather than summed because the classical sources make the class the
    cost: a DRAM access is "a couple of orders-of-magnitude higher than
    the cost of an internal cache access" (Horowitz, ISSCC 2014 lines
    232-247) and an accelerator's access costs what the memory it reads
    costs (Dally, CACM 2020 lines 231-234).

        decsim plot <run_dir>... --figure data_movement
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
        series = memory_class_series(run_dir)
        _draw_movement_panel(axis, series)
        title = _study_config_name(run_dir)
        axis.set_title(title, fontsize=9)
    first_axis = axes[0][0]
    first_axis.set_ylabel("Bits per shot")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def stage_breakdown_plot(run_dir, path: Path) -> None:
    """One stacked bar per distance: where a window's time goes.

    From syndrome arrival in the buffer to the Pauli-frame commit.

        python -m decsim.front.plots stage_breakdown <run_dir> <out.png>
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _shot_rows(run_dir)
    medians_by_distance = _median_stage_us_by_distance(rows)
    distances = list(medians_by_distance)
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    bar_positions = range(len(distances))
    stacked_left = _draw_stage_bars(axis, medians_by_distance, distances)
    _label_stage_totals(axis, bar_positions, stacked_left)
    axis.set_yticks(list(bar_positions))
    axis.set_yticklabels([f"d={distance}" for distance in distances])
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


def latency_samples_by_distance(measurements: list) -> dict:
    """Every window's algorithm-stage wall clock in us, keyed by distance.

    Pooled over shots. The algorithm stage alone, because that is the
    only measured quantity and the papers' comparable number (Helios
    2301.08419 Fig. 6); fetch, release and the links are priced from
    the config and belong to the stage-breakdown figure.
    """
    pooled = {}
    for measurement in measurements:
        samples = pooled.setdefault(measurement.distance, [])
        samples.extend(measurement.samples["algorithm"])
    by_distance = {}
    items = pooled.items()
    for distance, samples in sorted(items):
        if samples:
            by_distance[distance] = samples
    return by_distance


def latency_plot(config, measurements: list, path: Path) -> None:
    """Decode wall clock per window against code distance.

    One violin per d (median marked, worst window flagged), microsecond
    log axis, with the window-generation deadline drawn as the
    throughput boundary. The violin and deadline shape follows Helios
    2301.08419 Fig. 7 and Google 2408.13687 Fig. 4d.
    """
    import matplotlib.pyplot as plt

    pooled = latency_samples_by_distance(measurements)
    distances = list(pooled)
    round_period_us = measurements[0].round_period_us
    probability = measurements[0].physical_error_probability
    algorithm = config.active_decoder.kind
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    _latency_violins(axis, pooled, distances, 1.4, "C0", algorithm)
    _deadline_line(axis, distances, round_period_us)
    log_values = _log_values_of(pooled)
    _log_decade_axis(axis, log_values)
    axis.set_xticks(distances)
    axis.set_xlabel("Code distance")
    axis.set_ylabel("Decode wall clock per window (µs)")
    axis.set_title(f"{algorithm} decode latency, p={probability:g}")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def combined_latency_plot(sample_files: list, path: Path) -> None:
    """Both tiers on one axes from their runs' latency_samples.csv.

    One violin pair per distance. The log axis is what makes this
    legible: the tiers sit decades apart, which is itself the figure's
    message.

        python -m decsim.front.plots latency <run_dir> <run_dir> <out.png>
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    all_log_values = []
    round_period_us = None
    distances = []
    for file_index, sample_file in enumerate(sample_files):
        rows = _csv_rows(sample_file)
        pooled = _samples_by_distance(rows)
        seen = set(distances) | set(pooled)
        distances = sorted(seen)
        round_period_us = float(rows[0]["round_period_us"])
        algorithm = rows[0]["algorithm"]
        color = f"C{file_index}"
        positions = list(pooled)
        _latency_violins(axis, pooled, positions, 1.4, color, algorithm)
        log_values = _log_values_of(pooled)
        all_log_values.extend(log_values)
    _deadline_line(axis, distances, round_period_us)
    _log_decade_axis(axis, all_log_values)
    axis.set_xticks(distances)
    axis.set_xlabel("Code distance")
    axis.set_ylabel("Decode wall clock per window (µs)")
    axis.set_title("Decode latency, weak and strong tiers")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plots(config, rows: list, report_dir: Path, measurements=None) -> None:
    """The figures of one run folder, each one drawn when its input is there.

    timeline.png needs a traced shot, so it is drawn only when the
    observation section asked for a trace; ler.png needs more than one
    p; latency.png needs more than one distance under a wall-clock
    algorithm, because a numeric card is a fixed latency, flat in d, so
    its figure would be a horizontal line.
    """
    import matplotlib

    matplotlib.use("Agg")
    trace_path = first_trace_file(report_dir)
    if trace_path is not None:
        timeline_path = report_dir / "timeline.png"
        timeline_plot(trace_path, timeline_path)
    probabilities = _column_values(rows, "physical_error_probability")
    if len(probabilities) > 1:
        ler_path = report_dir / "ler.png"
        title = decoder_title(config)
        ler_plot(rows, ler_path, title=title)
    measured_wall_clock = isinstance(config.active_decoder.kind, str)
    swept_distances = set()
    for measurement in measurements or ():
        swept_distances.add(measurement.distance)
    if measured_wall_clock and len(swept_distances) > 1:
        latency_path = report_dir / "latency.png"
        latency_plot(config, measurements, latency_path)


def figure(name: str, run_dirs: list, out_path=None, probability=None):
    """Draw one named figure from run folders, and return where it went.

    Every figure here reads files: a run folder's csv rows, or the
    Chrome trace of one of its shots. The names are the rows of
    FIGURES: what each one needs is what its run folders must hold.
    """
    import matplotlib

    matplotlib.use("Agg")
    if name not in FIGURES:
        listed = ", ".join(FIGURES)
        raise refusal.RefusalError(
            f"no figure named {name}; the figures are {listed}"
        )
    first_dir = Path(run_dirs[0])
    if out_path is None:
        out_path = first_dir / FIGURES[name]
    out_path = Path(out_path)
    _draw_named_figure(name, run_dirs, out_path, probability)
    return out_path


def _draw_named_figure(
    name: str, run_dirs: list, out_path: Path, probability
) -> None:
    """The one figure the name asks for, from the folders it was given."""
    first_dir = Path(run_dirs[0])
    if name == "timeline":
        trace_path = _timeline_source(first_dir)
        timeline_plot(trace_path, out_path)
        return
    if name == "stage_breakdown":
        stage_breakdown_plot(first_dir, out_path)
        return
    if name == "latency":
        sample_files = _sample_files(run_dirs)
        combined_latency_plot(sample_files, out_path)
        return
    if name == "data_movement":
        data_movement_plot(run_dirs, out_path)
        return
    if probability is None:
        raise refusal.RefusalError(
            "the ler_vs_d figure is drawn at one physical error rate; "
            "name it with --probability"
        )
    ler_vs_distance_plot(run_dirs, probability, out_path)


def _timeline_source(run_dir: Path) -> Path:
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

    round_period_us: float
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
    store_name = "buffer 0"
    input_path = "weak_buffer_to_weak_decoder"
    output_path = "weak_decoder_to_frame"
    if store_path not in channels:
        store_path = "controller_to_strong_buffer"
        store_name = "syndrome buffer 1"
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


def _timeline_shot(document) -> _TimelineShot:
    """Every span the figure draws, indexed by round, window and stage."""
    moves_by_round = {}
    moves_by_window = {}
    for event in document.of_phase("X"):
        _index_move(event, moves_by_round, moves_by_window)
    windows = _timeline_windows(document)
    stages = _timeline_stages(document)
    frame = _frame_spans(document)
    round_period_us = _round_period_microseconds(moves_by_round)
    return _TimelineShot(
        round_period_us=round_period_us,
        moves_by_round=moves_by_round,
        moves_by_window=moves_by_window,
        windows=windows,
        stages=stages,
        frame=frame,
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
    rounds_text = event["args"].get("rounds")
    if rounds_text is None:
        return
    round_lo, _round_hi = trace_file.range_of(rounds_text)
    by_round[(channel, round_lo)] = span


def _timeline_windows(document) -> dict:
    """Window id -> the rounds it reads and the tick its unit took it."""
    dispatch_us = {}
    for event in document.of_phase("X"):
        if not event["name"].endswith(" queued"):
            continue
        window_id = trace_file.window_id_of(event)
        dispatch_ticks = trace_file.end_tick_of(event)
        dispatch_us[window_id] = ticks_to_microseconds(dispatch_ticks)
    windows = {}
    for event in document.of_phase("i"):
        if not event["name"].endswith(" ready"):
            continue
        window = _timeline_window(event, dispatch_us)
        windows[window.window_id] = window
    return windows


def _timeline_window(event: dict, dispatch_us: dict) -> _TimelineWindow:
    """One window's rounds and the moment its unit was assigned."""
    window_id = trace_file.window_id_of(event)
    read_lo, read_hi = trace_file.range_of(event["args"]["rounds"])
    commit_lo, commit_hi = trace_file.range_of(event["args"]["commit"])
    ready_ticks = trace_file.tick_of(event)
    ready_us = ticks_to_microseconds(ready_ticks)
    dispatch = dispatch_us.get(window_id, ready_us)
    return _TimelineWindow(
        window_id=window_id,
        read_lo=read_lo,
        read_hi=read_hi,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        dispatch_us=dispatch,
    )


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
    accepted_us = {}
    for event in document.of_phase("X"):
        if not event["name"].endswith(" correction"):
            continue
        window_id = trace_file.window_id_of(event)
        accepted_ticks = trace_file.tick_of(event)
        accepted_us[window_id] = ticks_to_microseconds(accepted_ticks)
    spans = {}
    for event in document.of_phase("i"):
        if not event["name"].endswith(" committed"):
            continue
        window_id = trace_file.window_id_of(event)
        committed_ticks = trace_file.tick_of(event)
        committed = ticks_to_microseconds(committed_ticks)
        accepted = accepted_us.get(window_id, committed)
        spans[window_id] = _Span(start_us=accepted, end_us=committed)
    return spans


def _span_of(event: dict) -> _Span:
    """A complete event's bar, from its own ticks and not its float ts."""
    start_ticks = trace_file.tick_of(event)
    end_ticks = trace_file.end_tick_of(event)
    start_us = ticks_to_microseconds(start_ticks)
    end_us = ticks_to_microseconds(end_ticks)
    return _Span(start_us=start_us, end_us=end_us)


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
    for earlier, later in zip(ordered, ordered[1:]):
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
        round_start = sent - shot.round_period_us
        timeline.round_bar(
            "qpu round", round_start, shot.round_period_us, shade
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
        f" · rounds every {shot.round_period_us:g} µs"
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


def _column_values(rows: list, column: str) -> list:
    """The distinct values one column takes over the rows, sorted."""
    values = set()
    for row in rows:
        values.add(row[column])
    return sorted(values)


def _curves_at_distance(rows: list, distance: int, periods: list) -> list:
    """One (distance, period, rows) curve per period that swept two p."""
    curves = []
    for period in periods:
        group = _rows_at(rows, distance, period)
        if _sweeps_one_probability(group):
            continue
        group.sort(key=_by_probability)
        curves.append((distance, period, group))
    return curves


def _rows_at(rows: list, distance: int, period: float) -> list:
    """The rows of one distance and round period."""
    selected = []
    for row in rows:
        if row["distance"] != distance:
            continue
        if row["round_period_us"] != period:
            continue
        selected.append(row)
    return selected


def _sweeps_one_probability(group: list) -> bool:
    """Whether the group has fewer than two physical error rates."""
    probabilities = set()
    for row in group:
        probabilities.add(row["physical_error_probability"])
    return len(probabilities) < 2


def _by_probability(row: dict) -> float:
    return row["physical_error_probability"]


def _draw_ler_curve(axis, group: list, label: str) -> None:
    """One LER curve with its Wilson 95% error bars."""
    probabilities = []
    rates = []
    lower = []
    upper = []
    for row in group:
        rate = row["logical_error_rate"]
        probabilities.append(row["physical_error_probability"])
        rates.append(rate)
        below = rate - row["ler_wilson_low"]
        lower.append(below)
        above = row["ler_wilson_high"] - rate
        upper.append(above)
    axis.errorbar(
        probabilities,
        rates,
        yerr=[lower, upper],
        fmt="o-",
        capsize=3,
        label=label,
    )


def _label_swept_probabilities(axis, rows: list) -> None:
    """Tick every swept p, in the y axis's power-of-ten notation.

    A decades-only log axis labels two of our seven p values.
    """
    from matplotlib.ticker import NullFormatter

    swept = _column_values(rows, "physical_error_probability")
    axis.set_xticks(swept)
    tick_labels = []
    for probability in swept:
        tick_label = _power_of_ten_label(probability)
        tick_labels.append(tick_label)
    axis.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
    no_minor_labels = NullFormatter()
    axis.xaxis.set_minor_formatter(no_minor_labels)


def _power_of_ten_label(value: float) -> str:
    r"""5e-4 -> $5{\times}10^{-4}$, 1e-3 -> $10^{-3}$: the axis's notation."""
    logarithm = math.log10(value)
    exponent = math.floor(logarithm)
    mantissa = value / 10.0**exponent
    if math.isclose(mantissa, 1.0):
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}{{\\times}}10^{{{exponent}}}$"


def _csv_tier_label(algorithm_field: str) -> str:
    """The csv's tier label, read back from its algorithm column.

    "pymatching (weak)" or "belief matching (strong)". A numeric card
    reads as pymatching, the same ruling as decoder_title.
    """
    try:
        float(algorithm_field)
        algorithm_name = "pymatching"
    except ValueError:
        algorithm_name = algorithm_field
    tier = "weak"
    if algorithm_name == "belief_matching":
        tier = "strong"
    display_name = algorithm_name.replace("_", " ")
    return f"{display_name} ({tier})"


def _memory_class_rows(run_dir) -> list:
    """One run folder's per-class data-movement rows, by rising distance.

    A folder whose run had observation.data_movement off wrote no file
    and is refused, because the figure has nothing to draw for it.
    """
    movement_path = Path(run_dir) / "data_movement.csv"
    if not movement_path.is_file():
        raise refusal.RefusalError(
            f"{run_dir} has no data_movement.csv; the data movement figure "
            "reads the copy and move bits per memory class, which a run "
            "records when its observation section says data_movement: true"
        )
    all_rows = _csv_rows(movement_path)
    class_rows = []
    for row in all_rows:
        if row["grouping"] == "memory_class":
            class_rows.append(row)
    return sorted(class_rows, key=_row_distance)


def _row_distance(row: dict) -> int:
    """One row's code distance, as the csv wrote it."""
    return int(row["distance"])


def _classes_in_order(rows: list) -> list:
    """The memory classes these rows carry, in the order they appear."""
    classes = []
    for row in rows:
        if row["name"] not in classes:
            classes.append(row["name"])
    return classes


def _rows_of_class(rows: list, memory_class: str) -> list:
    """Every row of one memory class, in the order they were sorted."""
    found = []
    for row in rows:
        if row["name"] == memory_class:
            found.append(row)
    return found


def _draw_movement_panel(axis, series: dict) -> None:
    """One config's classes, copied solid and moved dashed, on a log axis."""
    swept = _swept_distances_of(series)
    class_index = 0
    for memory_class, by_word in series.items():
        color = f"C{class_index}"
        _draw_class_series(axis, by_word, memory_class, color)
        class_index += 1
    axis.set_yscale("log")
    axis.set_xticks(swept)
    axis.set_xlabel("Code distance")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)


def _draw_class_series(axis, by_word: dict, memory_class: str, color) -> None:
    """One memory class's two lines; a line of only zeros is not drawn."""
    for _column, word, marker, style in MOVEMENT_SERIES:
        drawn_distances, drawn_bits = by_word[word]
        if not drawn_bits:
            continue
        axis.plot(
            drawn_distances,
            drawn_bits,
            marker=marker,
            linestyle=style,
            color=color,
            label=f"{memory_class} {word}",
        )


def _swept_distances_of(series: dict) -> list:
    """Every distance any class of one run carries, rising."""
    swept = set()
    for by_word in series.values():
        for distances, _bits in by_word.values():
            swept.update(distances)
    return sorted(swept)


def _series_of(rows: list, column: str) -> tuple:
    """The distances and bits of one column, zeros left off the log axis."""
    distances = []
    bits = []
    for row in rows:
        value = float(row[column])
        if value <= 0:
            continue
        distance = _row_distance(row)
        distances.append(distance)
        bits.append(value)
    return distances, bits


def _study_config_name(run_dir) -> str:
    """The yaml a run folder ran, off the manifest it recorded."""
    folder = Path(run_dir)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        return folder.name
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    config_files = manifest["config_files"]
    named = Path(config_files[0])
    return named.stem


def _csv_rows(path) -> list:
    """Every row of one csv file, as dicts of text."""
    with open(path) as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _ler_rows_at_probability(run_dir, probability: float) -> list:
    """A run's sweep.csv rows at one p, sorted by distance.

    A run that never swept that p is refused.
    """
    sweep_path = Path(run_dir) / "sweep.csv"
    if not sweep_path.is_file():
        raise refusal.RefusalError(
            f"{run_dir} has no sweep.csv; the ler_vs_d figure reads the "
            "logical_error_rate, ler_wilson_low and ler_wilson_high "
            "columns of a `decsim collect` run folder"
        )
    all_rows = _csv_rows(sweep_path)
    selected_rows = []
    for row in all_rows:
        row_probability = float(row["physical_error_probability"])
        if math.isclose(row_probability, probability):
            selected_rows.append(row)
    if not selected_rows:
        raise refusal.RefusalError(
            f"{run_dir} swept no p={probability:g} point"
        )
    selected_rows.sort(key=_by_distance_text)
    return selected_rows


def _by_distance_text(row: dict) -> int:
    return int(row["distance"])


def _rows_with_failures(rows: list, swept_distances: set) -> list:
    """The rows that saw at least one failure; every distance recorded."""
    measured_rows = []
    for row in rows:
        swept_distances.add(int(row["distance"]))
        if int(row["logical_failures"]) > 0:
            measured_rows.append(row)
    return measured_rows


def _draw_measured_ler_points(axis, rows: list, color: str, label: str) -> None:
    """The failures > 0 rows: a connected line with Wilson 95% bars."""
    distances = []
    rates = []
    bars_below = []
    bars_above = []
    for row in rows:
        rate = float(row["logical_error_rate"])
        distances.append(int(row["distance"]))
        rates.append(rate)
        low = float(row["ler_wilson_low"])
        high = float(row["ler_wilson_high"])
        below = rate - low
        bars_below.append(below)
        above = high - rate
        bars_above.append(above)
    axis.errorbar(
        distances,
        rates,
        yerr=[bars_below, bars_above],
        fmt="o-",
        capsize=3,
        color=color,
        label=label,
    )


def _shot_rows(run_dir) -> list:
    """Every row of the run's shots.csv; a run without one is refused."""
    shots_path = Path(run_dir) / "shots.csv"
    if not shots_path.exists():
        raise refusal.RefusalError(
            f"{run_dir} has no shots.csv; the stage breakdown reads the "
            f"per-shot stage means a closed-loop run records"
        )
    return _csv_rows(shots_path)


def _median_stage_us_by_distance(rows: list) -> dict:
    """The median us per stage, in breakdown order, keyed by distance.

    The median is over shots of each shot's per-window mean, so one
    slow shot cannot move the bar the way a mean of means would let it.
    """
    samples_by_distance = {}
    for row in rows:
        distance = int(row["distance"])
        empty = _empty_stage_samples()
        per_stage = samples_by_distance.setdefault(distance, empty)
        _collect_stage_samples(per_stage, row)
    medians = {}
    items = samples_by_distance.items()
    for distance, per_stage in sorted(items):
        medians[distance] = _stage_medians(per_stage)
    return medians


def _empty_stage_samples() -> list:
    """One empty sample list per stage of the breakdown."""
    per_stage = []
    for _column, _label in STAGE_BREAKDOWN_STAGES:
        per_stage.append([])
    return per_stage


def _collect_stage_samples(per_stage: list, row: dict) -> None:
    """One shot's per-stage means appended to the distance's samples."""
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


def _draw_stage_bars(axis, medians_by_distance: dict, distances: list) -> list:
    """One stacked segment per stage; returns each bar's running total."""
    stacked_left = [0.0] * len(distances)
    bar_positions = range(len(distances))
    for stage_index, stage in enumerate(STAGE_BREAKDOWN_STAGES):
        stage_label = stage[1]
        stage_widths = _stage_widths(
            medians_by_distance, distances, stage_index
        )
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
    medians_by_distance: dict, distances: list, stage_index: int
) -> list:
    """One stage's bar width per distance, in milliseconds."""
    widths = []
    for distance in distances:
        median_us = medians_by_distance[distance][stage_index]
        median_ms = median_us / 1000.0
        widths.append(median_ms)
    return widths


def _stacked(stacked_left: list, stage_widths: list) -> list:
    """The running totals after one stage's segment is laid down."""
    totals = []
    for left, width in zip(stacked_left, stage_widths):
        total = left + width
        totals.append(total)
    return totals


def _label_stage_totals(axis, bar_positions, stacked_left: list) -> None:
    """The total beside each stacked bar."""
    for position, total in zip(bar_positions, stacked_left):
        label = f"{total:,.0f}"
        if total < 100:
            label = f"{total:.3g}"
        axis.text(total, position, f"  {label}", va="center", fontsize=8)


def _breakdown_title(algorithm) -> str:
    """The breakdown figure's title for the tier that ran."""
    if algorithm in BREAKDOWN_TITLES:
        return BREAKDOWN_TITLES[algorithm]
    label = card_label(algorithm)
    return f"Time breakdown: {label}"


def _log_values_of(pooled: dict) -> list:
    """log10 of every sample, one list per distance."""
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
    """One violin per distance on log10(us) values, median marked."""
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


def _deadline_line(axis, distances: list, round_period_us: float) -> None:
    """The deadline: a new window arrives every d rounds.

    The code's default commit region is d rounds, so decode must beat
    d times the round period.
    """
    deadline_log_us = []
    for distance in distances:
        deadline_us = distance * round_period_us
        log_deadline = math.log10(deadline_us)
        deadline_log_us.append(log_deadline)
    axis.plot(
        distances,
        deadline_log_us,
        "--",
        color="grey",
        label=f"window generation ({round_period_us:g} µs rounds)",
    )


def _samples_by_distance(rows: list) -> dict:
    """The algorithm wall clock of every row, keyed by distance."""
    pooled = {}
    for row in rows:
        distance = int(row["distance"])
        samples = pooled.setdefault(distance, [])
        samples.append(float(row["algorithm_us"]))
    items = pooled.items()
    ordered = sorted(items)
    return dict(ordered)


def _sample_files(run_dirs) -> list:
    """Each folder's latency_samples.csv; one without it is refused."""
    paths = []
    for run_dir in run_dirs:
        samples_path = Path(run_dir) / "latency_samples.csv"
        if not samples_path.is_file():
            raise refusal.RefusalError(
                f"{run_dir} has no latency_samples.csv; the latency figure "
                "reads the decode wall clock per window, which a run whose "
                "decoder is named and not a latency card records"
            )
        paths.append(samples_path)
    return paths
