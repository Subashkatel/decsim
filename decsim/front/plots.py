"""The experiment figures.

timeline.png   one shot of the first sweep point, seed 0: every stage of
               every window on its own row, in real time, the weak
               baseline's figure style (commit reads solid, buffer reads
               lighter, geometry in the subtitle)
ler.png        logical error rate vs physical error rate, Wilson 95%
               bars, drawn when more than one p was swept
latency.png    decode wall clock per window vs code distance, violins,
               drawn when a wall-clock algorithm swept more than one d;
               the cross-tier combined figure comes from
               `python -m decsim.front.plots latency <run_dir> <run_dir>
               <out.png>` reading each run's latency_samples.csv
ler vs d       both tiers' logical error rate against code distance at
               one physical error rate, from each run's ler.csv:
               `python -m decsim.front.plots ler_vs_d <run_dir> <run_dir>
               <p> <out.png>`

Every time is in microseconds.
"""

import csv
import dataclasses
import math
import statistics
import sys
from pathlib import Path

import decsim.config as config_module
import decsim.front.measure as measure
import decsim.machine as machine_module

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
USAGE = (
    "usage: python -m decsim.front.plots "
    "latency <run_dir> <run_dir> <out.png>\n"
    "       python -m decsim.front.plots "
    "ler_vs_d <run_dir> <run_dir> <p> <out.png>\n"
    "       python -m decsim.front.plots "
    "stage_breakdown <run_dir> <out.png>"
)


def ticks_to_microseconds(ticks) -> float:
    """A tick count as a float of microseconds, the figures' time unit."""
    return ticks / config_module.TICKS_PER_MICROSECOND


def card_label(algorithm) -> str:
    """A named algorithm capitalized, a latency card as its microseconds."""
    if isinstance(algorithm, str):
        return algorithm.capitalize()
    return f"{algorithm:g} µs"


def timeline_plot(config, path: Path) -> None:
    """One shot at the first sweep point, seed 0, every stage in time."""
    import matplotlib.pyplot as plt

    point = _first_sweep_point(config)
    settings = config.point_settings(
        physical_error_probability=point.physical_error_probability,
        distance=point.distance,
        round_period_us=point.round_period_us,
    )
    completed = machine_module.Machine.build(settings, 0)
    result = completed.run()
    escalation_kind = config.settings.escalation.kind
    lanes = _timeline_lanes(escalation_kind)
    transfers = result.link_traffic["transfers"]
    qpu_link = _transfers_by_round(transfers, "qpu_to_controller")
    store = _transfers_by_round(transfers, lanes.store_path)
    hops = _window_hops(completed, transfers, lanes)
    windows = _timeline_windows(completed)
    rows = _timeline_rows(lanes, store)
    lane_count = max(len(windows), 3)
    height = 0.45 * len(rows) + 1.6
    figure, axis = plt.subplots(figsize=(11, height))
    timeline = _TimelineAxes(axis, rows, lane_count)
    _draw_rounds(timeline, lanes, point.round_period_us, qpu_link, store)
    stored_row = store
    if not store:
        stored_row = qpu_link
    rounds = config.settings.workload.rounds_per_shot.rounds_for(point.distance)
    window_ranges = _draw_windows(
        timeline, lanes, hops, windows, stored_row, rounds
    )
    _label_timeline(timeline, config, point)
    _add_timeline_legend(timeline, windows, window_ranges)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


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
    tier = config.settings.escalation.decodes_on
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

    Read from each run's ler.csv. Measured points carry Wilson 95% bars;
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
    Fig. 6, SWIPER Fig. 3); fetch, release and the links are priced from
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
    Fig. 7, Google Fig. 4d and SWIPER Fig. 3.
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
    """timeline.png always, ler.png and latency.png when the sweep asks.

    ler.png needs more than one p; latency.png needs more than one
    distance under a wall-clock algorithm, because a numeric card is a
    fixed latency, flat in d, so its figure would be a horizontal line.
    """
    import matplotlib

    matplotlib.use("Agg")
    timeline_path = report_dir / "timeline.png"
    timeline_plot(config, timeline_path)
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


def main(argv) -> None:
    """The command line: one figure name and the run folders it reads."""
    if len(argv) == 5 and argv[1] == "latency":
        sample_files = _sample_files(argv[2:4])
        out_path = Path(argv[4])
        combined_latency_plot(sample_files, out_path)
        print(argv[4])
        return
    if len(argv) == 6 and argv[1] == "ler_vs_d":
        probability = float(argv[4])
        out_path = Path(argv[5])
        ler_vs_distance_plot(argv[2:4], probability, out_path)
        print(argv[5])
        return
    if len(argv) == 4 and argv[1] == "stage_breakdown":
        out_path = Path(argv[3])
        stage_breakdown_plot(argv[2], out_path)
        print(argv[3])
        return
    print(USAGE, file=sys.stderr)
    raise SystemExit(2)


@dataclasses.dataclass(frozen=True)
class _SweepPoint:
    """The sweep point the timeline draws: the figure's first point."""

    physical_error_probability: float
    distance: int
    round_period_us: float


@dataclasses.dataclass(frozen=True)
class _TimelineLanes:
    """Which link each hop of the timeline reads, and the store's name.

    The escalation kind picks the wires: weak_baseline moves windows on
    weak_buffer_to_weak_decoder, strong_only on
    strong_buffer_to_strong_decoder.
    """

    input_path: str
    output_path: str
    store_path: str
    store_name: str


@dataclasses.dataclass(frozen=True)
class _WindowHops:
    """One shot's per-window lookups, keyed by window id."""

    input_link: dict
    decoder_handoff: dict
    output_link: dict
    frame_records: dict
    stages: object


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


def _first_sweep_point(config) -> _SweepPoint:
    """The first point of the first sweep block: the timeline's shot."""
    block = config.sweep[0]
    return _SweepPoint(
        physical_error_probability=block.physical_error_probabilities[0],
        distance=block.distances[0],
        round_period_us=block.round_periods_microseconds[0],
    )


def _timeline_lanes(escalation_kind: str) -> _TimelineLanes:
    """The links and the store name this escalation kind moves data on."""
    store_path = "controller_to_weak_buffer"
    store_name = "buffer 0"
    if escalation_kind == "strong_only":
        store_path = "controller_to_strong_buffer"
        store_name = "syndrome buffer 1"
    return _TimelineLanes(
        input_path=measure.INPUT_LINK[escalation_kind],
        output_path=measure.OUTPUT_LINK[escalation_kind],
        store_path=store_path,
        store_name=store_name,
    )


def _transfers_by_round(transfers: list, path_name: str) -> dict:
    """The last transfer on one path per round index it carried."""
    selected = {}
    for transfer in transfers:
        if transfer["path"] != path_name:
            continue
        round_index = transfer["attribution"]["round_lo"]
        selected[round_index] = transfer
    return selected


def _transfers_by_window(transfers: list, path_name: str) -> dict:
    """The last transfer on one path per window id it carried."""
    selected = {}
    for transfer in transfers:
        if transfer["path"] != path_name:
            continue
        window_id = transfer["attribution"]["window_id"]
        selected[window_id] = transfer
    return selected


def _window_hops(completed, transfers: list, lanes: _TimelineLanes):
    """Every per-window lookup the window bars read."""
    frame_snapshot = completed.pauli_frame.snapshot()
    frame_records = {}
    for record in frame_snapshot.records:
        frame_records[record.window_key[1]] = record
    input_link = _transfers_by_window(transfers, lanes.input_path)
    handoff = _transfers_by_window(transfers, "decoder_to_decoder")
    output_link = _transfers_by_window(transfers, lanes.output_path)
    return _WindowHops(
        input_link=input_link,
        decoder_handoff=handoff,
        output_link=output_link,
        frame_records=frame_records,
        stages=completed.observation.stages,
    )


def _timeline_windows(completed) -> dict:
    """Window id -> the run's window record, in window order."""
    ledger = completed.observation.windows
    items = ledger.windows.items()
    ordered = sorted(items)
    windows = {}
    for key, window in ordered:
        windows[key[1]] = window
    return windows


def _timeline_rows(lanes: _TimelineLanes, store: dict) -> list:
    """The timeline's row labels, top to bottom, one per hop."""
    rows = ["qpu round", "qc link"]
    if store:
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


def _draw_rounds(
    timeline: _TimelineAxes,
    lanes: _TimelineLanes,
    round_period_us: float,
    qpu_link: dict,
    store: dict,
) -> None:
    """Every round's QPU time, its QC hop, and its hop into the store."""
    for round_number in sorted(qpu_link):
        shade = "0.75"
        if round_number % 2:
            shade = "0.55"
        sent = ticks_to_microseconds(qpu_link[round_number]["send_ticks"])
        round_start = sent - round_period_us
        timeline.round_bar("qpu round", round_start, round_period_us, shade)
        delivered = ticks_to_microseconds(
            qpu_link[round_number]["delivery_ticks"]
        )
        link_time = delivered - sent
        timeline.round_bar("qc link", sent, link_time, shade)
        if round_number not in store:
            continue
        store_sent = ticks_to_microseconds(store[round_number]["send_ticks"])
        store_delivered = ticks_to_microseconds(
            store[round_number]["delivery_ticks"]
        )
        store_time = store_delivered - store_sent
        timeline.round_bar(
            f"{lanes.store_name} link", store_sent, store_time, shade
        )


def _draw_windows(
    timeline: _TimelineAxes,
    lanes: _TimelineLanes,
    hops: _WindowHops,
    windows: dict,
    stored_row: dict,
    rounds: int,
) -> list:
    """Every decoded window's stages, and the legend text for each."""
    window_ranges = []
    for window_id, window in windows.items():
        if window.t_done is None:
            continue
        color = WINDOW_COLORS[window_id % len(WINDOW_COLORS)]
        read_hi = min(window.buffer_hi, rounds)
        range_text = _window_range_text(window, window_id, read_hi)
        window_ranges.append(range_text)
        _draw_store_fill(
            timeline, lanes, window, window_id, color, stored_row, rounds
        )
        stored_tick = ticks_to_microseconds(
            stored_row[read_hi]["delivery_ticks"]
        )
        dispatched = ticks_to_microseconds(window.t_dispatch)
        timeline.window_bar("wait", stored_tick, dispatched, window_id, color)
        _draw_window_transfers(timeline, lanes, hops, window_id, color)
        _draw_window_stages(timeline, hops, window_id, color)
    return window_ranges


def _window_range_text(window, window_id: int, read_hi: int) -> str:
    """The legend line for one window: what it commits and what it reads."""
    return (
        f"window {window_id}: "
        f"commits {window.commit_lo}-{window.commit_hi}, "
        f"reads {window.start_round}-{read_hi}"
    )


def _draw_store_fill(
    timeline: _TimelineAxes,
    lanes: _TimelineLanes,
    window,
    window_id: int,
    color: str,
    stored_row: dict,
    rounds: int,
) -> None:
    """The window's rounds landing in the store it reads.

    Commit rounds land solid; the trailing buffer reads land lighter.
    """
    row = f"{lanes.store_name} fill"
    first_round = ticks_to_microseconds(window.t_first_round)
    last_commit_round = min(window.commit_hi, rounds)
    commit_stored = ticks_to_microseconds(
        stored_row[last_commit_round]["delivery_ticks"]
    )
    timeline.window_bar(row, first_round, commit_stored, window_id, color)
    read_hi = min(window.buffer_hi, rounds)
    if read_hi <= window.commit_hi:
        return
    stored_tick = ticks_to_microseconds(stored_row[read_hi]["delivery_ticks"])
    timeline.window_bar(
        row, commit_stored, stored_tick, window_id, color, alpha=0.45
    )


def _draw_window_transfers(
    timeline: _TimelineAxes,
    lanes: _TimelineLanes,
    hops: _WindowHops,
    window_id: int,
    color: str,
) -> None:
    """The window's input link, boundary handoff and output link bars."""
    _draw_transfer_bar(
        timeline,
        f"transfer ({lanes.input_path})",
        hops.input_link,
        window_id,
        color,
    )
    _draw_transfer_bar(
        timeline, "dd handoff", hops.decoder_handoff, window_id, color
    )
    _draw_transfer_bar(
        timeline,
        f"{lanes.output_path} link",
        hops.output_link,
        window_id,
        color,
    )
    if window_id not in hops.frame_records:
        return
    record = hops.frame_records[window_id]
    accepted = ticks_to_microseconds(record.accepted_ticks)
    committed = ticks_to_microseconds(record.committed_ticks)
    timeline.window_bar("frame commit", accepted, committed, window_id, color)


def _draw_transfer_bar(
    timeline: _TimelineAxes,
    row: str,
    by_window: dict,
    window_id: int,
    color: str,
) -> None:
    """One link hop's bar, when the window crossed that link."""
    if window_id not in by_window:
        return
    transfer = by_window[window_id]
    sent = ticks_to_microseconds(transfer["send_ticks"])
    delivered = ticks_to_microseconds(transfer["delivery_ticks"])
    timeline.window_bar(row, sent, delivered, window_id, color)


def _draw_window_stages(
    timeline: _TimelineAxes, hops: _WindowHops, window_id: int, color: str
) -> None:
    """The decoder engine's fetch, algorithm and release bars."""
    stage_records = hops.stages.records_for(1, window_id)
    stages = {}
    for record in stage_records:
        stages[record.stage] = record
    for stage in ("fetch", "algorithm", "release"):
        if stage not in stages:
            continue
        record = stages[stage]
        started = ticks_to_microseconds(record.start_ticks)
        ended = ticks_to_microseconds(record.end_ticks)
        timeline.window_bar(stage, started, ended, window_id, color)


def _label_timeline(
    timeline: _TimelineAxes, config, point: _SweepPoint
) -> None:
    """The timeline's axes labels, title and geometry subtitle."""
    axis = timeline.axis
    axis.set_yticks(range(len(timeline.rows)))
    axis.set_yticklabels(timeline.rows, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("time from shot start (µs)")
    escalation_kind = config.settings.escalation.kind
    title = TIMELINE_TITLES.get(escalation_kind, config.name)
    axis.set_title(title, pad=22)
    subtitle = _timeline_subtitle(config, point)
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


def _timeline_subtitle(config, point: _SweepPoint) -> str:
    """The geometry line above the timeline: code, windows, rounds, card."""
    distance = point.distance
    commit_rounds = config.settings.windows.commit_rounds or distance
    buffer_rounds = config.settings.windows.buffer_rounds or distance
    code_task = config.settings.workload.code_task
    code_words = code_task.split(":")
    code_name = code_words[0]
    algorithm = config.active_decoder.kind
    algorithm_text = card_label(algorithm)
    return (
        f"d={distance} {code_name}"
        f" · {config.settings.windows.kind} windows: commit {commit_rounds},"
        f" buffer {buffer_rounds} rounds"
        f" · rounds every {point.round_period_us:g} µs"
        f" · algorithm {algorithm_text}"
        f" · p={point.physical_error_probability:g}"
    )


def _add_timeline_legend(
    timeline: _TimelineAxes, windows: dict, window_ranges: list
) -> None:
    """The legend: the rounds, one entry per window, and the lighter fill."""
    import matplotlib.pyplot as plt

    handles = [plt.Rectangle((0, 0), 1, 1, color="0.6")]
    labels = ["rounds"]
    window_ids = list(windows)
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


def _csv_rows(path) -> list:
    """Every row of one csv file, as dicts of text."""
    with open(path) as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _ler_rows_at_probability(run_dir, probability: float) -> list:
    """A run's ler.csv rows at one p, sorted by distance.

    A run that never swept that p is refused.
    """
    ler_path = Path(run_dir) / "ler.csv"
    all_rows = _csv_rows(ler_path)
    selected_rows = []
    for row in all_rows:
        row_probability = float(row["physical_error_probability"])
        if math.isclose(row_probability, probability):
            selected_rows.append(row)
    if not selected_rows:
        raise ValueError(f"{run_dir} swept no p={probability:g} point")
    selected_rows.sort(key=_by_distance_text)
    return selected_rows


def _by_distance_text(row: dict) -> int:
    return int(row["distance"])


def _rows_with_failures(rows: list, swept_distances: set) -> list:
    """The rows that saw at least one failure; every distance recorded."""
    measured_rows = []
    for row in rows:
        swept_distances.add(int(row["distance"]))
        if int(row["failures"]) > 0:
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
        raise FileNotFoundError(
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
    """Each run folder's latency_samples.csv."""
    paths = []
    for run_dir in run_dirs:
        samples_path = Path(run_dir) / "latency_samples.csv"
        paths.append(samples_path)
    return paths


if __name__ == "__main__":
    main(sys.argv)
