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
               `python -m experiments.plots latency <run_dir> <run_dir>
               <out.png>` reading each run's latency_samples.csv
ler vs d       both tiers' logical error rate against code distance at
               one physical error rate, from each run's ler.csv:
               `python -m experiments.plots ler_vs_d <run_dir> <run_dir>
               <p> <out.png>`

Every time is in microseconds.
"""

import math
import sys
from pathlib import Path

from decsim.config import TICKS_PER_US

from experiments.build_run import build_run
from experiments.experiment_config import ExperimentConfig
from experiments.measure_shot import INPUT_LINK, OUTPUT_LINK

WINDOW_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red",
                 "tab:purple", "tab:brown", "tab:pink", "tab:olive")
MAX_LEGEND_WINDOWS = 8


def us(ticks) -> float:
    return ticks / TICKS_PER_US


def card_label(algorithm) -> str:
    if isinstance(algorithm, str):
        return algorithm.capitalize()
    return f"{algorithm:g} µs"


# ---- the window timeline ---------------------------------------------------

def timeline_plot(config: ExperimentConfig, path: Path) -> None:
    """One shot at the first sweep point, seed 0, every stage in time."""
    import matplotlib.pyplot as plt

    block = config.sweep[0]
    physical_error_probability = block.physical_error_probabilities[0]
    distance = block.distances[0]
    round_period_us = block.round_periods_us[0]
    algorithm = config.active_decoder.algorithm
    spec, engine = build_run(
        config, physical_error_probability=physical_error_probability,
        distance=distance, round_period_us=round_period_us, seed=0)
    completed = spec.build()

    input_path = INPUT_LINK[config.mode]
    output_path = OUTPUT_LINK[config.mode]
    store_path = "csb" if config.mode == "strong_only" else "cwb"
    store_name = ("syndrome buffer 1" if config.mode == "strong_only"
                  else "buffer 0")

    transfers = completed.result.link_traffic["transfers"]

    def on(path_name, by_round):
        selected = {}
        for transfer in transfers:
            if transfer["path"] != path_name:
                continue
            key = (transfer["attribution"]["round_lo"] if by_round
                   else transfer["attribution"]["window_id"])
            selected[key] = transfer
        return selected

    qc = on("qc", by_round=True)
    store = on(store_path, by_round=True)
    input_link = on(input_path, by_round=False)
    dd = on("dd", by_round=False)
    output_link = on(output_path, by_round=False)
    windows = {window_id: window for (_, window_id), window
               in sorted(completed.window_manager.windows.items())}
    frame_records = {record.window_key[1]: record
                     for record in completed.pauli_frame.snapshot().records}
    rounds = config.rounds_per_shot.rounds_for(distance)

    rows = ["qpu round", "qc link"]
    if store:
        rows.append(f"{store_name} link")
    rows += [f"{store_name} fill", "wait", f"transfer ({input_path})",
             "fetch", "algorithm", "release", "dd handoff",
             f"{output_path} link", "frame commit"]
    row_index = {name: index for index, name in enumerate(rows)}

    figure, axis = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.6))
    for round_number in sorted(qc):
        sent = us(qc[round_number]["send_ticks"])
        shade = "0.55" if round_number % 2 else "0.75"
        axis.barh(row_index["qpu round"], round_period_us,
                  left=sent - round_period_us, color=shade, height=0.55)
        axis.barh(row_index["qc link"],
                  us(qc[round_number]["delivery_ticks"]) - sent,
                  left=sent, color=shade, height=0.55)
        if round_number in store:
            axis.barh(row_index[f"{store_name} link"],
                      us(store[round_number]["delivery_ticks"])
                      - us(store[round_number]["send_ticks"]),
                      left=us(store[round_number]["send_ticks"]),
                      color=shade, height=0.55)

    # the round's landing tick in the store the windows read
    stored_row = store if store else qc
    window_ranges = []
    lane_count = max(len(windows), 3)
    for window_id, window in windows.items():
        if window.t_done is None:
            continue
        color = WINDOW_COLORS[window_id % len(WINDOW_COLORS)]

        def bar(row, start, end, window_id=window_id, color=color, alpha=1.0):
            lane = (row_index[row]
                    + (window_id - (lane_count - 1) / 2) * (0.5 / lane_count))
            axis.barh(lane, end - start, left=start, color=color,
                      height=0.17, alpha=alpha)

        stages = {record.stage: record
                  for record in engine.stage_records_for(1, window_id)}
        read_hi = min(window.buffer_hi, rounds)
        stored_tick = us(stored_row[read_hi]["delivery_ticks"])
        window_ranges.append(
            f"window {window_id}: "
            f"commits {window.commit_lo}-{window.commit_hi}, "
            f"reads {window.start_round}-{read_hi}")
        # commit rounds land solid; the trailing buffer reads land lighter
        commit_stored = us(
            stored_row[min(window.commit_hi, rounds)]["delivery_ticks"])
        bar(f"{store_name} fill", us(window.t_first_round), commit_stored)
        if read_hi > window.commit_hi:
            bar(f"{store_name} fill", commit_stored, stored_tick, alpha=0.45)
        bar("wait", stored_tick, us(window.t_dispatch))
        if window_id in input_link:
            bar(f"transfer ({input_path})",
                us(input_link[window_id]["send_ticks"]),
                us(input_link[window_id]["delivery_ticks"]))
        for stage in ("fetch", "algorithm", "release"):
            if stage in stages:
                bar(stage, us(stages[stage].start_ticks),
                    us(stages[stage].end_ticks))
        if window_id in dd:
            bar("dd handoff", us(dd[window_id]["send_ticks"]),
                us(dd[window_id]["delivery_ticks"]))
        if window_id in output_link:
            bar(f"{output_path} link",
                us(output_link[window_id]["send_ticks"]),
                us(output_link[window_id]["delivery_ticks"]))
        if window_id in frame_records:
            bar("frame commit",
                us(frame_records[window_id].accepted_ticks),
                us(frame_records[window_id].committed_ticks))

    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(rows, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("time from shot start (µs)")
    timeline_titles = {"weak_baseline": "Weak only path timeline",
                       "strong_only": "Strong only path timeline",
                       "switching": "Switching path timeline"}
    axis.set_title(timeline_titles.get(config.mode, config.name), pad=22)
    commit_rounds = config.windowing.commit_rounds or distance
    buffer_rounds = config.windowing.buffer_rounds or distance
    axis.text(0.5, 1.005,
              f"d={distance} {config.code_task.split(':')[0]}"
              f" · {config.windowing.scheme} windows: commit {commit_rounds},"
              f" buffer {buffer_rounds} rounds"
              f" · rounds every {round_period_us:g} µs"
              f" · algorithm {card_label(algorithm)}"
              f" · p={physical_error_probability:g}",
              transform=axis.transAxes, ha="center", va="bottom",
              fontsize=8.5, color="0.35")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.6")]
    labels = ["rounds"]
    for window_id in list(windows)[:MAX_LEGEND_WINDOWS]:
        handles.append(plt.Rectangle(
            (0, 0), 1, 1, color=WINDOW_COLORS[window_id % len(WINDOW_COLORS)]))
        labels.append(window_ranges[window_id]
                      if window_id < len(window_ranges)
                      else f"window {window_id}")
    if len(windows) > MAX_LEGEND_WINDOWS:
        labels[-1] += "  (…)"
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.45))
    labels.append("lighter fill = buffer reads (not committed)")
    axis.legend(handles, labels, loc="lower left", fontsize=7.5)
    axis.grid(alpha=0.5, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


# ---- the logical error rate ------------------------------------------------

def ler_groups(rows: list) -> list:
    """(distance, round time, rows sorted by p) for every distance and
    round time that swept more than one physical error rate: the papers'
    convention, one LER curve per code distance."""
    distances = sorted({row["distance"] for row in rows})
    periods = sorted({row["round_period_us"] for row in rows})
    groups = []
    for distance in distances:
        for period in periods:
            group = [row for row in rows
                     if row["distance"] == distance
                     and row["round_period_us"] == period]
            probabilities = {row["physical_error_probability"] for row in group}
            if len(probabilities) < 2:
                continue
            group.sort(key=lambda row: row["physical_error_probability"])
            groups.append((distance, period, group))
    return groups


def decoder_title(config: ExperimentConfig) -> str:
    """"pymatching decoder (weak)" / "belief matching decoder (strong)".
    A numeric card reads as pymatching: the card prices latency but its
    corrections come from the same MWPM path."""
    from experiments.experiment_config import MODE_TIER
    algorithm = config.active_decoder.algorithm
    name = algorithm if isinstance(algorithm, str) else "pymatching"
    return f"{name.replace('_', ' ')} decoder ({MODE_TIER[config.mode]})"


def _power_of_ten_label(value: float) -> str:
    """5e-4 -> $5{\\times}10^{-4}$, 1e-3 -> $10^{-3}$: the y axis's
    notation."""
    exponent = math.floor(math.log10(value))
    mantissa = value / 10.0 ** exponent
    if math.isclose(mantissa, 1.0):
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}{{\\times}}10^{{{exponent}}}$"


def ler_plot(rows: list, path: Path,
             title: str = "Logical error rate") -> None:
    """Logical error rate against physical error rate, Wilson 95% bars, one
    line per card and round time that swept more than one p."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    groups = ler_groups(rows)
    plotted_periods = {period for _, period, _ in groups}
    for distance, period, group in groups:
        probabilities = [row["physical_error_probability"] for row in group]
        rates = [row["logical_error_rate"] for row in group]
        lower = [row["logical_error_rate"] - row["ler_wilson_low"]
                 for row in group]
        upper = [row["ler_wilson_high"] - row["logical_error_rate"]
                 for row in group]
        label = (f"d={distance}" if len(plotted_periods) == 1
                 else f"d={distance}, {period:g} µs")
        axis.errorbar(probabilities, rates, yerr=[lower, upper], fmt="o-",
                      capsize=3, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    # a decades-only log axis labels two of our seven p values; tick
    # every swept p, in the y axis's power-of-ten notation
    swept = sorted({row["physical_error_probability"] for row in rows})
    axis.set_xticks(swept)
    axis.set_xticklabels([_power_of_ten_label(probability)
                          for probability in swept], fontsize=8,
                         rotation=30, ha="right")
    from matplotlib.ticker import NullFormatter
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Physical error rate")
    axis.set_ylabel("Logical error rate")
    axis.set_title(title)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _csv_tier_label(algorithm_field: str) -> str:
    """"pymatching (weak)" / "belief matching (strong)" from ler.csv's
    algorithm column. A numeric card reads as pymatching, the same
    ruling as decoder_title."""
    try:
        float(algorithm_field)
        algorithm_name = "pymatching"
    except ValueError:
        algorithm_name = algorithm_field
    tier = "strong" if algorithm_name == "belief_matching" else "weak"
    display_name = algorithm_name.replace("_", " ")
    return f"{display_name} ({tier})"


def _ler_rows_at_probability(run_dir, probability: float) -> list:
    """A run's ler.csv rows at one physical error rate, sorted by
    distance; a run that never swept that p is refused."""
    import csv
    with open(Path(run_dir) / "ler.csv") as handle:
        all_rows = list(csv.DictReader(handle))
    selected_rows = []
    for row in all_rows:
        row_probability = float(row["physical_error_probability"])
        if math.isclose(row_probability, probability):
            selected_rows.append(row)
    if not selected_rows:
        raise ValueError(f"{run_dir} swept no p={probability:g} point")
    selected_rows.sort(key=lambda row: int(row["distance"]))
    return selected_rows


def _draw_measured_ler_points(axis, rows: list, color: str,
                              label: str) -> None:
    """The failures > 0 rows: a connected line with Wilson 95% bars."""
    distances = []
    rates = []
    bars_below = []
    bars_above = []
    for row in rows:
        rate = float(row["logical_error_rate"])
        distances.append(int(row["distance"]))
        rates.append(rate)
        bars_below.append(rate - float(row["ler_wilson_low"]))
        bars_above.append(float(row["ler_wilson_high"]) - rate)
    axis.errorbar(distances, rates, yerr=[bars_below, bars_above],
                  fmt="o-", capsize=3, color=color, label=label)


def ler_vs_distance_plot(run_dirs: list, probability: float,
                         path: Path) -> None:
    """Both tiers' logical error rate against code distance at one
    physical error rate, from each run's ler.csv. Measured points carry
    Wilson 95% bars; a zero-failure point cannot sit on a log axis, so
    its curve simply ends at the last distance that saw failures.

        python -m experiments.plots ler_vs_d <run_dir> <run_dir> <p>
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
        measured_rows = []
        for row in rows:
            swept_distances.add(int(row["distance"]))
            if int(row["failures"]) > 0:
                measured_rows.append(row)
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


# ---- the stage breakdown ---------------------------------------------------

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


def _shot_rows(run_dir) -> list:
    """Every row of the run's shots.csv; a run without one is refused."""
    import csv

    shots_path = Path(run_dir) / "shots.csv"
    if not shots_path.exists():
        raise FileNotFoundError(
            f"{run_dir} has no shots.csv; the stage breakdown reads the "
            f"per-shot stage means a closed-loop run records")
    with open(shots_path) as handle:
        return list(csv.DictReader(handle))


def _median_stage_us_by_distance(rows: list) -> dict:
    """distance -> [median us per stage, in STAGE_BREAKDOWN_STAGES order].

    The median is over shots of each shot's per-window mean, so one
    slow shot cannot move the bar the way a mean of means would let it.
    """
    import statistics

    samples_by_distance = {}
    for row in rows:
        distance = int(row["distance"])
        per_stage = samples_by_distance.setdefault(
            distance, [[] for _ in STAGE_BREAKDOWN_STAGES])
        for stage_index, (column, _) in enumerate(STAGE_BREAKDOWN_STAGES):
            per_stage[stage_index].append(float(row[column]))
    return {
        distance: [statistics.median(values) for values in per_stage]
        for distance, per_stage in sorted(samples_by_distance.items())
    }


def stage_breakdown_plot(run_dir, path: Path) -> None:
    """One stacked bar per distance: where a window's time goes, from
    syndrome arrival in the buffer to the Pauli-frame commit.

        python -m experiments.plots stage_breakdown <run_dir> <out.png>
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _shot_rows(run_dir)
    medians_by_distance = _median_stage_us_by_distance(rows)
    distances = list(medians_by_distance)
    # every breakdown is drawn in ms so the two tiers' figures share
    # one unit; the axis stays linear with plain tick numbers
    unit_divisor = 1000.0
    unit_name = "ms"
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    bar_positions = range(len(distances))
    stacked_left = [0.0] * len(distances)
    for stage_index, (_, stage_label) in enumerate(STAGE_BREAKDOWN_STAGES):
        stage_widths = [medians_by_distance[distance][stage_index] / unit_divisor
                        for distance in distances]
        axis.barh(bar_positions, stage_widths, left=stacked_left,
                  height=0.6, label=stage_label)
        stacked_left = [left + width
                        for left, width in zip(stacked_left, stage_widths)]
    for position, total in zip(bar_positions, stacked_left):
        label = f"{total:.3g}" if total < 100 else f"{total:,.0f}"
        axis.text(total, position, f"  {label}", va="center", fontsize=8)
    axis.set_yticks(list(bar_positions))
    axis.set_yticklabels([f"d={distance}" for distance in distances])
    axis.invert_yaxis()
    axis.set_xlim(0, max(stacked_left) * 1.12)
    axis.set_xlabel(f"median time per window ({unit_name})")
    breakdown_titles = {
        "pymatching": "Time breakdown: Weak decoder (pymatching)",
        "belief_matching": "Time breakdown: Strong decoder (belief matching)",
    }
    algorithm = rows[0]["algorithm"]
    title = breakdown_titles.get(
        algorithm, f"Time breakdown: {card_label(algorithm)}")
    axis.set_title(title)
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---- the decode latency ----------------------------------------------------

def latency_samples_by_distance(measurements: list) -> dict:
    """distance -> every window's algorithm-stage wall clock in us, pooled
    over shots. The algorithm stage alone, because that is the only
    measured quantity and the papers' comparable number (Helios Fig. 6,
    SWIPER Fig. 3); fetch, release and the links are priced from the
    config and belong to the stage-breakdown figure."""
    pooled = {}
    for measurement in measurements:
        pooled.setdefault(measurement.distance, []).extend(
            measurement.samples["algorithm"])
    return {distance: samples for distance, samples in sorted(pooled.items())
            if samples}


def _log_decade_axis(axis, log_values: list) -> None:
    """Label a log10-transformed time axis in plain microseconds. The
    violins are drawn on log10(us) values so their density is estimated
    in log space, where wall-clock latency is roughly symmetric; a raw
    linear KDE under a log axis would smear the tails."""
    lowest = math.floor(min(min(values) for values in log_values))
    highest = math.ceil(max(max(values) for values in log_values))
    ticks = list(range(lowest, highest + 1))
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{10.0 ** tick:g}" for tick in ticks])


def _latency_violins(axis, pooled: dict, positions: list, width: float,
                     color: str, label: str) -> None:
    """One violin per distance on log10(us) values, median marked."""
    log_samples = [[math.log10(sample) for sample in pooled[distance]]
                   for distance in pooled]
    parts = axis.violinplot(log_samples, positions=positions, widths=width,
                            showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.6)
    parts["cmedians"].set_color(color)
    maxima = [max(samples) for samples in log_samples]
    axis.plot(positions, maxima, "v", color=color, markersize=4,
              label=f"{label} (worst window marked)")


def _deadline_line(axis, distances: list, round_period_us: float) -> None:
    # the deadline: a new window arrives every d rounds (the code's
    # default commit region), so decode must beat d x round period
    deadline_log_us = [math.log10(distance * round_period_us)
                       for distance in distances]
    axis.plot(distances, deadline_log_us, "--", color="grey",
              label=f"window generation ({round_period_us:g} µs rounds)")


def latency_plot(config: ExperimentConfig, measurements: list,
                 path: Path) -> None:
    """Decode wall clock per window against code distance: one violin
    per d (median marked, worst window flagged), microsecond log axis,
    with the window-generation deadline drawn as the throughput
    boundary. The violin/deadline shape follows Helios Fig. 7, Google
    Fig. 4d and SWIPER Fig. 3."""
    import matplotlib.pyplot as plt
    pooled = latency_samples_by_distance(measurements)
    distances = list(pooled)
    round_period_us = measurements[0].round_period_us
    probability = measurements[0].physical_error_probability
    algorithm = config.active_decoder.algorithm

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    _latency_violins(axis, pooled, distances, 1.4, "C0", algorithm)
    _deadline_line(axis, distances, round_period_us)
    _log_decade_axis(axis, [[math.log10(sample) for sample in samples]
                            for samples in pooled.values()])
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
    """Both tiers on one axes from their runs' latency_samples.csv, one
    violin pair per distance. The log axis is what makes this legible:
    the tiers sit decades apart, which is itself the figure's message.

        python -m experiments.plots latency <run_dir> <run_dir> <out.png>
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    all_log_values = []
    round_period_us = None
    distances = []
    for file_index, sample_file in enumerate(sample_files):
        with open(sample_file) as handle:
            rows = list(csv.DictReader(handle))
        pooled = {}
        for row in rows:
            pooled.setdefault(int(row["distance"]), []).append(
                float(row["algorithm_us"]))
        pooled = dict(sorted(pooled.items()))
        distances = sorted(set(distances) | set(pooled))
        round_period_us = float(rows[0]["round_period_us"])
        algorithm = rows[0]["algorithm"]
        color = f"C{file_index}"
        _latency_violins(axis, pooled, list(pooled), 1.4, color, algorithm)
        all_log_values.extend(
            [math.log10(sample) for sample in samples]
            for samples in pooled.values())
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


def plots(config: ExperimentConfig, rows: list, report_dir: Path,
          measurements: list = None) -> None:
    """timeline.png always; ler.png when more than one p was swept;
    latency.png when more than one distance was swept with a wall-clock
    algorithm."""
    import matplotlib
    matplotlib.use("Agg")
    timeline_plot(config, report_dir / "timeline.png")
    if len({row["physical_error_probability"] for row in rows}) > 1:
        ler_plot(rows, report_dir / "ler.png", title=decoder_title(config))
    # a named algorithm charges measured wall clock; a numeric card is a
    # fixed latency, flat in d, so its figure would be a horizontal line
    measured_wall_clock = isinstance(config.active_decoder.algorithm, str)
    swept_distances = {measurement.distance
                       for measurement in measurements or ()}
    if measured_wall_clock and len(swept_distances) > 1:
        latency_plot(config, measurements, report_dir / "latency.png")


def main(argv) -> None:
    usage = ("usage: python -m experiments.plots "
             "latency <run_dir> <run_dir> <out.png>\n"
             "       python -m experiments.plots "
             "ler_vs_d <run_dir> <run_dir> <p> <out.png>\n"
             "       python -m experiments.plots "
             "stage_breakdown <run_dir> <out.png>")
    if len(argv) == 5 and argv[1] == "latency":
        sample_files = [Path(run_dir) / "latency_samples.csv"
                        for run_dir in argv[2:4]]
        combined_latency_plot(sample_files, Path(argv[4]))
        print(argv[4])
    elif len(argv) == 6 and argv[1] == "ler_vs_d":
        ler_vs_distance_plot(argv[2:4], float(argv[4]), Path(argv[5]))
        print(argv[5])
    elif len(argv) == 4 and argv[1] == "stage_breakdown":
        stage_breakdown_plot(argv[2], Path(argv[3]))
        print(argv[3])
    else:
        print(usage, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv)
