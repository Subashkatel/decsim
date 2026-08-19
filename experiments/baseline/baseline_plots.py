"""The baseline's figures, one question each, from the sweep rows.

keep_up.png        Can it keep up?   output rate / input rate vs input round time
reaction.png       Reaction time     window ready -> frame commit, median solid, p99 dashed
backlog_<card>.png Backlog           three panels (safe, near limit, overloaded)
one_window.png     One window        where the time goes, from window ready
ler.png            logical error rate vs physical error rate, when p was swept

Every time is in microseconds. The x axis of the sweep figures is the input
round time as categories, slow to fast from left to right.
"""

from pathlib import Path

# One window's path after it is ready, in order: (label, points summed, kind).
ONE_WINDOW_STAGES = (
    ("Wait", ("dep_block", "queue_wait"), "waiting"),
    ("Transfer", ("cwd_per_window",), "service"),
    ("Decode", ("fetch", "algorithm", "release"), "service"),
    ("Commit", ("wdo_per_window", "frame_commit"), "service"),
)

WAITING_COLOR = "tab:orange"
SERVICE_COLOR = "tab:blue"


def card_label(algorithm) -> str:
    """Legend name of a decoder card: 0.028 µs, 0.28 µs, Measured."""
    if isinstance(algorithm, str):
        return algorithm.capitalize()
    return f"{algorithm:g} µs"


def card_file_name(algorithm) -> str:
    if isinstance(algorithm, str):
        return algorithm
    return f"{algorithm:g}us"


def rows_of_card(rows: list, algorithm) -> list:
    """This card's rows at the lowest physical error rate, slow input first."""
    lowest_p = min(row["physical_error_probability"] for row in rows)
    selected = []
    for row in rows:
        if row["algorithm_latency_us"] != algorithm:
            continue
        if row["physical_error_probability"] != lowest_p:
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: row["round_period_us"], reverse=True)


def round_times(rows: list) -> list:
    """The swept input round times, slow to fast."""
    return sorted({row["round_period_us"] for row in rows}, reverse=True)


def category_axis(axis, times: list) -> None:
    """Input round time as evenly spaced categories, slow to fast."""
    positions = list(range(len(times)))
    labels = [f"{time:g}" for time in times]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Input round time (µs)")


def values_by_time(group: list, times: list, key) -> list:
    """One value per round time, in the order of `times`; key is a column name or a function of the row."""
    by_time = {}
    for row in group:
        if callable(key):
            by_time[row["round_period_us"]] = key(row)
        else:
            by_time[row["round_period_us"]] = row[key]
    values = []
    for time in times:
        values.append(by_time.get(time))
    return values


def output_over_input(row: dict) -> float:
    """Rounds committed per us over rounds arriving per us: 1 means the loop keeps up."""
    input_rounds_per_us = 1 / row["round_period_us"]
    return row["throughput_rounds_per_us"] / input_rounds_per_us


def keep_up_plot(rows: list, algorithms: list, path: Path) -> None:
    """Output rate over input rate against input round time: 1 is keeping up,
    below 1 the loop falls behind."""
    import matplotlib.pyplot as plt
    times = round_times(rows)
    positions = list(range(len(times)))
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    for algorithm in algorithms:
        group = rows_of_card(rows, algorithm)
        ratio = values_by_time(group, times, output_over_input)
        axis.plot(positions, ratio, "o-", label=card_label(algorithm))
    axis.axhline(1.0, color="k", ls="--", lw=0.8, label="Keeps up")
    category_axis(axis, times)
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Output rate / input rate")
    axis.set_title("Can it keep up?")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def reaction_plot(rows: list, algorithms: list, path: Path) -> None:
    """Reaction time (window ready -> frame commit) against input round time:
    median solid, p99 dashed, one color per decoder card."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    times = round_times(rows)
    positions = list(range(len(times)))
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    for algorithm in algorithms:
        group = rows_of_card(rows, algorithm)
        median = values_by_time(group, times, "buffer0_ready_to_frame_median_us")
        p99 = values_by_time(group, times, "buffer0_ready_to_frame_p99_us")
        line, = axis.plot(positions, median, "o-", label=card_label(algorithm))
        axis.plot(positions, p99, "--", color=line.get_color(), alpha=0.8)
    category_axis(axis, times)
    axis.set_ylabel("Reaction time (µs)")
    axis.set_title("Reaction time")
    axis.grid(alpha=0.3, axis="y")
    card_legend = axis.legend(fontsize=8, loc="upper left")
    axis.add_artist(card_legend)
    style_handles = [Line2D([], [], color="k", ls="-", marker="o"), Line2D([], [], color="k", ls="--")]
    axis.legend(style_handles, ["Median", "p99"], fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def operating_points(group: list) -> list:
    """Three rows of one card: the safest (lowest load), the one nearest
    load 1, and the most overloaded."""
    by_load = sorted(group, key=lambda row: row["load"])
    safest = by_load[0]
    nearest_limit = min(group, key=lambda row: abs(row["load"] - 1.0))
    most_overloaded = by_load[-1]
    points = []
    for row in (safest, nearest_limit, most_overloaded):
        if row not in points:
            points.append(row)
    return points


def backlog_plot(rows: list, algorithm, path: Path) -> None:
    """One card: backlog (windows ready but not decoded) over one shot at
    three input round times, safe to overloaded."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    group = rows_of_card(rows, algorithm)
    points = operating_points(group)
    figure, axes = plt.subplots(1, len(points), figsize=(3.4 * len(points), 2.8), sharey=True)
    axes = list(axes) if len(points) > 1 else [axes]
    for axis, row in zip(axes, points):
        times = []
        depths = []
        for time_us, depth in row["backlog_trajectory"]:
            times.append(time_us)
            depths.append(depth)
        axis.step(times, depths, where="post", color=SERVICE_COLOR)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(f"{row['round_period_us']:g} µs")
        axis.set_xlabel("Time (µs)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Backlog (windows)")
    figure.suptitle(f"Backlog, {card_label(algorithm)} decoder")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def one_window_plot(rows: list, algorithms: list, path: Path) -> None:
    """Where one window's time goes after it is ready, at the input round
    time nearest the limit: Wait, Transfer, Decode, Commit (mean per window)."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(algorithms), figsize=(3.6 * len(algorithms), 2.6), sharey=True)
    axes = list(axes) if len(algorithms) > 1 else [axes]
    stage_labels = []
    for label, _, _ in ONE_WINDOW_STAGES:
        stage_labels.append(label)
    for axis, algorithm in zip(axes, algorithms):
        group = rows_of_card(rows, algorithm)
        row = min(group, key=lambda candidate: abs(candidate["load"] - 1.0))
        start = 0.0
        for stage_index, (label, points, kind) in enumerate(ONE_WINDOW_STAGES):
            duration = 0.0
            for point in points:
                duration += row[f"{point}_mean_us"]
            color = WAITING_COLOR if kind == "waiting" else SERVICE_COLOR
            axis.barh(stage_index, duration, left=start, color=color, height=0.6)
            axis.text(start + duration, stage_index, f" {duration:.1f}", va="center", fontsize=7)
            start += duration
        axis.set_yticks(range(len(stage_labels)))
        axis.set_yticklabels(stage_labels)
        axis.invert_yaxis()
        axis.set_title(f"{card_label(algorithm)}, {row['round_period_us']:g} µs rounds", fontsize=9)
        axis.set_xlabel("Time from window ready (µs)")
        axis.grid(alpha=0.3, axis="x")
    figure.suptitle("One window")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def ler_groups(rows: list, algorithms: list) -> list:
    """(card, round time, rows sorted by p) for every card and round time
    that swept more than one physical error rate."""
    periods = sorted({row["round_period_us"] for row in rows})
    groups = []
    for algorithm in algorithms:
        for period in periods:
            group = []
            for row in rows:
                if row["algorithm_latency_us"] == algorithm and row["round_period_us"] == period:
                    group.append(row)
            probabilities = {row["physical_error_probability"] for row in group}
            if len(probabilities) < 2:
                continue
            group.sort(key=lambda row: row["physical_error_probability"])
            groups.append((algorithm, period, group))
    return groups


def ler_plot(rows: list, algorithms: list, path: Path) -> None:
    """Logical error rate against physical error rate, Wilson 95% bars, one
    line per card and round time that swept more than one p."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    groups = ler_groups(rows, algorithms)
    plotted_periods = {period for _, period, _ in groups}
    for algorithm, period, group in groups:
        probabilities = []
        rates = []
        lower = []
        upper = []
        for row in group:
            probabilities.append(row["physical_error_probability"])
            rates.append(row["logical_error_rate"])
            lower.append(row["logical_error_rate"] - row["ler_wilson_low"])
            upper.append(row["ler_wilson_high"] - row["logical_error_rate"])
        if len(plotted_periods) == 1:
            label = card_label(algorithm)
        else:
            label = f"{card_label(algorithm)}, {period:g} µs"
        axis.errorbar(probabilities, rates, yerr=[lower, upper], fmt="o-", capsize=3, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Physical error rate")
    axis.set_ylabel("Logical error rate")
    axis.set_title("Logical error rate")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def plots(rows: list, report_dir: Path) -> None:
    """The sweep figures when more than one round time was swept, the LER
    figure when more than one physical error rate was."""
    import matplotlib
    matplotlib.use("Agg")
    algorithms = sorted({row["algorithm_latency_us"] for row in rows}, key=str)
    periods = {row["round_period_us"] for row in rows}
    probabilities = {row["physical_error_probability"] for row in rows}
    if len(periods) > 1:
        keep_up_plot(rows, algorithms, report_dir / "keep_up.png")
        reaction_plot(rows, algorithms, report_dir / "reaction.png")
        for algorithm in algorithms:
            backlog_plot(rows, algorithm, report_dir / f"backlog_{card_file_name(algorithm)}.png")
        one_window_plot(rows, algorithms, report_dir / "one_window.png")
    if len(probabilities) > 1:
        ler_plot(rows, algorithms, report_dir / "ler.png")
