"""The baseline's figures, from the sweep rows (see baseline_closed_loop.py).

1. throughput.png: committed syndrome rate vs incoming syndrome rate, with
   the y = x line; where the curve leaves the line the loop saturates.
2. reaction.png: reaction time (window ready -> correction in the frame)
   vs incoming rate, median and p99, one curve per decoder card.
3. backlog.png: windows ready but not decoded, over the shot, at three
   operating points (safe, near saturation, overloaded) chosen from the
   measured load.
4. window_timeline.png: one window's path as bars: waiting steps and
   service steps, with the microseconds.
5. ler.png when more than one physical error probability was swept.
"""

from pathlib import Path

from experiments.baseline.baseline_closed_loop import algorithm_label

# A window's path in order: (label, point, kind). Waiting is time spent
# waiting on something else; service is time spent doing the step.
WINDOW_PATH = (
    ("buffer fill (waiting on the QPU)", "buffer_fill", "waiting"),
    ("dependency wait (previous window)", "dep_block", "waiting"),
    ("queue wait (free unit)", "queue_wait", "waiting"),
    ("CWD into decoder memory", "cwd_per_window", "service"),
    ("fetch", "fetch", "service"),
    ("algorithm", "algorithm", "service"),
    ("release", "release", "service"),
    ("WDO to Pauli frame", "wdo_per_window", "service"),
    ("frame commit", "frame_commit", "service"),
)


def rows_of_algorithm(rows: list, algorithm) -> list:
    """This algorithm's rows at the lowest physical error probability, slowest input first."""
    lowest_p = min(row["physical_error_probability"] for row in rows)
    selected = []
    for row in rows:
        if row["algorithm_latency_us"] != algorithm:
            continue
        if row["physical_error_probability"] != lowest_p:
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: row["round_period_us"], reverse=True)


def input_rate_mhz(row: dict) -> float:
    return 1 / row["round_period_us"]


def throughput_plot(rows: list, algorithms: list, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    for algorithm in algorithms:
        group = rows_of_algorithm(rows, algorithm)
        input_rates = []
        output_rates = []
        for row in group:
            input_rates.append(input_rate_mhz(row))
            output_rates.append(row["throughput_rounds_per_us"])
        axis.plot(input_rates, output_rates, "o-", label=f"{algorithm_label(algorithm)} decoder")
    fastest_input = max(input_rate_mhz(row) for row in rows)
    axis.plot([0.5, fastest_input], [0.5, fastest_input], "k--", lw=0.8, label="keeps up (out = in)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("incoming syndrome rate (MHz)")
    axis.set_ylabel("committed syndrome rate (MHz)")
    axis.set_title("Can the loop keep up?", fontsize=10)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def reaction_plot(rows: list, algorithms: list, path: Path) -> None:
    """Reaction time = frame commit - window ready (the last_round_to_frame point)."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    for algorithm in algorithms:
        group = rows_of_algorithm(rows, algorithm)
        rates = []
        medians = []
        p99s = []
        for row in group:
            rates.append(input_rate_mhz(row))
            medians.append(row["last_round_to_frame_median_us"])
            p99s.append(row["last_round_to_frame_p99_us"])
        line, = axis.plot(rates, medians, "o-", label=f"{algorithm_label(algorithm)} decoder, median")
        axis.plot(rates, p99s, "--", color=line.get_color(), alpha=0.7,
                  label=f"{algorithm_label(algorithm)} decoder, p99")
    axis.set_xscale("log")
    axis.set_xlabel("incoming syndrome rate (MHz)")
    axis.set_ylabel("reaction time, window ready -> frame (us)")
    axis.set_title("Reaction time vs input rate", fontsize=10)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def operating_points(group: list) -> list:
    """Three rows of one decoder card: the safest (lowest load), the one
    nearest load 1, and the most overloaded."""
    by_load = sorted(group, key=lambda row: row["load"])
    safest = by_load[0]
    nearest_saturation = min(group, key=lambda row: abs(row["load"] - 1.0))
    most_overloaded = by_load[-1]
    points = []
    for row in (safest, nearest_saturation, most_overloaded):
        if row not in points:
            points.append(row)
    return points


def backlog_plot(rows: list, algorithms: list, path: Path) -> None:
    """Windows ready but not decoded over one shot: one panel per operating
    point (safe, near saturation, overloaded) per decoder card, each with
    its own time axis. A bounded trace keeps up; a rising one does not."""
    import matplotlib.pyplot as plt
    columns = 3
    figure, axes = plt.subplots(len(algorithms), columns, figsize=(4.0 * columns, 2.6 * len(algorithms)),
                                squeeze=False)
    for row_index, algorithm in enumerate(algorithms):
        group = rows_of_algorithm(rows, algorithm)
        points = operating_points(group)
        for column_index in range(columns):
            axis = axes[row_index][column_index]
            if column_index >= len(points):
                axis.axis("off")
                continue
            row = points[column_index]
            times = []
            depths = []
            for time_us, depth in row["backlog_trajectory"]:
                times.append(time_us)
                depths.append(depth)
            axis.step(times, depths, where="post", color="tab:blue")
            axis.set_title(f"{algorithm_label(algorithm)} decoder, {input_rate_mhz(row):g} MHz, load {row['load']:.2f}",
                           fontsize=8)
            axis.set_xlabel("simulated time (us)", fontsize=8)
            axis.grid(alpha=0.3)
        axes[row_index][0].set_ylabel("windows ready,\nnot yet decoded", fontsize=8)
    figure.suptitle("Backlog over one shot: safe, near saturation, overloaded", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def window_timeline_plot(rows: list, algorithms: list, path: Path) -> None:
    """One window's path as a Gantt chart (mean per-window times at the
    operating point nearest saturation), one row per step: waiting steps in
    orange, service steps in blue, the duration written at the bar's end."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(algorithms), figsize=(4.6 * len(algorithms), 3.4), sharey=True)
    axes = list(axes) if len(algorithms) > 1 else [axes]
    step_labels = [label for label, _, _ in WINDOW_PATH]
    for axis, algorithm in zip(axes, algorithms):
        group = rows_of_algorithm(rows, algorithm)
        row = min(group, key=lambda candidate: abs(candidate["load"] - 1.0))
        start = 0.0
        for step_index, (label, point, kind) in enumerate(WINDOW_PATH):
            duration = row[f"{point}_mean_us"]
            color = "tab:orange" if kind == "waiting" else "tab:blue"
            axis.barh(step_index, duration, left=start, color=color, height=0.6)
            axis.text(start + duration, step_index, f" {duration:.2f}", va="center", fontsize=6)
            start += duration
        axis.set_yticks(range(len(step_labels)))
        axis.set_yticklabels(step_labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_title(f"{algorithm_label(algorithm)} decoder, {input_rate_mhz(row):g} MHz: "
                       f"{start:.1f} us first round -> frame", fontsize=8)
        axis.set_xlabel("us after the window's first round", fontsize=8)
        axis.grid(alpha=0.3, axis="x")
    figure.suptitle("One window's path (orange waiting, blue service)", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def ler_plot(rows: list, algorithms: list, path: Path) -> None:
    """Logical error rate against physical error probability, one curve per
    (decoder card, round period), Wilson 95% bars."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    periods = sorted({row["round_period_us"] for row in rows})
    for algorithm in algorithms:
        for period in periods:
            group = []
            for row in rows:
                if row["algorithm_latency_us"] == algorithm and row["round_period_us"] == period:
                    group.append(row)
            group.sort(key=lambda row: row["physical_error_probability"])
            probabilities = []
            rates = []
            lower = []
            upper = []
            for row in group:
                probabilities.append(row["physical_error_probability"])
                rates.append(row["logical_error_rate"])
                lower.append(row["logical_error_rate"] - row["ler_wilson_low"])
                upper.append(row["ler_wilson_high"] - row["logical_error_rate"])
            axis.errorbar(probabilities, rates, yerr=[lower, upper], fmt="o-", capsize=3,
                          label=f"{algorithm_label(algorithm)} decoder, {period:g} us round")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("physical error probability")
    axis.set_ylabel("logical error rate per shot")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def plots(rows: list, report_dir: Path) -> None:
    """The four timing figures when more than one round period was swept,
    the LER figure when more than one physical error probability was."""
    import matplotlib
    matplotlib.use("Agg")
    algorithms = sorted({row["algorithm_latency_us"] for row in rows}, key=str)
    periods = {row["round_period_us"] for row in rows}
    probabilities = {row["physical_error_probability"] for row in rows}
    if len(periods) > 1:
        throughput_plot(rows, algorithms, report_dir / "throughput.png")
        reaction_plot(rows, algorithms, report_dir / "reaction.png")
        backlog_plot(rows, algorithms, report_dir / "backlog.png")
        window_timeline_plot(rows, algorithms, report_dir / "window_timeline.png")
    if len(probabilities) > 1:
        ler_plot(rows, algorithms, report_dir / "ler.png")
