"""Plots of the baseline sweep rows (see baseline_closed_loop.py for the rows)."""

from pathlib import Path

from experiments.baseline.baseline_closed_loop import algorithm_label

# Where a window's reaction time goes, in four categories (components.png).
COMPONENTS = {
    "dependency wait": ("dep_block", "queue_wait"),
    "data collection": ("buffer_fill",),
    "decoder compute": ("fetch", "algorithm", "release"),
    "other classical path": ("cwd_per_window", "wdo_per_window", "frame_commit"),
}


# ---- the plots -------------------------------------------------------------

def rows_of_algorithm(rows: list, algorithm) -> list:
    """This algorithm's rows at the lowest physical error probability, slowest input first."""
    lowest_p = min(row["physical_error_probability"] for row in rows)
    selected = [row for row in rows
                if row["algorithm_latency_us"] == algorithm and row["physical_error_probability"] == lowest_p]
    return sorted(selected, key=lambda row: row["round_period_us"], reverse=True)


def ler_plot(rows: list, algorithms: list, path: Path) -> None:
    """Logical error rate against physical error probability, one curve per
    (decoder card, round period), Wilson 95% bars."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    periods = sorted({row["round_period_us"] for row in rows})
    for algorithm in algorithms:
        for period in periods:
            group = sorted((row for row in rows
                            if row["algorithm_latency_us"] == algorithm and row["round_period_us"] == period),
                           key=lambda row: row["physical_error_probability"])
            probabilities = [row["physical_error_probability"] for row in group]
            rates = [row["logical_error_rate"] for row in group]
            lower = [row["logical_error_rate"] - row["ler_wilson_low"] for row in group]
            upper = [row["ler_wilson_high"] - row["logical_error_rate"] for row in group]
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


def input_rate_mhz(row: dict) -> float:
    return 1 / row["round_period_us"]


def reaction_vs_rate_plot(rows: list, algorithms: list, path: Path) -> None:
    """End-to-end window reaction time (first round -> correction in the frame)
    against the syndrome input rate: mean solid, p99 dashed, one color per card."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    for algorithm in algorithms:
        group = rows_of_algorithm(rows, algorithm)
        rates = [input_rate_mhz(row) for row in group]
        means = [row["reaction_first_round_mean_us"] for row in group]
        p99s = [row["reaction_first_round_p99_us"] for row in group]
        line, = axis.plot(rates, means, "o-", label=f"{algorithm_label(algorithm)} decoder, mean")
        axis.plot(rates, p99s, "--", color=line.get_color(), alpha=0.7, label=f"{algorithm_label(algorithm)} decoder, p99")
    axis.set_xscale("log")
    axis.set_xlabel("syndrome input rate (MHz)")
    axis.set_ylabel("window reaction time, first round -> frame (us)")
    axis.set_title("Reaction time vs input rate (mean solid, p99 dashed)", fontsize=9)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def components_plot(rows: list, algorithms: list, path: Path) -> None:
    """Where the reaction time goes, per card: four categories as curves from zero."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(algorithms), figsize=(4.2 * len(algorithms), 3.6), sharey=True)
    axes = list(axes) if len(algorithms) > 1 else [axes]
    for axis, algorithm in zip(axes, algorithms):
        group = rows_of_algorithm(rows, algorithm)
        rates = [input_rate_mhz(row) for row in group]
        for name, points in COMPONENTS.items():
            values = []
            for row in group:
                values.append(sum(row[f"{point}_mean_us"] for point in points))
            axis.plot(rates, values, "o-", label=name)
        axis.set_xscale("log")
        axis.set_title(f"{algorithm_label(algorithm)} decoder")
        axis.set_xlabel("syndrome input rate (MHz)")
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("mean us per window")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def reaction_vs_load_plot(rows: list, algorithms: list, path: Path) -> None:
    """Reaction time against the chain load rho; rho = 1 is where the serial
    chain stops keeping up with window arrivals."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    for algorithm in algorithms:
        group = rows_of_algorithm(rows, algorithm)
        loads = [row["load"] for row in group]
        means = [row["reaction_first_round_mean_us"] for row in group]
        axis.plot(loads, means, "o-", label=f"{algorithm_label(algorithm)} decoder")
    axis.axvline(1.0, color="k", lw=0.8, ls=":", label="rho = 1 (chain saturated)")
    axis.set_xscale("log")
    axis.set_xlabel("chain load rho = service per window / window inter-arrival")
    axis.set_ylabel("window reaction time, first round -> frame (us)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def throughput_plot(rows: list, algorithms: list, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5, 3.6))
    for algorithm in algorithms:
        group = rows_of_algorithm(rows, algorithm)
        input_rounds_per_us = [input_rate_mhz(row) for row in group]
        decoded_rounds_per_us = [row["throughput_rounds_per_us"] for row in group]
        axis.plot(input_rounds_per_us, decoded_rounds_per_us, "o-",
                  label=f"{algorithm_label(algorithm)} decoder")
    fastest_input = max(input_rate_mhz(row) for row in rows)
    axis.plot([0, fastest_input], [0, fastest_input], "k--", lw=0.8, label="keeps up (out = in)")
    axis.set_xlabel("input rounds/us")
    axis.set_ylabel("decoded rounds/us")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def plots(rows: list, report_dir: Path) -> None:
    """Timing plots when more than one round period was swept, the LER plot
    when more than one physical error probability was."""
    import matplotlib
    matplotlib.use("Agg")
    algorithms = sorted({row["algorithm_latency_us"] for row in rows}, key=str)
    periods = {row["round_period_us"] for row in rows}
    probabilities = {row["physical_error_probability"] for row in rows}
    if len(periods) > 1:
        reaction_vs_rate_plot(rows, algorithms, report_dir / "reaction_vs_rate.png")
        components_plot(rows, algorithms, report_dir / "components.png")
        reaction_vs_load_plot(rows, algorithms, report_dir / "reaction_vs_load.png")
        throughput_plot(rows, algorithms, report_dir / "throughput.png")
    if len(probabilities) > 1:
        ler_plot(rows, algorithms, report_dir / "ler.png")


