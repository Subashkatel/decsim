"""The walkthrough's figures, one question per figure, from rows.json.

throughput.png         Does the output rate follow the input rate?
reaction_latency.png   What does a window's reaction time cost as input speeds up?
backlog.png            What does falling behind look like over a shot?
latency_breakdown.png  Where does a window's reaction time go when keeping up?

Style follows the real-time decoding papers (Google 2408.13687, LILLIPUT,
SWIPER): one quantity per figure, medians with an explicit tail percentile,
axes labeled with units, the decoder's limit drawn where it applies.

    PYTHONPATH=. python guide/walkthrough/frequency_plots.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"

KNEE_COLOR = "tab:red"
DATA_COLOR = "tab:blue"
WAITING_COLOR = "tab:orange"

# The reaction time (window ready in Buffer 0 -> correction in the frame),
# split into the sweep's per-window points, in path order.
BREAKDOWN_STAGES = (
    ("Dependency wait", ("dep_block_mean_us",)),
    ("Queue wait", ("queue_wait_mean_us",)),
    ("Transfer (cwd)", ("cwd_per_window_mean_us",)),
    ("Fetch", ("fetch_mean_us",)),
    ("Algorithm", ("algorithm_mean_us",)),
    ("Release", ("release_mean_us",)),
    ("Commit (wdo+frame)", ("wdo_per_window_mean_us", "frame_commit_mean_us")),
)


def load_results() -> tuple:
    payload = json.loads((RESULTS / "rows.json").read_text())
    rows = sorted(payload["rows"], key=lambda row: row["round_period_us"], reverse=True)
    return payload["commit_rounds"], rows


def input_frequency(row: dict) -> float:
    """Input rounds per microsecond: one syndrome round every round period."""
    return 1.0 / row["round_period_us"]


def knee_frequency(rows: list, commit_rounds: int) -> float:
    """The decoder's limit, in input rounds per microsecond: one window costs
    service + dd handoff and carries commit_rounds rounds. Measured at the
    slowest point, where nothing queues."""
    slowest = rows[0]
    window_cost_us = slowest["service_mean_us"] + slowest["dd_per_window_mean_us"]
    return commit_rounds / window_cost_us


def new_figure() -> tuple:
    figure, axis = plt.subplots(figsize=(4.6, 3.4))
    axis.grid(alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure, axis


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=200)
    plt.close(figure)
    print(FIGURES / name)


def mark_knee(axis, knee: float) -> None:
    axis.axvline(knee, color=KNEE_COLOR, ls=":", lw=1.2)
    axis.text(knee, 0.03, f" decoder limit\n {knee:.2f} rounds/µs",
              transform=axis.get_xaxis_transform(), color=KNEE_COLOR, fontsize=8, va="bottom")


def throughput_plot(rows: list, knee: float) -> None:
    """Committed rounds per microsecond against input rounds per microsecond:
    on the input line while the decoder keeps up, flat at its limit beyond."""
    frequencies = [input_frequency(row) for row in rows]
    committed = [row["throughput_rounds_per_us"] for row in rows]
    figure, axis = new_figure()
    axis.plot(frequencies, frequencies, ls="--", color="0.6", lw=1.0, label="output = input")
    axis.plot(frequencies, committed, "o-", color=DATA_COLOR, ms=4, label="measured")
    mark_knee(axis, knee)
    axis.set_xlabel("Input round frequency (rounds/µs)")
    axis.set_ylabel("Committed rounds (rounds/µs)")
    axis.set_title("Decoded throughput", fontsize=11)
    axis.legend(fontsize=8, loc="upper left")
    save(figure, "throughput.png")


def latency_plot(rows: list, knee: float) -> None:
    """Reaction time of a window (ready in Buffer 0 -> correction in the
    frame) against input round frequency: flat below the decoder's limit,
    growing without bound above it. Log scale so both regimes stay readable."""
    frequencies = [input_frequency(row) for row in rows]
    median = [row["buffer0_ready_to_frame_median_us"] for row in rows]
    p99 = [row["buffer0_ready_to_frame_p99_us"] for row in rows]
    figure, axis = new_figure()
    axis.plot(frequencies, median, "o-", color=DATA_COLOR, ms=4, label="median")
    axis.plot(frequencies, p99, "s--", color=DATA_COLOR, ms=4, alpha=0.6, label="p99")
    mark_knee(axis, knee)
    axis.set_yscale("log")
    axis.set_xlabel("Input round frequency (rounds/µs)")
    axis.set_ylabel("Reaction time (µs)")
    axis.set_title("Reaction latency", fontsize=11)
    axis.legend(fontsize=8, loc="upper left")
    save(figure, "reaction_latency.png")


def operating_points(rows: list) -> list:
    """Three rows: well below the decoder's limit, just past it, and well
    above it. At the limit itself the backlog still looks bounded over one
    shot; the growth only becomes visible once load clears 1."""
    safest = rows[0]
    just_past = next((row for row in rows if row["load"] > 1.02), rows[-1])
    most_overloaded = rows[-1]
    points = []
    for row in (safest, just_past, most_overloaded):
        if row not in points:
            points.append(row)
    return points


def backlog_plot(rows: list) -> None:
    """Backlog (windows ready in Buffer 0 but not yet decoded) over one shot
    at three input speeds: bounded below the limit, growing without bound
    above it."""
    points = operating_points(rows)
    figure, axes = plt.subplots(1, len(points), figsize=(3.1 * len(points), 2.8), sharey=True)
    for axis, row in zip(axes, points):
        times = [time_us for time_us, depth in row["backlog_trajectory"]]
        depths = [depth for time_us, depth in row["backlog_trajectory"]]
        axis.step(times, depths, where="post", color=DATA_COLOR, lw=1.0)
        axis.set_title(f"{input_frequency(row):.2f} rounds/µs (load {row['load']:.2f})",
                       fontsize=9)
        axis.set_xlabel("Time (µs)")
        axis.grid(alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Backlog (windows)")
    figure.suptitle("Decoder backlog over one shot", fontsize=11)
    save(figure, "backlog.png")


def breakdown_plot(rows: list) -> None:
    """Where one window's reaction time goes at a safe input speed: every
    point of the path from ready in Buffer 0 to the frame commit."""
    safe = rows[0]
    figure, axis = new_figure()
    start_us = 0.0
    for stage_index, (label, columns) in enumerate(BREAKDOWN_STAGES):
        duration_us = sum(safe[column] for column in columns)
        waiting = "wait" in label.lower()
        color = WAITING_COLOR if waiting else DATA_COLOR
        axis.barh(stage_index, duration_us, left=start_us, color=color, height=0.6)
        axis.text(start_us + duration_us + 0.02, stage_index, f"{duration_us:.3f}",
                  va="center", fontsize=8)
        start_us += duration_us
    axis.set_yticks(range(len(BREAKDOWN_STAGES)))
    axis.set_yticklabels([label for label, _ in BREAKDOWN_STAGES], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, start_us * 1.18)      # room for the duration labels
    axis.set_xlabel("Time from window ready (µs)")
    axis.set_title(f"One window's reaction time\n({input_frequency(safe):.2f} rounds/µs input)",
                   fontsize=10)
    save(figure, "latency_breakdown.png")


def main() -> None:
    commit_rounds, rows = load_results()
    knee = knee_frequency(rows, commit_rounds)
    throughput_plot(rows, knee)
    latency_plot(rows, knee)
    backlog_plot(rows)
    breakdown_plot(rows)


if __name__ == "__main__":
    main()
