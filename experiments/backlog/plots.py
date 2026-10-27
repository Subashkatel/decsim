"""The backlog study's figures, from study.py's results folder.

    python plots.py RESULTS_FOLDER

with this checkout first on the path, as study.py.

Writes into RESULTS_FOLDER/plots/:
  - trajectories.png: seed 0's backlog over time at gamma = 1e-3 for
    tau_weak / tau_gen = 0.4, 0.7, 0.9 (Toshio et al. 2510.25222 Fig. 20),
    time in Fig. 20's gate steps of 9 d rounds;
  - mean_backlog.png: the mean backlog against gamma, a line per
    tau_weak (Fig. 21 b), the median over seeds with their range;
  - capacity.png: the last window's backlog over the stream's rounds
    against gamma, each point a stream, beside the queue's load - 1
    (config.load): an overloaded server's last commit lags the stream's
    end by load - 1 of its length, a stable one's by none. Fig. 21 a's
    divergence threshold is marked per tau_weak;
  - weak_only.png: the same against tau_weak with no escalation, beside
    load - 1 and Eq. (7)'s tau_gen / 2.
"""

import collections
import csv
import pathlib
import statistics
import sys

import matplotlib

import decsim.experiments.sweeps as sweeps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
TRAJECTORY_GAMMA = 1e-3
TRAJECTORY_FRACTIONS = (0.4, 0.7, 0.9)
# Fig. 20's time step: one non-Clifford gate's r_op = 9 d rounds
GATE_STEP_ROUNDS = 9 * constants.distance
# the switching rate where Fig. 21 a's divergence probability turns to 1,
# read off the figure to the nearest grid line
PAPER_THRESHOLDS = {0.0: 2.5e-2, 0.1: 2e-2, 0.4: 5e-3, 0.7: 7e-4, 0.9: 1.5e-4}
COLORS = {
    0.0: "tab:blue",
    0.1: "tab:orange",
    0.4: "tab:green",
    0.7: "tab:red",
    0.9: "tab:purple",
}


def main(results: pathlib.Path) -> None:
    """Every figure of the folder."""
    folder = results / "plots"
    folder.mkdir(exist_ok=True)
    streams_path = results / "streams.csv"
    streams = _read(streams_path)
    switching = [row for row in streams if row["study"] == "switching"]
    weak_only = [row for row in streams if row["study"] == "weak_only"]
    trajectories_path = results / "trajectories.csv"
    trajectories = _read(trajectories_path)
    plot_trajectories(trajectories, folder)
    plot_mean_backlog(switching, folder)
    plot_capacity(switching, folder)
    plot_weak_only(weak_only, folder)


def plot_trajectories(rows: list, folder: pathlib.Path) -> None:
    """Fig. 20: backlog against gate steps, one line per tau_weak."""
    by_fraction = collections.defaultdict(list)
    for row in rows:
        if not _is_drawn_trajectory(row):
            continue
        fraction = _weak_fraction(row)
        committed_us = float(row["committed_us"])
        committed_rounds = committed_us / constants.round_period_microseconds
        step = committed_rounds / GATE_STEP_ROUNDS
        backlog = float(row["backlog_rounds"])
        by_fraction[fraction].append((step, backlog))
    figure, axes = plt.subplots(figsize=(5, 3.6))
    for fraction in sorted(by_fraction):
        points = sorted(by_fraction[fraction])
        steps = [step for step, _ in points]
        backlogs = [backlog for _, backlog in points]
        label = _fraction_label(fraction)
        color = COLORS[fraction]
        axes.plot(steps, backlogs, ".", ms=2, color=color, label=label)
    axes.set_yscale("log")
    axes.set_xlabel(f"time steps ({GATE_STEP_ROUNDS} rounds each)")
    axes.set_ylabel("backlog [rounds]")
    axes.set_title(f"$\\gamma_{{switch}}$ = {TRAJECTORY_GAMMA:g}, d = 21")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(markerscale=4)
    _save(figure, folder, "trajectories.png")


def _is_drawn_trajectory(row: dict) -> bool:
    """A switching window at Fig. 20's gamma and one of its tau_weak."""
    if row["study"] != "switching":
        return False
    if float(row["gamma"]) != TRAJECTORY_GAMMA:
        return False
    fraction = _weak_fraction(row)
    return fraction in TRAJECTORY_FRACTIONS


def plot_mean_backlog(rows: list, folder: pathlib.Path) -> None:
    """Fig. 21 b: mean backlog against gamma, median and range of seeds."""
    figure, axes = plt.subplots(figsize=(5, 3.6))
    grouped = _by_fraction_and_gamma(rows)
    for fraction, by_gamma in grouped.items():
        gammas = sorted(by_gamma)
        medians = []
        lows = []
        highs = []
        for gamma in gammas:
            means = [
                float(row["mean_backlog_rounds"]) for row in by_gamma[gamma]
            ]
            median = statistics.median(means)
            medians.append(median)
            lows.append(min(means))
            highs.append(max(means))
        color = COLORS[fraction]
        label = _fraction_label(fraction)
        axes.plot(gammas, medians, "o--", ms=3, color=color, label=label)
        axes.fill_between(gammas, lows, highs, color=color, alpha=0.2)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("switching rate $\\gamma_{switch}$ (per d rounds)")
    axes.set_ylabel("mean backlog [rounds]")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8)
    _save(figure, folder, "mean_backlog.png")


def plot_capacity(rows: list, folder: pathlib.Path) -> None:
    """The undecoded share at the end against gamma, beside the queue's."""
    figure, axes = plt.subplots(figsize=(5, 3.6))
    grouped = _by_fraction_and_gamma(rows)
    for fraction, by_gamma in grouped.items():
        color = COLORS[fraction]
        gammas = sorted(by_gamma)
        for gamma in gammas:
            shares = [_undecoded_share(row) for row in by_gamma[gamma]]
            repeated = [gamma] * len(shares)
            axes.plot(repeated, shares, ".", color=color, ms=3)
        theory = []
        for gamma in gammas:
            commit = config.commit_rounds(fraction)
            rate = config.window_rate(gamma, commit)
            load = config.load(fraction, rate, commit)
            share = _queue_share(load)
            theory.append(share)
        label = _fraction_label(fraction)
        axes.plot(gammas, theory, "-", color=color, lw=1, label=label)
        threshold = PAPER_THRESHOLDS[fraction]
        axes.axvline(threshold, color=color, ls=":", lw=1)
    axes.set_xscale("log")
    axes.set_xlabel("switching rate $\\gamma_{switch}$ (per d rounds)")
    axes.set_yscale("symlog", linthresh=0.01)
    axes.set_ylim(bottom=0.0)
    axes.set_ylabel("last backlog / stream rounds")
    axes.set_title("dots: decsim; lines: load - 1; dotted: Fig. 21a")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8)
    _save(figure, folder, "capacity.png")


def plot_weak_only(rows: list, folder: pathlib.Path) -> None:
    """No escalation: the undecoded share against tau_weak, Eq. (7)."""
    points = sorted(rows, key=_weak_fraction)
    fractions = [_weak_fraction(row) for row in points]
    shares = [_undecoded_share(row) for row in points]
    theory = []
    for row in points:
        load = float(row["load"])
        share = _queue_share(load)
        theory.append(share)
    figure, axes = plt.subplots(figsize=(5, 3.6))
    axes.plot(fractions, shares, "o", ms=4, label="decsim")
    axes.plot(fractions, theory, "-", lw=1, label="load - 1")
    axes.axvline(0.5, color="gray", ls=":", label="Eq. (7): $\\tau_{gen}$/2")
    axes.set_xlabel(
        "$\\tau^{weak}_{dec} / \\tau_{gen}$ ($r_{com} = r_{buf} = d$)"
    )
    axes.set_ylabel("last backlog / stream rounds")
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8)
    _save(figure, folder, "weak_only.png")


def _by_fraction_and_gamma(rows: list) -> dict:
    """Rows by tau_weak / tau_gen, then by gamma."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        fraction = _weak_fraction(row)
        gamma = float(row["gamma"])
        grouped[fraction][gamma].append(row)
    items = grouped.items()
    ordered = sorted(items)
    return dict(ordered)


def _queue_share(load: float) -> float:
    """The last commit's lag over the stream's length: load - 1, or 0."""
    if load <= 1.0:
        return 0.0
    return load - 1.0


def _undecoded_share(row: dict) -> float:
    """The last window's backlog over the rounds the stream measured."""
    final = float(row["final_backlog_rounds"])
    return final / constants.horizon_rounds


def _weak_fraction(row: dict) -> float:
    return float(row["weak_fraction"])


def _fraction_label(fraction: float) -> str:
    return f"$\\tau^{{weak}}_{{dec}}/\\tau_{{gen}}$ = {fraction:g}"


def _read(path: pathlib.Path) -> list:
    with path.open() as rows_file:
        reader = csv.DictReader(rows_file)
        return list(reader)


def _save(figure, folder: pathlib.Path, name: str) -> None:
    figure.tight_layout()
    path = folder / name
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    folder_argument = pathlib.Path(sys.argv[1])
    main(folder_argument)
