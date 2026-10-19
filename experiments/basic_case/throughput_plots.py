"""The outer loop's figures and table, accuracy read off the sweep.

    python throughput_plots.py THROUGHPUT_FOLDER SWEEP_FOLDER

THROUGHPUT_FOLDER is throughput.py's results, SWEEP_FOLDER
threshold_sweep.py's. Writes into THROUGHPUT_FOLDER/plots/:
  - backlog.png: each control's backlog in rounds against the round,
    seed 0, a row per study and a column per d, the change dotted;
  - share.png: each control's escalated share of every hundred windows,
    the seeds' mean;
  - throughput.csv: per study, control and d, the seeds' mean shares
    before and after the change, the backlog after it, and the error
    rate per shot the sweep measured at each share. Between two swept
    shares the rate is the line between them: a controller holding a
    share between two steps of the gap's staircase escalates at the two
    thresholds in turn, so it fails as the mix of the two.
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

throughput = sweeps.load_beside(__file__, "throughput.py")
PROBABILITY = "workload.arguments.physical_error_probability"
DISTANCE = "qpu.distance"
STUDIES = ("steady", "p_rises", "strong_slows")
# the p of each study before the change and after it
PROBABILITIES = {
    "steady": (0.002, 0.002),
    "p_rises": (0.002, 0.003),
    "strong_slows": (0.002, 0.002),
}
COLORS = {
    "fixed_0.05": "tab:green",
    "fixed_0.1": "tab:olive",
    "fixed_0.15": "tab:orange",
    "fixed_0.2": "tab:red",
    "fixed_20dB": "tab:red",
    "loop": "tab:blue",
    "loop_scaled": "tab:purple",
}
FIELDS = (
    "study",
    "control",
    "d",
    "seeds",
    "share_before",
    "share_after",
    "mean_backlog_after_rounds",
    "max_backlog_rounds",
    "error_rate_before",
    "error_rate_after",
)
SAMPLE_WINDOWS = 100


def main(folder: pathlib.Path, sweep: pathlib.Path) -> None:
    """Both figures and the table."""
    out = folder / "plots"
    out.mkdir(exist_ok=True)
    streams_path = folder / "streams.csv"
    windows_path = folder / "windows.csv"
    sweep_path = sweep / "sweep.csv"
    streams = _read(streams_path)
    windows = _read(windows_path)
    swept = _read(sweep_path)
    frontier = _frontier(swept)
    write_table(streams, frontier, out)
    plot_backlog(windows, out)
    plot_shares(windows, out)


def write_table(streams: list, frontier: dict, out: pathlib.Path) -> None:
    """One row per study, control and d."""
    grouped = collections.defaultdict(list)
    for row in streams:
        key = (row["study"], row["control"], int(row["d"]))
        grouped[key].append(row)
    keys = sorted(grouped)
    rows = [_table_row(key, grouped[key], frontier) for key in keys]
    path = out / "throughput.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"wrote {path}")


def _table_row(key: tuple, rows: list, frontier: dict) -> tuple:
    """The seeds' means, and the sweep's rate at each mean share."""
    study, _control, distance = key
    before = _mean(rows, "share_before")
    after = _mean(rows, "share_after")
    backlog = _mean(rows, "mean_backlog_after_rounds")
    worst = max(float(row["max_backlog_rounds"]) for row in rows)
    first_p, later_p = PROBABILITIES[study]
    rate_before = _rate_at(frontier[(distance, first_p)], before)
    rate_after = _rate_at(frontier[(distance, later_p)], after)
    shares = (round(before, 4), round(after, 4))
    backlogs = (round(backlog, 1), round(worst, 1))
    rates = (round(rate_before, 5), round(rate_after, 5))
    return (*key, len(rows), *shares, *backlogs, *rates)


def plot_backlog(windows: list, out: pathlib.Path) -> None:
    """Seed 0's backlog against the round, every control."""
    figure, axes_grid = _grid()
    seed_zero = [row for row in windows if row["seed"] == "0"]
    for (row, column), axes in _cells(axes_grid):
        study = STUDIES[row]
        distance = throughput.DISTANCES[column]
        for control, rounds, backlogs in _series(seed_zero, study, distance):
            color = COLORS[control]
            axes.plot(rounds, backlogs, lw=0.7, color=color, label=control)
        _finish(axes, study, distance, "backlog (rounds)")
        axes.set_yscale("log")
    path = out / "backlog.png"
    _save(figure, axes_grid, path)


def plot_shares(windows: list, out: pathlib.Path) -> None:
    """The escalated share of every hundred windows, the seeds' mean."""
    figure, axes_grid = _grid()
    for (row, column), axes in _cells(axes_grid):
        study = STUDIES[row]
        distance = throughput.DISTANCES[column]
        for control, rounds, shares in _share_series(windows, study, distance):
            color = COLORS[control]
            axes.plot(rounds, shares, lw=1, color=color, label=control)
        _finish(axes, study, distance, "windows escalated")
    path = out / "share.png"
    _save(figure, axes_grid, path)


def _series(windows: list, study: str, distance: int) -> list:
    """(control, rounds, backlogs) of each control's stream."""
    by_control = _by_control(windows, study, distance)
    series = []
    for control in sorted(by_control):
        rows = by_control[control]
        rounds = [float(row["committed_us"]) for row in rows]
        backlogs = [float(row["backlog_rounds"]) for row in rows]
        series.append((control, rounds, backlogs))
    return series


def _share_series(windows: list, study: str, distance: int) -> list:
    """(control, rounds, shares) of each hundred windows, seeds averaged."""
    by_control = _by_control(windows, study, distance)
    series = []
    for control in sorted(by_control):
        blocks = _blocks(by_control[control])
        numbers = sorted(blocks)
        block_rounds = SAMPLE_WINDOWS * distance
        # a block is drawn at its last round
        ends = [number + 1 for number in numbers]
        rounds = [end * block_rounds for end in ends]
        shares = [statistics.fmean(blocks[number]) for number in numbers]
        series.append((control, rounds, shares))
    return series


def _by_control(windows: list, study: str, distance: int) -> dict:
    """The study's windows at d, by control."""
    by_control = collections.defaultdict(list)
    for row in windows:
        if row["study"] == study and int(row["d"]) == distance:
            by_control[row["control"]].append(row)
    return by_control


def _blocks(rows: list) -> dict:
    """Whether each window escalated, by its hundred windows."""
    blocks = collections.defaultdict(list)
    for row in rows:
        block = int(row["window"]) // SAMPLE_WINDOWS
        escalated = row["tier"] == "strong"
        blocks[block].append(escalated)
    return blocks


def _frontier(swept: list) -> dict:
    """{(d, p): [(share, error rate)]} of the sweep, by share."""
    frontier = collections.defaultdict(list)
    for row in swept:
        point = (int(row[DISTANCE]), float(row[PROBABILITY]))
        share = float(row["escalated_fraction"])
        rate = float(row["logical_error_rate_estimate"])
        frontier[point].append((share, rate))
    return {point: sorted(pairs) for point, pairs in frontier.items()}


def _rate_at(pairs: list, share: float) -> float:
    """The sweep's rate at share, on the line between its neighbours."""
    neighbours = zip(pairs, pairs[1:], strict=False)
    for low, high in neighbours:
        if low[0] <= share <= high[0]:
            return _between(low, high, share)
    lowest, highest = pairs[0][0], pairs[-1][0]
    raise ValueError(
        f"share {share} is outside the sweep's {lowest} to {highest}"
    )


def _between(low: tuple, high: tuple, share: float) -> float:
    """The rate on the line from low to high at share."""
    low_share, low_rate = low
    high_share, high_rate = high
    span = high_share - low_share
    if span == 0:
        return low_rate
    weight = (share - low_share) / span
    return low_rate + weight * (high_rate - low_rate)


def _grid() -> tuple:
    rows, columns = len(STUDIES), len(throughput.DISTANCES)
    size = (5 * columns, 3.2 * rows)
    figure, axes_grid = plt.subplots(rows, columns, figsize=size, sharex=True)
    return figure, axes_grid


def _cells(axes_grid) -> list:
    return [
        ((row, column), axes)
        for row, axes_row in enumerate(axes_grid)
        for column, axes in enumerate(axes_row)
    ]


def _finish(axes, study: str, distance: int, label: str) -> None:
    axes.axvline(throughput.CHANGE_ROUND, color="grey", ls=":", lw=0.8)
    axes.set_title(f"{study}, d = {distance}")
    axes.set_ylabel(label)
    axes.set_xlabel("round")
    axes.grid(True, alpha=0.3)


def _save(figure, axes_grid, path: pathlib.Path) -> None:
    for axes_row in axes_grid:
        axes_row[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def _mean(rows: list, column: str) -> float:
    return statistics.fmean(float(row[column]) for row in rows)


def _read(path: pathlib.Path) -> list:
    with path.open() as rows_file:
        reader = csv.DictReader(rows_file)
        return list(reader)


if __name__ == "__main__":
    throughput_folder = pathlib.Path(sys.argv[1])
    sweep_folder = pathlib.Path(sys.argv[2])
    main(throughput_folder, sweep_folder)
