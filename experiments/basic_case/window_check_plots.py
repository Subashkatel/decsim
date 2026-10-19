"""Sliding windows against one window a shot, shot by shot.

    python window_check_plots.py CHECK_FOLDER PILOT_FOLDER

CHECK_FOLDER is window_check.py's results, PILOT_FOLDER pilot.py's: the
same seeds of the same datasets, decoded in windows of d rounds and as
one window a shot. Writes into CHECK_FOLDER/plots/:
  - window_check.csv: per run and point, each windowing's failures, the
    shots only one of them failed, and McNemar's exact two-sided p on
    those two counts (Skoric et al. 2209.08552 find the two "within the
    numerical error of each other");
  - window_check.png: the error rate per shot of each windowing against
    p, a panel per run, with 95% Wilson intervals.
"""

import collections
import csv
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROBABILITY = "workload.arguments.physical_error_probability"
DISTANCE = "qpu.distance"
DECODINGS = ("weak_alone", "strong_alone", "switching")
COLORS = {5: "tab:blue", 7: "tab:orange"}
FIELDS = (
    "decoding",
    "d",
    "p",
    "shots",
    "windowed_failures",
    "whole_failures",
    "only_windowed_failed",
    "only_whole_failed",
    "mcnemar_p",
)
# the z of a two-sided 95% interval
Z_95 = 1.959964


def main(check: pathlib.Path, pilot: pathlib.Path) -> None:
    """The table and the figure."""
    folder = check / "plots"
    folder.mkdir(exist_ok=True)
    whole_path = check / "shots.csv"
    windowed_path = pilot / "shots.csv"
    whole = _failures_by_run(whole_path)
    windowed = _failures_by_run(windowed_path)
    keys = sorted(whole)
    rows = [_row(key, windowed[key], whole[key]) for key in keys]
    write_table(rows, folder)
    plot_rates(rows, folder)


def write_table(rows: list, folder: pathlib.Path) -> None:
    """One row per run and point."""
    path = folder / "window_check.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_rates(rows: list, folder: pathlib.Path) -> None:
    """Each windowing's rate against p, a panel per run."""
    size = (5 * len(DECODINGS), 4)
    figure, axes_row = plt.subplots(1, len(DECODINGS), figsize=size)
    for axes, decoding in zip(axes_row, DECODINGS, strict=True):
        ours = [row for row in rows if row[0] == decoding]
        _draw_run(axes, ours)
        axes.set_title(decoding)
        axes.set_xlabel("physical error rate p")
        axes.set_ylabel("logical error rate per shot")
        axes.set_yscale("log")
        axes.grid(True, which="both", alpha=0.3)
    axes_row[0].legend(fontsize=7)
    figure.tight_layout()
    path = folder / "window_check.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def _draw_run(axes, rows: list) -> None:
    """Windows of d rounds solid, one window a shot dashed, a color per d."""
    by_distance = collections.defaultdict(list)
    for row in rows:
        by_distance[row[1]].append(row)
    distances = sorted(by_distance)
    for distance in distances:
        ordered = by_distance[distance]
        color = COLORS[distance]
        ordered.sort(key=lambda row: row[2])
        _draw_line(
            axes, ordered, 4, ("-", f"d={distance}, windows of d"), color
        )
        _draw_line(axes, ordered, 5, ("--", f"d={distance}, one a shot"), color)


def _draw_line(axes, rows: list, column: int, look: tuple, color) -> None:
    style, label = look
    probabilities = [row[2] for row in rows]
    intervals = [_wilson(row[column], row[3]) for row in rows]
    rates = [row[column] / row[3] for row in rows]
    below = [
        rate - low for rate, (low, _) in zip(rates, intervals, strict=True)
    ]
    above = [
        high - rate for rate, (_, high) in zip(rates, intervals, strict=True)
    ]
    errors = (below, above)
    marker = "o" + style
    axes.errorbar(
        probabilities, rates, yerr=errors, fmt=marker, color=color, label=label
    )


def _row(key: tuple, windowed: dict, whole: dict) -> tuple:
    """A run's two failure counts on the seeds both decoded."""
    shared = set(windowed) & set(whole)
    seeds = sorted(shared)
    windowed_failures = sum(windowed[seed] for seed in seeds)
    whole_failures = sum(whole[seed] for seed in seeds)
    only_windowed = sum(windowed[seed] and not whole[seed] for seed in seeds)
    only_whole = sum(whole[seed] and not windowed[seed] for seed in seeds)
    p_value = _mcnemar(only_windowed, only_whole)
    counts = (windowed_failures, whole_failures, only_windowed, only_whole)
    return (*key, len(seeds), *counts, round(p_value, 4))


def _mcnemar(only_first: int, only_second: int) -> float:
    """McNemar's exact two-sided p: the discordant pairs a fair coin."""
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    fewer = min(only_first, only_second)
    last = fewer + 1
    counts = range(last)
    tail = sum(math.comb(discordant, k) for k in counts)
    both_sides = 2 * tail / 2**discordant
    return min(1.0, both_sides)


def _wilson(failures: int, shots: int) -> tuple:
    """The Wilson score interval of failures / shots at 95%."""
    rate = failures / shots
    z_squared = Z_95**2
    centre = rate + z_squared / (2 * shots)
    variance = rate * (1 - rate) / shots + z_squared / (4 * shots**2)
    spread = Z_95 * math.sqrt(variance)
    scale = 1 + z_squared / shots
    return ((centre - spread) / scale, (centre + spread) / scale)


def _failures_by_run(path: pathlib.Path) -> dict:
    """{(decoding, d, p): {seed: failed}} of a shots.csv."""
    runs = collections.defaultdict(dict)
    with path.open() as rows_file:
        for row in csv.DictReader(rows_file):
            distance = int(row[DISTANCE])
            probability = float(row[PROBABILITY])
            key = (row["decoding"], distance, probability)
            seed = int(row["seed"])
            runs[key][seed] = row["logical_failure"] == "True"
    return runs


if __name__ == "__main__":
    check_folder = pathlib.Path(sys.argv[1])
    pilot_folder = pathlib.Path(sys.argv[2])
    main(check_folder, pilot_folder)
