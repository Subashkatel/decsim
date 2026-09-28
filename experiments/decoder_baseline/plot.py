"""The decoder baseline's figures, drawn from its stats.csv.

`python plot.py <results folder>` writes the folder's plots/: for each
decoder, the logical error rate per round against the physical error
rate, a curve per distance, X and Z side by side; and for each basis,
the four decoders compared, a panel per distance. The rate per round is
sinter's (failure_units_per_shot_func, sinter/_plotting.py), the
per-round rate the Tesseract paper reports (Beni et al. 2503.10988,
eq. 8); the shading is sinter's likelihood band around each point.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot  # noqa: E402
import run  # noqa: E402
import sinter  # noqa: E402

PLOTS_FOLDER = "plots"
STATS_FILE = "stats.csv"
BASIS_TITLES = {"x": "memory X", "z": "memory Z"}
COMPARISON_ROWS = 2
COMPARISON_COLUMNS = 3
DOTS_PER_INCH = 150
X_MARGIN_BELOW = 0.8
X_MARGIN_ABOVE = 1.25


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's stats.csv, into its plots/."""
    stats_path = folder / STATS_FILE
    stats = sinter.read_stats_from_csv_files(stats_path)
    plots = folder / PLOTS_FOLDER
    plots.mkdir(exist_ok=True)
    for decoder in run.DECODERS:
        figure = decoder_figure(stats, decoder)
        path = plots / f"{decoder}.png"
        save(figure, path)
    for basis in run.BASES:
        figure = comparison_figure(stats, basis)
        path = plots / f"decoders_{basis}.png"
        save(figure, path)


def save(figure: pyplot.Figure, path: pathlib.Path) -> None:
    """One figure written and closed, so a run of them holds one open."""
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def decoder_figure(stats: list, decoder: str) -> pyplot.Figure:
    """One decoder: a curve per distance, a panel per basis."""
    figure, axes = pyplot.subplots(1, len(run.BASES), sharey=True)
    figure.set_size_inches(11, 4.5)
    for axis, basis in zip(axes, run.BASES, strict=True):
        points = chosen(stats, decoder=decoder, basis=basis)
        draw(axis, points, curve_of=distance_of)
        axis.set_title(BASIS_TITLES[basis])
        axis.legend(title="d")
    figure.suptitle(decoder)
    return figure


def comparison_figure(stats: list, basis: str) -> pyplot.Figure:
    """One basis: a curve per decoder, a panel per distance."""
    figure, axes = pyplot.subplots(
        COMPARISON_ROWS, COMPARISON_COLUMNS, sharex=True, sharey=True
    )
    figure.set_size_inches(14, 8)
    for axis, distance in zip(axes.flat, run.DISTANCES, strict=True):
        points = chosen(stats, basis=basis, distance=distance)
        draw(axis, points, curve_of=decoder_of)
        axis.set_title(f"d = {distance}")
        axis.legend()
    title = BASIS_TITLES[basis]
    figure.suptitle(f"{title}, decoders compared")
    return figure


def chosen(stats: list, **wanted) -> list:
    """The points whose decoder, basis or distance are the wanted ones."""
    points = []
    for point in stats:
        found = {
            "decoder": point.decoder,
            "basis": point.json_metadata["basis"],
            "distance": point.json_metadata["d"],
        }
        is_wanted = all(found[key] == value for key, value in wanted.items())
        if is_wanted:
            points.append(point)
    return points


def draw(axis: pyplot.Axes, points: list, curve_of) -> None:
    """Logical error rate per round against p, log on both axes."""
    sinter.plot_error_rate(
        ax=axis,
        stats=points,
        x_func=physical_error_rate_of,
        failure_units_per_shot_func=rounds_of,
        group_func=curve_of,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    # every panel spans the swept rates, so a panel still filling reads
    # on the same axis as a full one
    lowest_rate = min(run.ERROR_RATES)
    highest_rate = max(run.ERROR_RATES)
    left = lowest_rate * X_MARGIN_BELOW
    right = highest_rate * X_MARGIN_ABOVE
    axis.set_xlim(left, right)
    axis.set_xlabel("physical error rate p")
    axis.set_ylabel("logical error rate per round")
    axis.grid(which="major")
    axis.grid(which="minor", alpha=0.3)


def physical_error_rate_of(point: sinter.TaskStats) -> float:
    """The x of a point."""
    return point.json_metadata["p"]


def rounds_of(point: sinter.TaskStats) -> int:
    """The rounds a shot holds, which turn a shot's rate into a round's."""
    del point
    return run.ROUNDS


def distance_of(point: sinter.TaskStats) -> int:
    """A point's curve in a decoder's figure."""
    return point.json_metadata["d"]


def decoder_of(point: sinter.TaskStats) -> str:
    """A point's curve in a comparison figure."""
    return point.decoder


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
