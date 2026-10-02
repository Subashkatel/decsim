"""The decoder baseline's figures, drawn from its stats.csv.

`python plot.py <results folder>` writes the folder's plots/: for each
decoder, the logical error rate per round against the physical error
rate, a curve per distance, a panel per basis; and for each basis, the
decoders compared, a panel per distance. The rate per round is the one
the Tesseract paper reports (Beni et al. 2503.10988, eq. 8).

The decoders, bases, distances and rates are the ones stats.csv holds,
each row's decoder and json_metadata, so a folder of any date is drawn
by this script without running the run.py it keeps.
"""

import pathlib
import sys

import matplotlib.axes
import sinter

import decsim.plots as plots

# Every baseline shot is a 100-round memory (run.py, ROUNDS), which
# stats.csv does not carry.
ROUNDS = 100


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's stats.csv, into its plots/."""
    stats_path = folder / "stats.csv"
    stats = sinter.read_stats_from_csv_files(stats_path)
    decoders = sorted({stat.decoder for stat in stats})
    bases = labels(stats, "basis")
    distances = labels(stats, "d")
    rates = labels(stats, "p")
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    for decoder in decoders:
        figure, axis_by_basis = plots.panels("basis", bases)
        for basis, axis in axis_by_basis.items():
            points = plots.chosen(stats, decoder=decoder, basis=basis)
            draw(axis, points, "d", distances, rates)
        figure.suptitle(decoder)
        path = plots_folder / f"{decoder}.png"
        plots.save(figure, path)
    for basis in bases:
        figure, axis_by_distance = plots.panels("d", distances)
        for distance, axis in axis_by_distance.items():
            points = plots.chosen(stats, basis=basis, d=distance)
            draw(axis, points, "decoder", decoders, rates)
        figure.suptitle(f"memory {basis}, decoders compared")
        path = plots_folder / f"decoders_{basis}.png"
        plots.save(figure, path)


def labels(stats: list, key: str) -> list:
    """Every value the stats hold under a json_metadata key, sorted."""
    return sorted({stat.json_metadata[key] for stat in stats})


def draw(
    axis: matplotlib.axes.Axes,
    points: list,
    curve: str,
    order: list,
    rates: list,
) -> None:
    """One panel: the rate per round against p, on the swept rates.

    order lists every value of curve, so a curve keeps its colour in a
    panel another has not reached yet. Every panel spans the rates of
    the whole stats.csv, so a panel with fewer points reads on the same
    axis as a full one, and carries a label at every rate, since a log
    axis over one decade shows one.
    """
    plots.error_rate(
        axis, points, x="p", curve=curve, rounds=ROUNDS, order=order
    )
    left_limit = min(rates) * 0.8
    right_limit = max(rates) * 1.25
    axis.set_xlim(left_limit, right_limit)
    rate_labels = [f"{rate:g}" for rate in rates]
    axis.set_xticks(rates, rate_labels, rotation=45)
    axis.set_xlabel("physical error rate")


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
