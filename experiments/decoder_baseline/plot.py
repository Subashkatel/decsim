"""The decoder baseline's figures, drawn from its stats.csv.

`python plot.py <results folder>` writes the folder's plots/: for each
decoder, the logical error rate per round against the physical error
rate, a curve per distance, a panel per basis; and for each basis, the
four decoders compared, a panel per distance. The rate per round is the
one the Tesseract paper reports (Beni et al. 2503.10988, eq. 8).
"""

import pathlib
import sys

import matplotlib.axes
import run
import sinter

import decsim.plots as plots

# every panel spans the swept rates, so a panel still filling reads on
# the same axis as a full one
LOWEST_RATE = min(run.ERROR_RATES)
HIGHEST_RATE = max(run.ERROR_RATES)
X_LIMITS = (LOWEST_RATE * 0.8, HIGHEST_RATE * 1.25)
# a label at every swept rate, since a log axis over one decade shows one
RATE_LABELS = [f"{rate:g}" for rate in run.ERROR_RATES]


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's stats.csv, into its plots/."""
    stats_path = folder / "stats.csv"
    stats = sinter.read_stats_from_csv_files(stats_path)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    for decoder in run.DECODERS:
        figure, axis_by_basis = plots.panels("basis", run.BASES)
        for basis, axis in axis_by_basis.items():
            points = plots.chosen(stats, decoder=decoder, basis=basis)
            draw(axis, points, "d", run.DISTANCES)
        figure.suptitle(decoder)
        path = plots_folder / f"{decoder}.png"
        plots.save(figure, path)
    for basis in run.BASES:
        figure, axis_by_distance = plots.panels("d", run.DISTANCES)
        for distance, axis in axis_by_distance.items():
            points = plots.chosen(stats, basis=basis, d=distance)
            draw(axis, points, "decoder", run.DECODERS)
        figure.suptitle(f"memory {basis}, decoders compared")
        path = plots_folder / f"decoders_{basis}.png"
        plots.save(figure, path)


def draw(
    axis: matplotlib.axes.Axes, points: list, curve: str, order: list
) -> None:
    """One panel: the rate per round against p, on the swept rates.

    order lists every value of curve, so a curve keeps its colour in a
    panel another has not reached yet.
    """
    plots.error_rate(
        axis, points, x="p", curve=curve, rounds=run.ROUNDS, order=order
    )
    axis.set_xlim(X_LIMITS)
    axis.set_xticks(run.ERROR_RATES, RATE_LABELS, rotation=45)
    axis.set_xlabel("physical error rate p")


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
