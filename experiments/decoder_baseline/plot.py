"""The decoder baseline's figures, drawn from its stats.csv.

`python plot.py <results folder>` writes the folder's plots/: for each
decoder, the logical error rate per round against the physical error
rate, a curve per distance, a panel per basis; and for each basis, the
decoders compared, a panel per distance. The rate per round is the one
the Tesseract paper reports (Beni et al. 2503.10988, eq. 8).

The decoders, bases, distances, rates and rounds are the ones in the
folder's own run.py, the script that made its stats: the run copies it
there (decsim/experiments/run_folder.py, start_run), and this
directory's run.py is today's recipe, which may name other decoders.
"""

import ast
import pathlib
import sys
import types

import matplotlib.axes
import sinter

import decsim.plots as plots


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's stats.csv, into its plots/."""
    recipe = recipe_of(folder)
    stats_path = folder / "stats.csv"
    stats = sinter.read_stats_from_csv_files(stats_path)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    for decoder in recipe.DECODERS:
        figure, axis_by_basis = plots.panels("basis", recipe.BASES)
        for basis, axis in axis_by_basis.items():
            points = plots.chosen(stats, decoder=decoder, basis=basis)
            draw(axis, points, "d", recipe.DISTANCES, recipe)
        figure.suptitle(decoder)
        path = plots_folder / f"{decoder}.png"
        plots.save(figure, path)
    for basis in recipe.BASES:
        figure, axis_by_distance = plots.panels("d", recipe.DISTANCES)
        for distance, axis in axis_by_distance.items():
            points = plots.chosen(stats, basis=basis, d=distance)
            draw(axis, points, "decoder", recipe.DECODERS, recipe)
        figure.suptitle(f"memory {basis}, decoders compared")
        path = plots_folder / f"decoders_{basis}.png"
        plots.save(figure, path)


def recipe_of(folder: pathlib.Path) -> types.SimpleNamespace:
    """The constants of the run.py the folder's stats were made by.

    The script is read, not run: each module-level assignment is
    evaluated in order from the names before it, and one that needs an
    import (a decoder object) is left out. A recipe whose imports have
    since moved on still names its grid, and nothing runs.
    """
    path = folder / "run.py"
    source = path.read_text()
    module = ast.parse(source)
    constants = {}
    for statement in module.body:
        _add_the_constant(statement, constants)
    return types.SimpleNamespace(**constants)


def _add_the_constant(statement: ast.stmt, constants: dict) -> None:
    """One `NAME = value` the names before it can evaluate, into constants."""
    if not isinstance(statement, ast.Assign):
        return
    (target,) = statement.targets
    if not isinstance(target, ast.Name):
        return
    expression = ast.Expression(statement.value)
    code = compile(expression, "<recipe>", "eval")
    # No builtins: a constant is a literal or is built from the names
    # before it, and anything else needs the script's imports.
    namespace = {"__builtins__": {}, **constants}
    try:
        value = eval(code, namespace)
    except (NameError, AttributeError, TypeError):
        return
    constants[target.id] = value


def draw(
    axis: matplotlib.axes.Axes,
    points: list,
    curve: str,
    order: list,
    recipe: types.SimpleNamespace,
) -> None:
    """One panel: the rate per round against p, on the swept rates.

    order lists every value of curve, so a curve keeps its colour in a
    panel another has not reached yet. Every panel spans the swept
    rates, so a panel still filling reads on the same axis as a full
    one, and carries a label at every swept rate, since a log axis over
    one decade shows one.
    """
    plots.error_rate(
        axis, points, x="p", curve=curve, rounds=recipe.ROUNDS, order=order
    )
    rates = recipe.ERROR_RATES
    left_limit = min(rates) * 0.8
    right_limit = max(rates) * 1.25
    axis.set_xlim(left_limit, right_limit)
    rate_labels = [f"{rate:g}" for rate in rates]
    axis.set_xticks(rates, rate_labels, rotation=45)
    axis.set_xlabel("physical error rate")


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
