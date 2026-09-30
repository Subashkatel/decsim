"""The baseline's plot script against a small stats.csv of sinter rows."""

import importlib.util
import itertools
import pathlib
import shutil

import pytest
import sinter

pytest.importorskip("relay_bp")
pytest.importorskip("tesseract_decoder")

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT_FOLDER = REPOSITORY_ROOT / "experiments" / "decoder_baseline"
SAVED_RESULTS = REPOSITORY_ROOT / "results" / "2026-09-27_decoder_baseline"
EXPECTED_PLOTS = {
    "union-find.png",
    "pymatching.png",
    "xyz-relay-bp-5.png",
    "tesseract-short-beam.png",
    "decoders_x.png",
    "decoders_z.png",
}


def test_every_decoder_and_basis_gets_its_figure(tmp_path):
    plot = script_module()
    recipe_path = SCRIPT_FOLDER / "run.py"
    shutil.copy(recipe_path, tmp_path)
    write_stats(tmp_path)

    plot.main(tmp_path)

    written = {path.name for path in (tmp_path / "plots").iterdir()}
    assert written == EXPECTED_PLOTS


def test_saved_results_are_drawn_by_the_recipe_that_made_them(tmp_path):
    """The 2026-09-27 stats name relay-bp-1, which today's run.py does not.

    Their figures are the ones that commit drew, relay-bp-1's among them.
    """
    plot = script_module()
    stats_path = SAVED_RESULTS / "stats.csv"
    recipe_path = SAVED_RESULTS / "run.py"
    shutil.copy(stats_path, tmp_path)
    shutil.copy(recipe_path, tmp_path)

    plot.main(tmp_path)

    drawn = {path.name for path in (tmp_path / "plots").iterdir()}
    saved = {path.name for path in (SAVED_RESULTS / "plots").iterdir()}
    assert drawn == saved
    assert "relay-bp-1.png" in drawn


def write_stats(folder: pathlib.Path) -> None:
    """Two rates of one distance for each decoder and basis."""
    lines = [sinter.CSV_HEADER]
    decoders = ("union-find", "pymatching", "xyz-relay-bp-5")
    rates = ((0.003, 20), (0.005, 60))
    points = itertools.product(decoders, ("x", "z"), rates)
    for decoder, basis, (rate, errors) in points:
        metadata = {"basis": basis, "d": 5, "p": rate}
        point = sinter.TaskStats(
            strong_id=f"{decoder}{basis}{rate}",
            decoder=decoder,
            json_metadata=metadata,
            shots=1000,
            errors=errors,
        )
        line = point.to_csv_line()
        lines.append(line)
    lines.append("")
    text = "\n".join(lines)
    stats_path = folder / "stats.csv"
    stats_path.write_text(text)


def script_module():
    """experiments/decoder_baseline/plot.py, imported, its main not run."""
    plot_path = SCRIPT_FOLDER / "plot.py"
    return _loaded(plot_path, "decoder_baseline_plot")


def _loaded(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
