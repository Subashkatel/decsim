"""The baseline's plot script against stats.csv files of sinter rows."""

import importlib.util
import itertools
import pathlib
import shutil

import sinter

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT_FOLDER = REPOSITORY_ROOT / "experiments" / "decoder_baseline"
# its run.py imports a runner since removed, so only its data can draw it
COMMITTED_FOLDER = REPOSITORY_ROOT / "results" / "2026-09-27_decoder_baseline"
DECODERS = (
    "union-find",
    "pymatching",
    "xyz-relay-bp-5",
    "tesseract-short-beam",
)
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
    write_stats(tmp_path)

    plot.main(tmp_path)

    written = {path.name for path in (tmp_path / "plots").iterdir()}
    assert written == EXPECTED_PLOTS


def test_the_committed_baseline_is_drawn_from_its_stats_alone(tmp_path):
    plot = script_module()
    committed_stats = COMMITTED_FOLDER / "stats.csv"
    shutil.copy(committed_stats, tmp_path)

    plot.main(tmp_path)

    written = {path.name for path in (tmp_path / "plots").iterdir()}
    assert written == EXPECTED_PLOTS


def test_the_committed_baselines_grid_is_read_back_sorted():
    """stats.csv lists the distances out of order; the figures do not."""
    plot = script_module()
    committed_stats = COMMITTED_FOLDER / "stats.csv"
    stats = sinter.read_stats_from_csv_files(committed_stats)

    distances = plot.labels(stats, "d")
    rates = plot.labels(stats, "p")

    assert distances == [5, 7, 9, 11, 13, 15]
    assert rates == [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]


def test_the_decoders_keep_run_pys_order():
    """stats.csv lists Relay-BP last, as it was rerun after the others."""
    plot = script_module()
    committed_stats = COMMITTED_FOLDER / "stats.csv"
    stats = sinter.read_stats_from_csv_files(committed_stats)

    decoders = plot.decoders_in_order(stats)

    assert decoders == list(DECODERS)


def write_stats(folder: pathlib.Path) -> None:
    """Two rates of one distance for each decoder and basis."""
    lines = [sinter.CSV_HEADER]
    rates = ((0.003, 20), (0.005, 60))
    points = itertools.product(DECODERS, ("x", "z"), rates)
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
