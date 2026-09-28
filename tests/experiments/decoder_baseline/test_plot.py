"""The baseline's plot script against a small stats.csv of sinter rows."""

import importlib.util
import itertools
import pathlib
import sys

import pytest
import sinter

pytest.importorskip("relay_bp")
pytest.importorskip("tesseract_decoder")

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT_FOLDER = REPOSITORY_ROOT / "experiments" / "decoder_baseline"
EXPECTED_PLOTS = {
    "union-find.png",
    "pymatching.png",
    "relay-bp-1.png",
    "tesseract-short-beam.png",
    "decoders_x.png",
    "decoders_z.png",
}


def test_every_decoder_and_basis_gets_its_figure(tmp_path, monkeypatch):
    plot = script_module(monkeypatch)
    write_stats(tmp_path)

    plot.main(tmp_path)

    written = {path.name for path in (tmp_path / "plots").iterdir()}
    assert written == EXPECTED_PLOTS


def write_stats(folder: pathlib.Path) -> None:
    """Two rates of one distance for each decoder and basis."""
    lines = [sinter.CSV_HEADER]
    decoders = ("union-find", "pymatching", "relay-bp-1")
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


def script_module(monkeypatch: pytest.MonkeyPatch):
    """experiments/decoder_baseline/plot.py, imported beside its own run.py.

    plot.py imports run.py by the name run, and every experiment has one,
    so this test's run is pinned in sys.modules until the test ends.
    """
    run_path = SCRIPT_FOLDER / "run.py"
    plot_path = SCRIPT_FOLDER / "plot.py"
    run = _loaded(run_path, "decoder_baseline_run")
    monkeypatch.setitem(sys.modules, "run", run)
    return _loaded(plot_path, "decoder_baseline_plot")


def _loaded(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
