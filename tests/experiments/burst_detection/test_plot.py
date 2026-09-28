"""The burst detection plot's one number, worked by hand."""

import importlib.util
import math
import pathlib
import sys

import pytest

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
SCRIPT_FOLDER = _TEST_FILE.parents[3] / "experiments" / "burst_detection"


def test_rounds_to_90_percent_is_the_ninth_of_ten_sorted_delays(
    monkeypatch,
):
    """One burst never caught still leaves 9 of 10 caught by round 9."""
    plot = plot_module(monkeypatch)
    delays = [5, 1, 3, math.inf, 2, 4, 6, 7, 8, 9]
    twice_missed = [1, 2, 3, 4, 5, 6, 7, 8, math.inf, math.inf]

    assert plot.rounds_to_share(delays) == 9
    assert plot.rounds_to_share(twice_missed) == math.inf


def plot_module(monkeypatch: pytest.MonkeyPatch):
    """experiments/burst_detection/plot.py, beside its own run.py."""
    monkeypatch.syspath_prepend(str(SCRIPT_FOLDER))
    monkeypatch.delitem(sys.modules, "run", raising=False)
    path = SCRIPT_FOLDER / "plot.py"
    spec = importlib.util.spec_from_file_location("burst_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
