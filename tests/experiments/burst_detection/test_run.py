"""The burst detection script, run whole on a tiny grid."""

import dataclasses
import importlib.util
import pathlib

import pytest

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT_FOLDER = REPOSITORY_ROOT / "experiments" / "burst_detection"
SCRIPT = SCRIPT_FOLDER / "run.py"


def test_a_hit_qubit_runs_at_2_5_p_6_p_or_saturated():
    run = script_module("run")

    weak = run.added_probability("weak", 0.002)
    strong = run.added_probability("strong", 0.002)
    saturating = run.added_probability("saturating", 0.002)

    assert weak == pytest.approx(0.003)
    assert strong == pytest.approx(0.01)
    assert saturating == 0.75


def test_a_point_whose_calibration_is_not_saved_is_refused(tmp_path):
    run = script_module("run")
    labels = {"d": 5, "p": 0.003, "part": 0}

    with pytest.raises(ValueError) as refused:
        run.quiet_point(labels, 1, tmp_path)

    path = tmp_path / "points" / "3.csv"
    assert str(refused.value) == (
        f"no alarm levels at {path}: run calibration point 3 first (on "
        "Slurm, --dependency=afterok)"
    )


def test_a_tiny_grid_runs_and_combines(tmp_path, monkeypatch):
    """Only d = 5 and p = 0.003: 40 rounds, 2 trials a class, 4 streams."""
    run = script_module("run")
    one_shot_calibration = dataclasses.replace(
        run.DETECTOR_SETTINGS, calibration_shot_count=1000
    )
    monkeypatch.setattr(run, "DETECTOR_SETTINGS", one_shot_calibration)
    monkeypatch.setattr(run, "RATE_POINTS", [(5, 0.003)])
    monkeypatch.setattr(run, "ROUNDS", 40)
    monkeypatch.setattr(run, "ONSET_ROUND", 20)
    monkeypatch.setattr(run, "TRIALS", 2)
    monkeypatch.setattr(run, "QUIET_PARTS", 1)
    monkeypatch.setattr(run, "QUIET_BATCHES", 1)
    monkeypatch.setattr(run, "QUIET_STREAMS", 4)
    folder = tmp_path / "out"

    run.main(["--out", str(folder)])

    trials_text = (folder / "trials.csv").read_text()
    quiet_text = (folder / "quiet.csv").read_text()
    assert trials_text.count("\n") == 1 + 12 * 2 * 3
    quiet_lines = quiet_text.splitlines()
    assert len(quiet_lines) == 1 + 3
    assert quiet_lines[1].startswith("5,0.003,0,1.0,0.00016,")


def script_module(name: str):
    """experiments/burst_detection/<name>.py, imported, its main not run."""
    path = SCRIPT_FOLDER / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"burst_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
