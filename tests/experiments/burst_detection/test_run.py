"""The burst detection script against decsim's workload and Stim's sampler.

The circuit's referent is decsim.frontends.settings.memory_circuit, the
maker the decoder baseline samples. A strength's firing is read off the
detector error model, so Stim's own shots of the same burst are its
referent.
"""

import importlib.util
import pathlib

import numpy
import pytest

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.frontends.settings as workload_settings
import decsim.qpu.stim_device as stim_device

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT = REPOSITORY_ROOT / "experiments" / "burst_detection" / "run.py"


def test_the_circuit_is_decsims_z_memory():
    run = script_module()

    circuit = run.circuit(5, 0.003, 2000)

    reference = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 2000, 5, 0.003
    )
    assert circuit == reference


def test_the_calibrations_come_first_then_each_rates_eighteen_points():
    """36 calibrations; then 12 bursts, 4 quiet parts, levels, traces."""
    run = script_module()

    experiment = run.burst_detection()

    files = [point.results_file for point in experiment.function_points]
    calibration_labels = experiment.function_points[3].labels
    burst_labels = experiment.function_points[94].labels
    rate_files = ["trials.csv"] * 12 + ["quiet.csv"] * 4
    rate_files += ["levels.csv", "traces.csv"]
    assert files == ["alarm_levels.csv"] * 36 + rate_files * 36
    assert calibration_labels == {"d": 5, "p": 0.003}
    assert burst_labels == {
        "d": 5,
        "p": 0.003,
        "size": "medium",
        "strength": "strong",
    }


def test_a_point_whose_calibration_is_not_saved_is_refused(tmp_path):
    run = script_module()
    labels = {"d": 5, "p": 0.003, "part": 0}

    with pytest.raises(ValueError) as refused:
        run.quiet_point(labels, 1, tmp_path)

    path = tmp_path / "points" / "3.csv"
    assert str(refused.value) == (
        f"no alarm levels for d = 5, p = 0.003 at {path}; run calibration "
        "point 3 first (on Slurm, submit this point with "
        "--dependency=afterok on its job)"
    )


def test_quiet_time_is_whole_shots_of_2000_rounds_of_1_us():
    """The calibration's block, not the 1,999 rounds a shot scores."""
    run = script_module()

    seconds = run.quiet_seconds(1000)

    assert seconds == 2.0


def test_a_strength_reaches_its_multiple_of_the_quiet_firing():
    run = script_module()

    weak = run.burst_level(5, 0.003, "weak")
    strong = run.burst_level(5, 0.003, "strong")
    saturating = run.burst_level(5, 0.003, "saturating")

    quiet_firing = weak["quiet_firing"]
    weak_target = 2.5 * quiet_firing
    strong_target = 6 * quiet_firing
    assert weak["firing"] == pytest.approx(weak_target, rel=1e-6)
    assert strong["firing"] == pytest.approx(strong_target, rel=1e-6)
    assert saturating["level"] == 0.75
    assert saturating["firing"] == pytest.approx(0.5, abs=1e-3)


def test_the_firing_read_off_the_model_is_stims_sampled_firing():
    """20,000 shots of the strong whole-patch burst, its peak round."""
    run = script_module()
    strong = run.burst_level(5, 0.003, "strong")
    circuit = run.circuit(5, 0.003, run.SEARCH_ROUNDS)
    table = detector_formation.build_formation_table(circuit, run.SEARCH_ROUNDS)
    settings = stim_device.BurstStimDevice.Settings(
        burst_onset_round=run.SEARCH_ONSET_ROUND,
        burst_rise_rounds=run.RISE_ROUNDS,
        burst_decay_rounds=run.DECAY_ROUNDS,
        burst_error_probability=strong["level"],
    )
    burst = stim_device.burst_circuit(circuit, table, settings)
    sampler = burst.compile_detector_sampler(seed=3)

    events = sampler.sample(20_000)

    coordinates = burst.get_detector_coordinates()
    time_list = [coordinates[index][2] for index in coordinates]
    times = numpy.array(time_list, dtype=int)
    firing = events.mean(axis=0)
    firing_sums = numpy.bincount(times, weights=firing)
    check_counts = numpy.bincount(times)
    round_firing = firing_sums / check_counts
    bulk_firing = round_firing[1:-1]
    sampled_peak = bulk_firing.max()
    # 24 checks of 20,000 shots a round: a standard error near 0.0007
    assert sampled_peak == pytest.approx(strong["firing"], abs=0.003)


def script_module():
    """experiments/burst_detection/run.py, imported, its main not run."""
    spec = importlib.util.spec_from_file_location("burst_detection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
