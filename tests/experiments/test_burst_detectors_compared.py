"""The burst detector comparison: four detectors, each burst and quiet.

configs/experiments/burst_detection/burst_detection.yaml compares four
detectors on one burst and on quiet shots, so a difference between two
of its rows is the detector's or the burst's only when its points
differ in nothing else.
"""

import pytest

import decsim.collect as collect
import decsim.experiments.experiment as experiment
import tests.experiments.yaml_configs as yaml_configs

GRID = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "burst_detection"
    / "burst_detection.yaml"
)
# each detector's burst_detector mapping, as the grid writes it
DETECTORS = {
    "event_count": {"kind": "event_count"},
    "whole_patch_cusum": {
        "kind": "masked_regional_cusum",
        "mask_count": None,
        "region_radii": [],
    },
    "regional_cusum": {"kind": "masked_regional_cusum", "mask_count": None},
    "masked_regional_cusum": {"kind": "masked_regional_cusum"},
}
QUIET = {"kind": "stim_device"}
BURST = {
    "kind": "burst_stim",
    "burst_onset_round": 50,
    "burst_decay_rounds": 600.0,
    "burst_radius": 3.1,
    "burst_error_probability": 0.01,
}
# the settings a detector and a burst own, the two the grid sweeps
SWEPT_SETTINGS = ("burst_detector", "qpu")


def test_the_grid_holds_a_quiet_and_a_burst_point_per_detector():
    config = experiment.load_experiment(GRID)
    (block,) = config.sweep
    points = block.points()

    conditions = _conditions(points)

    assert conditions == [
        (DETECTORS["event_count"], QUIET),
        (DETECTORS["event_count"], BURST),
        (DETECTORS["whole_patch_cusum"], QUIET),
        (DETECTORS["whole_patch_cusum"], BURST),
        (DETECTORS["regional_cusum"], QUIET),
        (DETECTORS["regional_cusum"], BURST),
        (DETECTORS["masked_regional_cusum"], QUIET),
        (DETECTORS["masked_regional_cusum"], BURST),
    ]


def test_every_point_differs_only_in_its_detector_and_its_burst():
    config = experiment.load_experiment(GRID)
    (block,) = config.sweep
    points = block.points()

    unswept = _unswept_settings(config, points)

    first = unswept[0]
    assert unswept == [first] * len(DETECTORS) * 2


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_point_loads_with_its_detector_and_the_catch_deadline(detector):
    config = experiment.load_experiment(GRID)
    burst_values = _point_values(detector, BURST)
    quiet_values = _point_values(detector, QUIET)

    burst_point = config.point_task(burst_values)
    quiet_point = config.point_task(quiet_values)

    burst = burst_point.settings
    quiet = quiet_point.settings
    kind = DETECTORS[detector]["kind"]
    assert burst.burst_detector.kind == kind
    assert quiet.burst_detector == burst.burst_detector
    assert burst.burst_detector.catch_deadline_rounds == 300
    assert burst.qpu.kind == "burst_stim"
    assert quiet.qpu.kind == "stim_device"


def _conditions(points: list) -> list:
    """Each point's detector and QPU mappings, in the grid's order."""
    conditions = []
    for point in points:
        condition = (point["burst_detector"], point["qpu"])
        conditions.append(condition)
    return conditions


def _unswept_settings(config, points: list) -> list:
    """Each point's settings as json, the detector and the QPU left out."""
    unswept = []
    for point in points:
        task = config.point_task(point)
        settings = collect.json_value(task.settings)
        for name in SWEPT_SETTINGS:
            del settings[name]
        unswept.append(settings)
    return unswept


def _point_values(detector: str, qpu: dict) -> dict:
    """One point of the grid, its axes in the order the grid writes them."""
    return {
        "burst_detector": DETECTORS[detector],
        "qpu": qpu,
        yaml_configs.ERROR_RATE_PATH: 0.003,
        "qpu.distance": 5,
        "qpu.round_period_microseconds": 1.0,
    }
