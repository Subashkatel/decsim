"""The burst detector comparison folder: one base, one change per file.

configs/burst_detectors_compared/ compares four detectors on one burst
and on quiet shots, so a difference between two files' rows is the
detector's or the burst's only when each file changes nothing else.
"""

import pytest
import yaml

import decsim.experiments.experiment as experiment
import tests.experiments.yaml_configs as yaml_configs

FOLDER = yaml_configs.CONFIGS_DIR / "burst_detectors_compared"
BASE = "../common/burst_detectors_compared_base.yaml"
# each detector's burst_detector section, as its burst file writes it
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


def test_the_folder_holds_a_burst_and_a_quiet_file_per_detector():
    paths = FOLDER.glob("*.yaml")
    names = sorted(path.stem for path in paths)

    assert names == [
        "event_count_burst",
        "event_count_quiet",
        "masked_regional_cusum_burst",
        "masked_regional_cusum_quiet",
        "regional_cusum_burst",
        "regional_cusum_quiet",
        "whole_patch_cusum_burst",
        "whole_patch_cusum_quiet",
    ]


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_burst_file_changes_only_its_detector(detector):
    burst_path = FOLDER / f"{detector}_burst.yaml"
    written = _written(burst_path)

    assert written == {"extends": BASE, "burst_detector": DETECTORS[detector]}


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_quiet_file_changes_only_its_burst(detector):
    quiet_path = FOLDER / f"{detector}_quiet.yaml"
    written = _written(quiet_path)
    burst_name = f"{detector}_burst.yaml"

    assert written == {"extends": burst_name, "qpu": {"kind": "stim_device"}}


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_file_loads_with_its_detector_and_the_catch_deadline(detector):
    burst_path = FOLDER / f"{detector}_burst.yaml"
    quiet_path = FOLDER / f"{detector}_quiet.yaml"

    burst_config = experiment.load_experiment(burst_path)
    quiet_config = experiment.load_experiment(quiet_path)
    burst_point = burst_config.first_point_task()
    quiet_point = quiet_config.first_point_task()

    burst = burst_point.settings
    quiet = quiet_point.settings
    kind = DETECTORS[detector]["kind"]
    assert burst.burst_detector.kind == kind
    assert quiet.burst_detector == burst.burst_detector
    assert burst.burst_detector.catch_deadline_rounds == 300
    assert burst.qpu.kind == "burst_stim"
    assert quiet.qpu.kind == "stim_device"


def _written(path) -> dict:
    text = path.read_text()
    return yaml.safe_load(text)
