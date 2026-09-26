"""The referee wrapper's timing is the inner row's in every respect.

The wrapper re-decodes every window with Tesseract and charges nothing
for it (decsim/decoders/verify_windows.py): the inner row starts every
job, so a strong backend's measured line prices the decode whether the
referee checks it or not.
"""

import pytest

import decsim.decoders.staged_decoder as staged_decoder
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
import tests.experiments.yaml_configs as yaml_configs


def _algorithm_spans(tmp_path, check_windows_with: str) -> list:
    """(start, end) ticks of every algorithm stage of a measured_table run."""
    strong_decoder = yaml_configs.strong_unit("measured_table")
    strong_decoder["strong_decoder"]["device"] = "a100"
    observation = {"check_windows_with": check_windows_with}
    card = {"escalation": {"kind": "strong_only"}, "observation": observation}
    card.update(strong_decoder)
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=5,
        round_period_microseconds=1.0,
    )
    machine = machine_module.Machine.build(settings)
    machine.run()
    records = machine.observation.stages.records
    return [
        (record.start_ticks, record.end_ticks)
        for record in records
        if record.stage == staged_decoder.ALGORITHM_STAGE
    ]


def test_the_referee_leaves_a_measured_table_tiers_spans_unchanged(tmp_path):
    pytest.importorskip("relay_bp")
    pytest.importorskip("tesseract_decoder")
    unchecked_folder = tmp_path / "unchecked"
    checked_folder = tmp_path / "checked"
    unchecked_folder.mkdir()
    checked_folder.mkdir()
    unchecked_spans = _algorithm_spans(unchecked_folder, "none")
    checked_spans = _algorithm_spans(checked_folder, "tesseract")
    assert len(unchecked_spans) > 0
    assert checked_spans == unchecked_spans
