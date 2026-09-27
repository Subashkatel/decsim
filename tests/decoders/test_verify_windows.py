"""The referee wrapper's timing is the inner row's in every respect.

The wrapper re-decodes every window with Tesseract and charges nothing
for it (decsim/decoders/verify_windows.py): the inner row starts every
job, so a strong backend's measured line prices the decode whether the
referee checks it or not.
"""

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.verify_windows as verify_windows
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
import tests.decoders.windows as windows
import tests.experiments.yaml_configs as yaml_configs


def _algorithm_spans(folder, card: dict, check_windows_with: str) -> list:
    """(start, end) ticks of every algorithm stage of one card's run."""
    folder.mkdir()
    observation = {"check_windows_with": check_windows_with}
    checked_card = {**card, "observation": observation}
    config_path = yaml_configs.write_config(folder, checked_card)
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 5,
            "qpu.round_period_microseconds": 1.0,
        },
        1,
    )
    machine = machine_module.Machine.build(point.settings)
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
    strong_decoder = yaml_configs.strong_unit("measured_table")
    strong_decoder["strong_decoder"]["device"] = "a100"
    card = {"escalation": {"kind": "strong_only"}, **strong_decoder}

    unchecked_folder = tmp_path / "unchecked"
    checked_folder = tmp_path / "checked"

    unchecked_spans = _algorithm_spans(unchecked_folder, card, "none")
    checked_spans = _algorithm_spans(checked_folder, card, "tesseract")

    assert len(unchecked_spans) > 0
    assert checked_spans == unchecked_spans


def test_a_cancelled_job_delivers_nothing_and_is_not_checked():
    """A cancelled job's decode delivers None (DecoderBase.start).

    The referee passes that None on and reaches no verdict, although the
    job carries a window model it could have checked.
    """
    pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, 3, 0.001)
    requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
    model = windows.whole_circuit_window(circuit, 3, requirement)
    events, _observables = windows.sampled_shots(circuit, 1, 3)
    job = windows.job_for(model, events[0])
    job.cancelled = True
    inner = decoders.PresetLatencyDecoder(2.0)
    referee = verify_windows.TesseractCheckedDecoder(inner)
    verdicts = []
    delivered = []

    def note_verdict(window_key, is_agreement) -> None:
        verdicts.append((window_key, is_agreement))

    referee.window_checked.connect(note_verdict)
    engine = engine_module.Engine()

    referee.start(job, engine, delivered.append)
    engine.run()

    assert delivered == [None]
    assert verdicts == []
