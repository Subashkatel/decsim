"""`decsim stream` (experiments/stream_command.py).

A memory task runs as a stream and writes one record; each shape a
stream does not build yet is refused by name, "not implemented yet".
"""

import dataclasses
import json

import pytest

import decsim
import decsim.confidence.complementary as complementary
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.command as command
import decsim.experiments.refusal as refusal
import decsim.experiments.stream_command as stream_command
import decsim.parameters.assignments as assignments
import decsim.presets.machines as machines
import decsim.qpu.syndrome_devices as syndrome_devices

RUN_FILE = """
import decsim
import decsim.presets.machines as machines

memory = decsim.Task("memory", machines.weak_baseline(3, 0.002, 1.0))
stop = decsim.CollectionSettings(max_shots=1)
experiment = decsim.Experiment("stream", [memory], stop)
"""


@pytest.fixture(autouse=True)
def _no_assignments_left_behind(monkeypatch):
    """Each test starts with DECSIM_SET empty."""
    monkeypatch.setenv(assignments.ENVIRONMENT_VARIABLE, "")


def _task() -> decsim.Task:
    machine = machines.weak_baseline(3, 0.002, 1.0)
    return decsim.Task("memory", machine)


def _with(task: decsim.Task, **changes) -> decsim.Task:
    """The task with its machine's parts replaced, each a whole record."""
    machine = dataclasses.replace(task.machine, **changes)
    return dataclasses.replace(task, machine=machine, graph=None)


def test_a_memory_task_streams_and_writes_one_record(tmp_path):
    run_file = tmp_path / "stream.py"
    run_file.write_text(RUN_FILE)
    out = tmp_path / "stream.json"

    arguments = ["stream", str(run_file), "--rounds", "150"]
    arguments += ["--warmup", "30", "--seed", "5", "--out", str(out)]
    command.main(arguments)

    text = out.read_text()
    record = json.loads(text)
    measurement = record["measurement"]
    assert (record["task"], record["rounds"], record["seed"]) == (
        "memory",
        150,
        5,
    )
    assert record["warmup_rounds"] == 30
    assert measurement["decoded_windows"] > 40
    assert measurement["logical_failure"] in (True, False)


def test_the_stream_runs_the_tasks_machine_folding_its_windows():
    task = _task()

    settings = stream_command.stream_settings(task, 200, 20)

    assert settings.observation.folds_windows
    assert settings.observation.warmup_rounds == 20
    assert settings.observation.first_recorded_round == 21
    assert settings.qpu.source.name == "flip_stream"
    assert settings.weak_decoder == task.machine.weak_decoder


def _observation(task, **changes):
    observation = dataclasses.replace(task.machine.observation, **changes)
    return {"observation": observation}


def _source(task):
    source = syndrome_devices.TimingOnlyDevice.Settings()
    qpu = dataclasses.replace(task.machine.qpu, source=source)
    return {"qpu": qpu}


def _double_window(task):
    del task
    strong_window = strong_window_shapes.DoubleWindow.Settings()
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=12.0
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    return {"switching": switching}


def _bulk_strong(task):
    manager = dataclasses.replace(
        task.machine.decoder_manager, bulk_strong=True
    )
    return {"decoder_manager": manager}


# each change a stream refuses, and the words its refusal says
REFUSED = [
    (lambda task: _observation(task, trace="chrome"), "keeps no trace"),
    (lambda task: _observation(task, log="file"), "keeps no log"),
    (
        lambda task: _observation(task, data_movement=True),
        "no data movement ledger",
    ),
    (_source, "this task's source is timing_only"),
    (_double_window, "the double_window absorbs windows"),
    (_bulk_strong, "(bulk_strong)"),
]


@pytest.mark.parametrize("change, words", REFUSED)
def test_a_shape_a_stream_does_not_build_is_refused_by_name(change, words):
    task = _task()
    changes = change(task)
    changed = _with(task, **changes)

    with pytest.raises(refusal.RefusalError) as refused:
        stream_command.stream_settings(changed, 100, 0)

    sentence = str(refused.value)
    assert sentence.startswith("not implemented yet")
    assert words in sentence


def test_a_warmup_past_the_rounds_is_refused(tmp_path):
    run_file = tmp_path / "stream.py"
    run_file.write_text(RUN_FILE)

    with pytest.raises(SystemExit):
        command.main(
            ["stream", str(run_file), "--rounds", "50", "--warmup", "50"]
        )
