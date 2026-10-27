"""`decsim stream` (experiments/stream_command.py).

A memory task runs as a stream and writes one record, with its moving
threshold's when it has one; each shape a stream does not build yet is
refused by name, "not implemented yet". Patches streamed on a pool each,
or sharing a strong pool, answer every window as one pool does
(tests/patch_pools.py: a router decides where a window decodes, never
what it reads).
"""

import dataclasses
import json

import pytest

import decsim
import decsim.confidence.complementary as complementary
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.target_steps as target_steps
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.command as command
import decsim.experiments.refusal as refusal
import decsim.experiments.stream_command as stream_command
import decsim.parameters.assignments as assignments
import decsim.presets.machines as machines
import decsim.qpu.syndrome_devices as syndrome_devices
import tests.experiments.run_files as run_files
import tests.patch_pools as patch_pools

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


def _with(task: decsim.Task, change) -> decsim.Task:
    """The task with its machine changed by change."""
    machine = change(task.machine)
    return dataclasses.replace(task, machine=machine)


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

    machine = stream_command.stream_machine(task, 200, 20)

    qpu = machine.held("qpu")
    weak = machine.part(machines.WEAK_POOL)
    task_weak = task.machine.part(machines.WEAK_POOL)
    assert machine.observation.folds_windows
    assert machine.observation.warmup_rounds == 20
    assert machine.observation.first_recorded_round == 21
    assert qpu.source.name == "flip_stream"
    assert weak.settings == task_weak.settings


def test_a_stream_whose_threshold_moves_records_the_source_beside_it():
    machine = run_files.switching_machine(rate={"target_escalation_rate": 0.2})
    task = decsim.Task("rate", machine)

    record = stream_command.stream_record(task, 300, 0, 5)

    online = record["online_threshold"]
    measurement = record["measurement"]
    summary = online["summary"]
    assert summary["windows"] == measurement["decoded_windows"]
    assert summary["escalated"] == measurement["escalated_windows"]
    assert online["trajectory"][0][2] == "start"


def test_a_stream_whose_target_steps_by_the_backlog_binds_its_queue():
    steps = target_steps.TargetSteps.Settings(
        start=0.2, levels=(0.1, 0.2), check_every_windows=10
    )
    rate = {"target_escalation_rate": 0.2, "steps": steps}
    machine = run_files.switching_machine(rate=rate)
    task = decsim.Task("loop", machine)

    record = stream_command.stream_record(task, 300, 0, 5)

    summary = record["online_threshold"]["summary"]
    checks = summary["target_steps"]
    assert len(checks) == summary["windows"] // 10
    assert {share for _windows, share, _mean in checks} <= {0.1, 0.2}


# the rounds the patch streams run: every weak pool escalates in them
PATCH_ROUNDS = 30


def test_patches_streamed_on_a_pool_each_answer_every_window_as_one_pool():
    one_pool = patch_pools.one_pool()
    per_patch = patch_pools.pool_per_patch(one_pool)

    one_answers, _one_pools = patch_pools.streamed(one_pool, PATCH_ROUNDS)
    routed_answers, routed_pools = patch_pools.streamed(per_patch, PATCH_ROUNDS)

    assert routed_answers == one_answers
    assert routed_pools == {"weak", "w1", "w2"}


def test_patches_streamed_beside_a_shared_strong_pool_answer_as_one_pair():
    pair = patch_pools.one_pair()
    shared = patch_pools.shared_strong_per_patch()
    shared_stores = patch_pools.shared_strong_per_patch(has_store_per_pool=True)

    pair_answers, _pair_pools = patch_pools.streamed(pair, PATCH_ROUNDS)
    shared_answers, shared_pools = patch_pools.streamed(shared, PATCH_ROUNDS)
    stores_answers, _stores_pools = patch_pools.streamed(
        shared_stores, PATCH_ROUNDS
    )

    assert shared_answers == pair_answers
    assert stores_answers == pair_answers
    assert shared_pools == {"weak", "w1", "w2", "strong"}


def _observation(machine, **changes):
    observation = dataclasses.replace(machine.observation, **changes)
    return dataclasses.replace(machine, observation=observation)


def _source(machine):
    source = syndrome_devices.TimingOnlyDevice.Settings()
    qpu = machine.held("qpu")
    timing_qpu = dataclasses.replace(qpu, source=source)
    return machine.with_held("qpu", timing_qpu)


def _double_window(machine):
    strong_window = strong_window_shapes.DoubleWindow.Settings()
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=12.0
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    strong = machines.STRONG_BASELINE_POOL
    return machines.with_strong_tier(machine, strong, switching)


def _bulk_strong(machine):
    manager = machine.held("decoder_manager")
    bulk_manager = dataclasses.replace(manager, bulk_strong=True)
    return machine.with_held("decoder_manager", bulk_manager)


# each change a stream refuses, and the words its refusal says
REFUSED = [
    (lambda machine: _observation(machine, trace="chrome"), "keeps no trace"),
    (lambda machine: _observation(machine, log="file"), "keeps no log"),
    (
        lambda machine: _observation(machine, data_movement=True),
        "no data movement ledger",
    ),
    (_source, "this task's source is timing_only"),
    (_double_window, "the double_window absorbs windows"),
    (_bulk_strong, "(bulk_strong)"),
]


@pytest.mark.parametrize("change, words", REFUSED)
def test_a_shape_a_stream_does_not_build_is_refused_by_name(change, words):
    task = _task()
    changed = _with(task, change)

    with pytest.raises(refusal.RefusalError) as refused:
        stream_command.stream_machine(changed, 100, 0)

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
