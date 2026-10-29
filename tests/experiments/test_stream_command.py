"""`decsim stream` (experiments/stream_command.py).

A memory task runs as a stream and writes one record, with its moving
threshold's when it has one; each shape a stream does not build yet is
refused by name, "not implemented yet". Patches streamed on a pool each,
or sharing a strong pool, answer every window as one pool does
(tests/patch_pools.py: a router decides where a window decodes, never
what it reads). The double window streams: with nothing escalated it
answers as union-find alone does, each window it escalates absorbs the
windows its region covers, and checkpoints beside it are refused.
"""

import dataclasses
import json

import pytest

import decsim
import decsim.confidence.cluster as cluster
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.target_steps as target_steps
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.command as command
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.experiments.stream_command as stream_command
import decsim.machine as machine_module
import decsim.parameters.assignments as assignments
import decsim.presets.experiment_one as experiment_one
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
# the same memory, its source bursting on 8 of the patch's 17 qubits
BURST_RUN_FILE = """
import dataclasses

import decsim
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device

machine = machines.weak_baseline(3, 0.002, 1.0)
burst = flip_stream_device.BurstSettings(
    onset_round=50, peak_factor=10.0, decay_rounds=30.0, qubit_count=8
)
source = flip_stream_device.FlipStreamDevice.Settings(burst=burst)
qpu = dataclasses.replace(machine.held("qpu"), source=source)
memory = decsim.Task("memory", machine.with_held("qpu", qpu))
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


def test_a_memory_task_whose_source_bursts_streams_to_its_readout(tmp_path):
    run_file = tmp_path / "burst.py"
    run_file.write_text(BURST_RUN_FILE)
    out = tmp_path / "burst.json"

    arguments = ["stream", str(run_file), "--rounds", "150"]
    arguments += ["--warmup", "30", "--seed", "5", "--out", str(out)]
    command.main(arguments)

    text = out.read_text()
    record = json.loads(text)
    measurement = record["measurement"]
    study = experiment.load_one_task(str(run_file), None)
    machine = stream_command.stream_machine(study.tasks[0], 150, 30)
    qpu = machine.held("qpu")
    assert qpu.source.burst.onset_round == 50
    assert record["rounds"] == 150
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


# Experiment 1's union-find memory at d = 3 and p = 0.01, where a cluster
# gap under 20 dB escalates many windows
UNION_FIND_ERROR_RATE = 0.01
# its stream's rounds: fifty windows of three
UNION_FIND_ROUNDS = 150
_DOUBLE_WINDOW = strong_window_shapes.DoubleWindow.Settings()
_REDO_WINDOW = strong_window_shapes.RedoWindow.Settings()
_STATED_TIMES = belief_matching.IterationTime(
    iteration_microseconds=0.5, matching_microseconds=10.0
)
_STATED_BELIEF_MATCHING = belief_matching.BeliefMatchingDecoder.Settings(
    max_iterations=30, timing=_STATED_TIMES
)


def _switching(machine, strong_window, decibels: float):
    """The machine escalating a window whose cluster gap is under decibels.

    The strong pool is belief matching on stated times, which draws
    nothing, so a run answers alike each time (tests/patch_pools.py).
    """
    confidence = cluster.ClusterGap.Settings(walk_microseconds=0.2)
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=decibels
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    strong = dataclasses.replace(
        machines.STRONG_BASELINE_POOL, algorithm=_STATED_BELIEF_MATCHING
    )
    return machines.with_strong_tier(machine, strong, switching)


def test_a_double_window_stream_that_never_escalates_answers_as_union_find():
    """At 0 dB no window escalates, so each keeps union-find's answer.

    The double window's stream answers every window and the outcome as
    union-find's alone does, and as the redo window's stream does, at
    the same seed and rounds.
    """
    union_find = experiment_one.weak_alone(3, UNION_FIND_ERROR_RATE)
    double = _switching(union_find, _DOUBLE_WINDOW, 0.0)
    redo = _switching(union_find, _REDO_WINDOW, 0.0)

    alone, alone_pools = patch_pools.streamed(union_find, UNION_FIND_ROUNDS)
    doubled, double_pools = patch_pools.streamed(double, UNION_FIND_ROUNDS)
    redone, redo_pools = patch_pools.streamed(redo, UNION_FIND_ROUNDS)

    assert doubled == alone
    assert redone == alone
    assert alone_pools == double_pools == redo_pools == {"weak"}


def test_a_double_window_stream_commits_or_absorbs_each_window_once():
    """Each window commits once, or a strong region absorbs it undecoded.

    The stream lays its windows as its rounds arrive; a region is laid
    out once the stream has the windows it covers, and absorbs them
    (escalation/strong_window_shapes.py DoubleWindow).
    """
    union_find = experiment_one.weak_alone(3, UNION_FIND_ERROR_RATE)
    double = _switching(union_find, _DOUBLE_WINDOW, 20.0)
    task = decsim.Task("double", double)
    graph = stream_command.stream_machine(task, UNION_FIND_ROUNDS, 0)
    machine = machine_module.Machine.build(graph, patch_pools.SEED)
    windows = _WindowFates()
    windows.attach(machine)

    machine.run()

    assert windows.absorbed
    assert "strong" in windows.committed.values()
    ended = sorted([*windows.committed, *windows.absorbed])
    assert ended == sorted(windows.planned)


def test_checkpoints_beside_a_double_window_are_refused_with_the_reason():
    union_find = experiment_one.weak_alone(3, UNION_FIND_ERROR_RATE)
    double = _switching(union_find, _DOUBLE_WINDOW, 20.0)
    checkpointed = _observation(double, checkpoint_every_rounds=30)
    task = decsim.Task("double", checkpointed)
    graph = stream_command.stream_machine(task, UNION_FIND_ROUNDS, 0)

    with pytest.raises(NotImplementedError) as refused:
        machine_module.Machine.build(graph, patch_pools.SEED)

    sentence = str(refused.value)
    assert "absorbs weak ones (double_window)" in sentence
    assert "is not implemented yet" in sentence


class _WindowFates:
    """Each laid window's key, and how it ended: committed or absorbed.

    committed maps a key to the tier that committed it.
    """

    def __init__(self) -> None:
        self.planned: list = []
        self.committed: dict = {}
        self.absorbed: list = []

    def attach(self, machine) -> None:
        sources = machine.windows.window_manager.window_sources()
        sources.window_planned.connect(self.window_planned)
        sources.window_absorbed.connect(self.window_absorbed)
        frame = machine.control.pauli_frame
        frame.trace.correction_committed.connect(self.correction_committed)

    def window_planned(self, window) -> None:
        self.planned.append(window.key)

    def window_absorbed(self, key: tuple, owner_key: tuple) -> None:
        del owner_key
        self.absorbed.append(key)

    def correction_committed(self, record) -> None:
        assert record.window_key not in self.committed, record.window_key
        self.committed[record.window_key] = record.tier
