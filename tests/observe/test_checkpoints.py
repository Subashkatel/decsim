"""Each checkpoint of a stream, closed as if the stream ended there.

The referents are the stream's own end and separate streams of each
checkpoint's rounds, on the same seed. The machine decodes every window
with PyMatching, as the referee does, so the window that closes a
checkpoint is the separate stream's last window, from the same noise,
and the outcomes must be the same, seed by seed. A checkpoint keeps the
windows that read only rounds before it (Tan et al. 2209.09219 lines
1003-1037), the closing window reads the boundary the last of them
hands on, as the next window would (windows/decode_requests.py), and a
listener only listens: the stream's numbers are the same with
checkpoints and without.
"""

import dataclasses

import pytest

import decsim
import decsim.experiments.stream_command as stream_command
import decsim.machine as machine_module
import decsim.observe.checkpoints as checkpoints_module
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.records.checkpoints as checkpoint_records
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# p = 0.01 at d = 3 fails about one stream of 40 rounds in three
PROBABILITY = 0.01
ROUNDS = 40
# the checkpoints before the end of a stream checkpointed every 10
EARLY_CHECKPOINTS = (10, 20, 30)


def _task(buffer_rounds: int, every_rounds) -> decsim.Task:
    """PyMatching alone at d = 3, read out cleanly, checkpointed."""
    machine = machines.weak_baseline(3, PROBABILITY, 1.0)
    windows = machine.held("windows")
    scheme = dataclasses.replace(windows.scheme, buffer_rounds=buffer_rounds)
    sized = dataclasses.replace(windows, scheme=scheme)
    workload = machine.held("workload")
    clean = dataclasses.replace(workload, noiseless_time_boundaries=True)
    observation = dataclasses.replace(
        machine.observation, checkpoint_every_rounds=every_rounds
    )
    machine = machine.with_held("windows", sized)
    machine = machine.with_held("workload", clean)
    machine = dataclasses.replace(machine, observation=observation)
    return decsim.Task("memory", machine)


def _measurement(task: decsim.Task, seed: int) -> dict:
    record = stream_command.stream_record(task, ROUNDS, 0, seed)
    return record["measurement"]


def _early_failures(measurement: dict) -> list:
    rows = measurement["checkpoints"][:-1]
    return [row["failed"][0] for row in rows]


def _separate_failures(task: decsim.Task, seed: int) -> list:
    """The outcomes of separate streams of the early checkpoints' rounds."""
    return [_stream_failed(task, rounds, seed) for rounds in EARLY_CHECKPOINTS]


def _stream_failed(task: decsim.Task, rounds: int, seed: int) -> bool:
    record = stream_command.stream_record(task, rounds, 0, seed)
    measurement = record["measurement"]
    return measurement["logical_failure"]


def test_the_last_checkpoint_is_the_streams_own_end_outcome():
    task = _task(0, 10)

    measurements = [_measurement(task, seed) for seed in range(8)]

    last_rows = [measurement["checkpoints"][-1] for measurement in measurements]
    ends = [[measurement["logical_failure"]] for measurement in measurements]
    assert [(row["round"], row["failed"]) for row in last_rows] == [
        (ROUNDS, end) for end in ends
    ]
    assert [False] in ends
    assert [True] in ends


def test_a_checkpoint_is_a_separate_stream_of_its_rounds_seed_by_seed():
    """A buffer of d rounds: the closing window is the separate one's last."""
    checkpointed_task = _task(3, 10)
    plain_task = _task(3, None)
    seeds = range(8)

    measurements = [_measurement(checkpointed_task, seed) for seed in seeds]
    separate = [_separate_failures(plain_task, seed) for seed in seeds]

    early = [_early_failures(measurement) for measurement in measurements]
    assert early == separate
    assert [False, False, False] in separate
    assert [True, True, True] in separate
    assert [False, True, True] in separate


@pytest.mark.parametrize(
    ("buffer_rounds", "kept_windows"),
    [(0, [3, 6, 9, 13]), (3, [2, 5, 8, 12])],
)
def test_a_checkpoint_keeps_the_windows_that_read_only_earlier_rounds(
    buffer_rounds, kept_windows
):
    """At d = 3 window k commits 3k+1 to 3k+3 and reads buffer_rounds more."""
    task = _task(buffer_rounds, 10)

    measurement = _measurement(task, 1)

    rows = measurement["checkpoints"]

    assert [row["round"] for row in rows] == [10, 20, 30, 40]
    assert [row["windows"] for row in rows] == kept_windows
    assert [row["escalated_windows"] for row in rows] == [0, 0, 0, 0]


def test_checkpoints_change_no_number_of_the_stream():
    checkpointed_task = _task(3, 5)
    plain_task = _task(3, None)

    checkpointed = _measurement(checkpointed_task, 2)
    plain = _measurement(plain_task, 2)

    rows = checkpointed.pop("checkpoints")
    plain.pop("checkpoints")
    assert checkpointed == plain
    assert len(rows) == ROUNDS // 5


def test_the_closing_window_reads_the_boundary_the_last_kept_one_hands_on():
    """Noiseless rounds; window 0 commits 1 to 3 and hands detector 21 on.

    d = 3: round 4's detectors are 20 to 27. Window 1 reads round 9, so
    the checkpoint at 9 keeps window 0 alone and closes 4 to 9.
    """
    referee = _RecordingReferee()
    listener = checkpoints_module.Checkpoints(9, referee)
    kept = window_records.Window(100, 0, 1, 3, 6, 6)
    reading = window_records.Window(100, 1, 4, 6, 9, 6)
    pool = window_records.PoolId("weak")
    key = window_records.DecoderRequestKey(100, 0, pool, 1)
    kept.published_request_key = key
    boundary = window_records.DependencyResidual(detector_ids=(5, 21))
    result = decoding_records.DecodeResult(
        100, 0, logical_observables=(0,), boundary_data=boundary
    )
    job = decoding_records.DecodeJob(100, 0, 6, request_key=key)

    listener.window_planned(kept)
    listener.window_planned(reading)
    _emit_noiseless_rounds(listener, 9)
    listener.request_ended(job, result, None, None)
    listener.window_final(kept)
    snapshot = _noiseless_snapshot(9)
    listener.checkpoint_reached(snapshot)

    syndrome = referee.syndromes[0]
    flipped = [detector for detector, bit in syndrome.items() if bit]
    assert flipped == [21]
    assert min(syndrome) == 20
    assert listener.outcomes[0].windows == 1


def test_checkpoints_on_a_source_that_copies_nothing_are_refused():
    task = _task(3, 10)
    machine = task.machine

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(machine, 1)

    assert "checkpoint_every_rounds needs a source that copies" in str(
        refusal.value
    )


class _RecordingReferee:
    """PyMatching that keeps each syndrome it is handed, by detector."""

    def __init__(self) -> None:
        self.decoder = minimum_weight_perfect_matching.PyMatchingDecoder()
        self.fault_model_requirement = self.decoder.fault_model_requirement
        self.syndromes = []

    def decode(self, job):
        model = job.detector_error_model
        bits = job.payloads[0].bits
        syndrome = dict(zip(model.detector_ids, bits, strict=True))
        self.syndromes.append(syndrome)
        return self.decoder.decode(job)


def _emit_noiseless_rounds(listener, round_count: int) -> None:
    """The raw rounds of a d = 3 memory with no noise: no event fires."""
    program = memory_programs.memory_program(3, "Z", 0.0)
    plan = flip_stream_device.RoundPlan(program, 10)
    history = flip_stream_device.FlipHistory(plan, 1, False)
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        bits = history.run_round(False)
        readout = round_records.QPUReadout(100, (0,), round_index, bits=bits)
        listener.round_emitted(readout)


def _noiseless_snapshot(round_index: int):
    """The checkpoint of a noiseless stream, on a noisy program's model."""
    program = memory_programs.memory_program(3, "Z", 0.001)
    return checkpoint_records.CheckpointSnapshot(
        100, round_index, (0, 0, 0, 0), (0,), False, program
    )
