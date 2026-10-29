"""Each checkpoint of a stream, closed as if the stream ended there.

The referents are the stream's own end and separate streams of each
checkpoint's rounds, on the same seed. A checkpoint is the machine's
own answer had the run ended there (Tan et al. 2209.09219 lines
1003-1037), so the outcomes must be the same, seed by seed, on three
machines: PyMatching alone on sliding windows and on parallel blocks,
and union-find escalating weak windows to belief matching at a fixed or
a moving threshold. A checkpoint keeps the windows that read only
rounds before it, but a parallel B block whose later A reads further,
each tail window reads the boundaries it takes, as the next window
would (windows/decode_requests.py), and a listener only listens: the
stream's numbers, its threshold's among them, are the same with
checkpoints and without.
"""

import dataclasses

import pytest

import decsim
import decsim.confidence.cluster as cluster
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find_decoder
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.experiments.stream_command as stream_command
import decsim.machine as machine_module
import decsim.observe.checkpoints as checkpoints_module
import decsim.presets.clocks as clocks
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.records.checkpoints as checkpoint_records
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.windows.schemes.parallel as parallel
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# p = 0.01 at d = 3 fails about one stream of 40 rounds in three
PROBABILITY = 0.01
ROUNDS = 40
# the checkpoints before the end of a stream checkpointed every 10
EARLY_CHECKPOINTS = (10, 20, 30)
# the switching streams: 20 rounds, checkpointed at 10 and at the end;
# at d = 3 each ends on two tail windows, one region cut short
SWITCHING_ROUNDS = 20
# a weak window under 10 dB escalates; a moving threshold holds three
# windows in ten escalated from there
FIXED = threshold_sources.FixedThreshold.Settings(threshold_decibels=10.0)
RATE = threshold_sources.RateThreshold.Settings(10.0, 0.3)
# seeds whose outcome a closing escalation (20 and 35) or the tail's two
# windows (11) changes, beside ones that fail throughout or never
SWITCHING_SEEDS = (0, 4, 11, 20, 35)
# the parallel blocks at d = 3 (schemes/parallel.py): checkpoints every 5
# rounds of 40 close on the first A alone (5), a B alone (10, 15, 25,
# 35), a B again beside the A after it (30), and a B, an A and a short
# B (20)
PARALLEL_EVERY = 5
PARALLEL_CHECKPOINTS = (5, 10, 15, 20, 25, 30, 35)
# union-find priced by its cycle count and belief matching by its
# iterations, at the basic case's stated times, so two runs price alike
CYCLE_COUNT = cycle_count.CycleCount(clock=clocks.FRIDGE_CLOCK)
UNION_FIND = union_find_decoder.UnionFindDecoder.Settings(timing=CYCLE_COUNT)
ITERATION_TIME = belief_matching.IterationTime(1.0, 10.0)
BELIEF_MATCHING = belief_matching.BeliefMatchingDecoder.Settings(
    timing=ITERATION_TIME
)


def _task(buffer_rounds: int, every_rounds) -> decsim.Task:
    """PyMatching alone at d = 3, read out cleanly, checkpointed."""
    machine = machines.weak_baseline(3, PROBABILITY, 1.0)
    return _checkpointed(machine, buffer_rounds, every_rounds)


def _switching_task(threshold, every_rounds, verdict=None) -> decsim.Task:
    """Union-find escalating to belief matching on its cluster gap, d = 3.

    Both decoders and the gap's walk are priced at fixed times, so two
    runs differ only by what their listeners do.
    """
    machine = machines.weak_baseline(3, PROBABILITY, 1.0)
    weak = machine.with_settings(machines.WEAK_POOL, algorithm=UNION_FIND)
    confidence = cluster.ClusterGap.Settings(walk_microseconds=0.0)
    redo = strong_window_shapes.RedoWindow.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=redo
    )
    if verdict is not None:
        switching = dataclasses.replace(switching, verdict=verdict)
    strong = dataclasses.replace(
        machines.STRONG_BASELINE_POOL, algorithm=BELIEF_MATCHING
    )
    machine = machines.with_strong_tier(weak, strong, switching)
    return _checkpointed(machine, 3, every_rounds)


def _checkpointed(machine, buffer_rounds: int, every_rounds) -> decsim.Task:
    """The machine read out cleanly, its buffer sized, checkpointed."""
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


def _parallel_task(every_rounds) -> decsim.Task:
    """PyMatching alone on Skoric's parallel blocks, read out cleanly."""
    machine = machines.weak_baseline(3, PROBABILITY, 1.0)
    windows = machine.held("windows")
    scheme = parallel.ParallelWindowScheme.Settings()
    on_blocks = dataclasses.replace(windows, scheme=scheme)
    machine = machine.with_held("windows", on_blocks)
    return _checkpointed(machine, 3, every_rounds)


def _parallel_separate_failures(task: decsim.Task, seed: int) -> list:
    """The outcomes of separate parallel streams of the checkpoints' rounds."""
    return [
        _stream_failed(task, rounds, seed) for rounds in PARALLEL_CHECKPOINTS
    ]


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


def _switching_record(task: decsim.Task, seed: int) -> dict:
    return stream_command.stream_record(task, SWITCHING_ROUNDS, 0, seed)


def _checkpoint_failures(record: dict) -> list:
    measurement = record["measurement"]
    rows = measurement["checkpoints"]
    return [row["failed"][0] for row in rows]


def _halfway_and_end_failures(task: decsim.Task, seed: int) -> list:
    """A separate stream of half the rounds, and the stream's own end."""
    halfway = SWITCHING_ROUNDS // 2
    early = _stream_failed(task, halfway, seed)
    end = _stream_failed(task, SWITCHING_ROUNDS, seed)
    return [early, end]


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


@pytest.mark.parametrize("threshold", [FIXED, RATE], ids=["fixed", "rate"])
def test_a_switching_checkpoint_is_a_separate_stream_of_its_rounds_seed_by_seed(
    threshold,
):
    """Its tail windows, verdicts and escalations are the separate stream's."""
    checkpointed_task = _switching_task(threshold, 10)
    plain_task = _switching_task(threshold, None)
    seeds = SWITCHING_SEEDS

    records = [_switching_record(checkpointed_task, seed) for seed in seeds]
    separate = [_halfway_and_end_failures(plain_task, seed) for seed in seeds]

    checkpoints = [_checkpoint_failures(record) for record in records]
    escalated = [
        record["measurement"]["escalated_windows"] for record in records
    ]
    assert checkpoints == separate
    assert [False, False] in separate
    assert [True, True] in separate
    assert sum(escalated) > 0


def test_a_parallel_checkpoint_is_a_separate_stream_seed_by_seed():
    """Its tail holds the A and B blocks a stream ending there lays."""
    checkpointed_task = _parallel_task(PARALLEL_EVERY)
    plain_task = _parallel_task(None)
    seeds = range(8)

    measurements = [_measurement(checkpointed_task, seed) for seed in seeds]
    separate = [_parallel_separate_failures(plain_task, seed) for seed in seeds]

    early = [_early_failures(measurement) for measurement in measurements]
    assert early == separate
    assert [False] * len(PARALLEL_CHECKPOINTS) in separate
    assert any(True in failures for failures in separate)


def test_checkpoints_teach_a_moving_threshold_nothing():
    checkpointed_task = _switching_task(RATE, 2)
    plain_task = _switching_task(RATE, None)

    checkpointed = _switching_record(checkpointed_task, 3)
    plain = _switching_record(plain_task, 3)

    checkpointed["measurement"].pop("checkpoints")
    plain["measurement"].pop("checkpoints")
    assert checkpointed["measurement"] == plain["measurement"]
    assert checkpointed["online_threshold"] == plain["online_threshold"]


def test_a_row_holds_the_threshold_its_edge_reads_and_none_without_one():
    switching_task = _switching_task(FIXED, 10)
    alone_task = _task(3, 10)

    switching = _switching_record(switching_task, 1)
    alone = _switching_record(alone_task, 1)

    switching_rows = switching["measurement"]["checkpoints"]
    alone_rows = alone["measurement"]["checkpoints"]
    assert [row["threshold"] for row in switching_rows] == [
        [FIXED.threshold_nats],
        [FIXED.threshold_nats],
    ]
    assert [row["threshold"] for row in alone_rows] == [[None], [None]]


def test_checkpoints_beside_a_verdict_that_draws_are_refused():
    bernoulli = verdicts.Bernoulli.Settings(rate=0.1)
    task = _switching_task(FIXED, 10, verdict=bernoulli)

    with pytest.raises(NotImplementedError) as refusal:
        _switching_record(task, 1)

    assert "Bernoulli draws from the run's seed" in str(refusal.value)


def test_checkpoints_beside_an_edge_past_the_entry_pool_are_refused():
    """Union-find, union-find again, then belief matching: two edges."""
    machine = machines.weak_baseline(3, PROBABILITY, 1.0)
    weak = machine.with_settings(machines.WEAK_POOL, algorithm=UNION_FIND)
    confidence = cluster.ClusterGap.Settings(walk_microseconds=0.0)
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=FIXED
    )
    pool = machines.STRONG_BASELINE_POOL
    middle = dataclasses.replace(pool, algorithm=UNION_FIND)
    last = dataclasses.replace(pool, algorithm=BELIEF_MATCHING)
    two_tiers = machines.with_strong_tier(weak, middle, switching)
    strong = machines.STRONG_POOL
    three_tiers = machines.with_escalation(
        two_tiers, strong, "last", last, switching
    )
    task = _checkpointed(three_tiers, 3, 10)

    with pytest.raises(NotImplementedError) as refusal:
        _switching_record(task, 1)

    assert "an escalate edge past the entry pool (strong to last)" in str(
        refusal.value
    )


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
    the checkpoint at 9 keeps window 0 alone; windows 1 and 2 close it.
    """
    closer = _RecordingCloser()
    listener = checkpoints_module.Checkpoints(9, closer)
    kept = window_records.Window(100, 0, 1, 3, 6, 6)
    reading = window_records.Window(100, 1, 4, 6, 9, 6)
    holding = window_records.Window(100, 2, 7, 9, 12, 6)

    listener.window_planned(kept)
    listener.window_planned(reading)
    listener.window_planned(holding)
    _emit_noiseless_rounds(listener, 9)
    _decided(listener, kept, (5, 21))
    snapshot = _noiseless_snapshot(9)
    listener.checkpoint_reached(snapshot)

    syndrome = closer.syndromes[0]
    flipped = [detector for detector, bit in syndrome.items() if bit]
    assert flipped == [21]
    assert min(syndrome) == 20
    assert len(closer.syndromes) == 2
    assert listener.outcomes[0].windows == 1


def test_a_b_block_whose_later_a_reads_past_the_checkpoint_closes_again():
    """Noiseless rounds at d = 3; the first A hands detector 45 on.

    The parallel blocks: the first A commits 1 to 6 and reads to 9, the B
    commits 7 to 15, the second A commits 16 to 18 reading 13 to 21. At
    17 the B reads only earlier rounds but takes the second A's boundary,
    which reads round 17, so the first A alone is kept. The second A is
    decided first, on its own rounds 13 to 17 (round 13 starts at
    detector 92), then the B (round 7 starts at 44) with the boundary.
    """
    closer = _RecordingCloser()
    listener = checkpoints_module.Checkpoints(17, closer)
    first_a = window_records.Window(100, 0, 1, 6, 9, 9)
    b_block = window_records.Window(
        100, 1, 7, 15, 15, 9, closed_temporal_boundaries=True
    )
    second_a = window_records.Window(100, 2, 16, 18, 21, 9, buffer_lo=13)

    listener.window_planned(first_a)
    listener.window_planned(b_block)
    listener.window_planned(second_a)
    _emit_noiseless_rounds(listener, 17)
    _decided(listener, first_a, (45,))
    snapshot = _noiseless_snapshot(17)
    listener.checkpoint_reached(snapshot)

    second_a_syndrome, b_syndrome = closer.syndromes
    second_a_bits = second_a_syndrome.values()
    flipped = [detector for detector, bit in b_syndrome.items() if bit]
    assert min(second_a_syndrome) == 92
    assert not any(second_a_bits)
    assert min(b_syndrome) == 44
    assert flipped == [45]
    assert listener.outcomes[0].windows == 1


def test_checkpoints_on_a_source_that_copies_nothing_are_refused():
    task = _task(3, 10)
    machine = task.machine

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(machine, 1)

    assert "checkpoint_every_rounds needs a source that copies" in str(
        refusal.value
    )


class _RecordingCloser:
    """PyMatching that keeps each syndrome it is handed, by detector."""

    def __init__(self) -> None:
        self.decoder = minimum_weight_perfect_matching.PyMatchingDecoder()
        self.requirement = self.decoder.fault_model_requirement
        self.syndromes = []

    def threshold_of(self, stream_id) -> None:
        del stream_id

    def decide(self, job):
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


def _decided(listener, window, boundary_detector_ids: tuple) -> None:
    """The window decoded, handing these detectors on, and final."""
    stream_id = window.stream_id
    window_index = window.window_index
    pool = window_records.PoolId("weak")
    key = window_records.DecoderRequestKey(stream_id, window_index, pool, 1)
    window.published_request_key = key
    boundary = window_records.DependencyResidual(
        detector_ids=boundary_detector_ids
    )
    result = decoding_records.DecodeResult(
        stream_id,
        window_index,
        logical_observables=(0,),
        boundary_data=boundary,
    )
    job = decoding_records.DecodeJob(
        stream_id, window_index, window.round_count, request_key=key
    )
    listener.request_ended(job, result, None, None)
    listener.window_final(window)
