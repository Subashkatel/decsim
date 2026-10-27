"""A stream's measures folded window by window (stream_measures.py).

The referent is measure_shot over the same run's whole ledgers: every
number both compute must be the same, the growth rates to 1e-9 (an
exact fit in ticks against statistics.linear_regression in floats). A
run that lets each window go must measure what the whole run does, and
leave no window, stage, frame write or round event behind.
"""

import collections
import dataclasses
import math

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.stream_command as stream_command
import decsim.experiments.stream_measures as stream_measures
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.presets.machines as machines
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.records.workload as workload_records
import decsim.windows.schemes.region_sliding as region_sliding
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.patch_pools as patch_pools
import tests.stream_machines as stream_machines

SEED = 17
RUNS = [
    stream_machines.StreamRun(60, 0.5, 1, False),
    stream_machines.StreamRun(120, 4.0, 20, False),
    stream_machines.StreamRun(90, 0.5, 1, True),
    # long enough that the machine lets windows go (window_release.py)
    stream_machines.StreamRun(600, 0.5, 1, True),
]
# measure_shot's names for the pool numbers the stream keeps together
POOL_FIELDS = (
    "weak_queue_max",
    "strong_queue_max",
    "weak_busy_fraction",
    "strong_busy_fraction",
    "weak_offered_load",
    "strong_offered_load",
)


def _run(graph):
    """(machine, result, the stream's measurement) of one run."""
    machine = machine_module.Machine.build(graph, SEED)
    measures = stream_measures.StreamMeasures(machine)
    result = machine.run()
    folded = measures.measurement(result)
    return machine, result, folded


def _differences(folded, shot) -> list:
    """Every number the stream and the shot both read, where they differ."""
    pairs = [
        ("decoded", folded.decoded_windows, shot.decoded_windows),
        ("failure", folded.logical_failure, shot.logical_failure),
        ("rounds", folded.rounds, shot.executed_rounds),
        ("recorded", folded.recorded_windows, len(shot.window_tiers)),
        (
            "missed",
            folded.deadline_missed_windows,
            shot.deadline_missed_windows,
        ),
        (
            "first miss",
            folded.first_deadline_miss_us,
            shot.first_deadline_miss_us,
        ),
        ("escalated", folded.escalated_windows, shot.escalated_windows),
        (
            "strong rounds",
            folded.strong_decoded_rounds,
            shot.strong_decoded_rounds,
        ),
        ("service", folded.strong_service_sum_us, shot.strong_service_sum_us),
        (
            "windows/us",
            folded.throughput_windows_per_us,
            shot.throughput_windows_per_us,
        ),
        (
            "rounds/us",
            folded.throughput_rounds_per_us,
            shot.throughput_rounds_per_us,
        ),
        ("queued", folded.max_queued_windows, shot.max_queued_windows),
        ("backlog", folded.backlog_peak_rounds, shot.backlog_peak_rounds),
        ("statuses", folded.window_statuses, shot.window_statuses),
        ("unscored", folded.unscored_reason, shot.unscored_reason),
        ("links", folded.link_totals, shot.link_totals),
        ("statistics", folded.statistics, shot.statistics),
    ]
    value_pairs = _value_pairs(folded, shot)
    pool_pairs = _pool_pairs(folded, shot)
    pairs.extend(value_pairs)
    pairs.extend(pool_pairs)
    tiers = collections.Counter(shot.window_tiers)
    pairs.append(("tiers", folded.window_tiers, dict(tiers)))
    end_samples = shot.samples["end_of_stream_reaction"]
    pairs.append(("end", [folded.end_of_stream_reaction_us], end_samples))
    differences = [(name, a, b) for name, a, b in pairs if a != b]
    growth_differences = _growth_differences(folded, shot)
    differences.extend(growth_differences)
    return differences


def _kept_whole(task: collect.Task, rounds: int, warmup: int):
    """The task's patches as a stream that keeps every ledger whole.

    measure_shot reads the whole ledgers; the stream folds as it goes.
    """
    streamed = stream_command.stream_machine(task, rounds, warmup)
    observation = dataclasses.replace(streamed.observation, folds_windows=False)
    return dataclasses.replace(streamed, observation=observation)


def _value_pairs(folded, shot) -> list:
    pairs = []
    for name, summary in folded.values.items():
        samples = shot.samples[name]
        pairs.append((f"{name} count", summary["count"], len(samples)))
        pairs.append((f"{name} mean", summary["mean"], shot.means[name]))
        pairs.append((f"{name} maximum", summary["maximum"], shot.maxes[name]))
    return pairs


def _pool_pairs(folded, shot) -> list:
    pairs = [
        (name, folded.pools[name], getattr(shot, name)) for name in POOL_FIELDS
    ]
    shot_rows = [dataclasses.asdict(row) for row in shot.pools]
    pairs.append(("pool rows", folded.pool_rows, shot_rows))
    return pairs


def _growth_differences(folded, shot) -> list:
    """The growth rates, each None in both or the same to 1e-9."""
    pairs = [
        (folded.weak_reaction_growth_rate, shot.weak_reaction_growth_rate),
        (folded.strong_reaction_growth_rate, shot.strong_reaction_growth_rate),
    ]
    return [
        ("growth", folded_rate, shot_rate)
        for folded_rate, shot_rate in pairs
        if not _same_rate(folded_rate, shot_rate)
    ]


def _same_rate(folded_rate, shot_rate) -> bool:
    if folded_rate is None or shot_rate is None:
        return folded_rate is shot_rate
    return math.isclose(folded_rate, shot_rate, rel_tol=1e-9)


@pytest.mark.parametrize("run", RUNS)
def test_the_stream_measures_what_measure_shot_reads_off_the_whole_run(run):
    graph = stream_machines.stream_graph(run, folds=False)
    machine, result, folded = _run(graph)
    task = collect.Task("stream", graph, {})
    shot = collect.Shot(task, SEED, machine, result, 0.0)

    measured = measure.measure_shot(shot)

    assert _differences(folded, measured) == []
    assert folded.decoded_windows > 10
    assert folded.escalated_windows > 0 or not run.escalates
    is_overloaded = run.weak_decode_microseconds > 1.0
    assert folded.deadline_missed_windows > 0 or not is_overloaded


@pytest.mark.parametrize("run", RUNS)
def test_a_stream_that_lets_each_window_go_measures_the_same_and_keeps_none(
    run,
):
    whole = stream_machines.stream_graph(run, folds=False)
    lean = stream_machines.stream_graph(run, folds=True)
    _machine, _result, kept = _run(whole)

    machine, _result, folded = _run(lean)

    observation = machine.observation
    assert folded == kept
    assert observation.windows.windows == {}
    assert observation.stages.records_for(100, 0) == ()
    assert observation.frame_corrections.committed == []
    assert observation.round_events.events == []


def test_patches_on_a_pool_each_beside_a_shared_strong_pool_measure_alike():
    shared = patch_pools.shared_strong_per_patch()
    task = collect.Task("patches", shared, {})
    graph = _kept_whole(task, 30, 0)
    machine, result, folded = _run(graph)
    stream_task = collect.Task("patches", graph, {})
    shot = collect.Shot(stream_task, SEED, machine, result, 0.0)

    measured = measure.measure_shot(shot)

    pools = [row["pool"] for row in folded.pool_rows]
    assert _differences(folded, measured) == []
    assert pools == ["weak", "w1", "w2", "strong"]


def test_a_pools_row_counts_the_windows_committed_past_the_warmup():
    one_pool = patch_pools.one_pool()
    per_patch = patch_pools.pool_per_patch(one_pool)
    task = collect.Task("patches", per_patch, {})
    graph = _kept_whole(task, 30, 10)
    machine, result, folded = _run(graph)
    stream_task = collect.Task("patches", graph, {})
    shot = collect.Shot(stream_task, SEED, machine, result, 0.0)

    measured = measure.measure_shot(shot)

    committed = [row["committed_windows"] for row in folded.pool_rows]
    assert _differences(folded, measured) == []
    assert 0 < sum(committed) < folded.decoded_windows


def test_the_escalated_share_is_sampled_off_the_frames_commits_in_order():
    graph = stream_machines.stream_graph(RUNS[3], folds=False)
    machine, _result, folded = _run(graph)
    observation = machine.observation
    windows = observation.windows.windows
    _weak_pools, strong_pools = machine.layout.tier_pools()
    every = stream_measures.ESCALATION_SAMPLED_EVERY
    expected = []
    escalated = 0
    commits = observation.frame_corrections.committed
    for count, record in enumerate(commits, start=1):
        escalated += window_records.PoolId(record.tier) in strong_pools
        if count % every == 0:
            first_round = windows[record.window_key].commit_lo
            expected.append((count, escalated, first_round))

    assert folded.escalation_series == expected
    assert len(expected) >= 2
    assert expected[-1][1] > 0


READ_HOP_MICROSECONDS = 5.0


def _two_patch_graph(folds: bool):
    """Two patches idling a round each, decoded on two units at once.

    Each stream counts its windows from 0, so both lay a window 0, and
    every decode's input rides the one read hop.
    """
    memory = program_records.OpKind.MEMORY
    operations = tuple(
        program_records.Operation(
            index, f"memory {index}", ((0, index),), kind=memory
        )
        for index in range(2)
    )
    program = workload_records.Workload(
        operations, {0: 1, 1: 1}, patch_streams=True
    )
    settings = workload_settings.WorkloadSettings.running(program)
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme)
    hops = dict.fromkeys(declared_run.DECLARED_HOP_MICROSECONDS, 0.0)
    hops["read_written"] = READ_HOP_MICROSECONDS
    fabric = declared_run.declared_fabric(**hops)
    observation = observe_settings.ObservationSettings(folds_windows=folds)
    qpu = qpu_settings.QpuSettings(distance=3)
    base = machines.base(
        settings, windows, qpu, observation=observation, fabric=fabric
    )
    algorithm = decoders.PresetLatencyDecoder.Settings(0.1)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE, unit_count=2
    )
    pooled = machines.with_pool(base, pool)
    return machines.with_frame(pooled)


def test_each_streams_window_0_keeps_its_own_input_hop_when_let_go():
    whole = _two_patch_graph(folds=False)
    lean = _two_patch_graph(folds=True)
    _machine, _result, kept = _run(whole)

    _machine, _result, folded = _run(lean)

    input_hop = folded.values["input_link_per_window"]
    assert input_hop["count"] == 2
    assert input_hop["mean"] == READ_HOP_MICROSECONDS
    assert folded == kept
