"""A stream's measures folded window by window (stream_measures.py).

The referent is measure_shot over the same run's whole ledgers: every
number both compute must be the same, the growth rates to 1e-9 (an
exact fit in ticks against statistics.linear_regression in floats). A
run that lets each window go must measure what the whole run does, and
leave no window, stage, frame write or round event behind.
"""

import collections
import math

import pytest

import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.stream_measures as stream_measures
import decsim.machine as machine_module
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


def _run(settings):
    """(machine, result, the stream's measurement) of one run."""
    machine = machine_module.Machine.build(settings, SEED)
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


def _value_pairs(folded, shot) -> list:
    pairs = []
    for name, summary in folded.values.items():
        samples = shot.samples[name]
        pairs.append((f"{name} count", summary["count"], len(samples)))
        pairs.append((f"{name} mean", summary["mean"], shot.means[name]))
        pairs.append((f"{name} maximum", summary["maximum"], shot.maxes[name]))
    return pairs


def _pool_pairs(folded, shot) -> list:
    return [
        (name, folded.pools[name], getattr(shot, name)) for name in POOL_FIELDS
    ]


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
    settings = stream_machines.stream_settings(run, folds=False)
    machine, result, folded = _run(settings)
    task = collect.Task("stream", settings, {})
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
    whole = stream_machines.stream_settings(run, folds=False)
    lean = stream_machines.stream_settings(run, folds=True)
    _machine, _result, kept = _run(whole)

    machine, _result, folded = _run(lean)

    observation = machine.observation
    assert folded == kept
    assert observation.windows.windows == {}
    assert observation.stages.records_for(100, 0) == ()
    assert observation.frame_corrections.committed == []
    assert observation.round_events.events == []
