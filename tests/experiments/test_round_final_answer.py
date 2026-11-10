"""The final-answer latency of every syndrome round (round_to_final_answer).

One sample a round: from the round leaving the QPU to the frame commit of
the answer that covers it. A window counts by the rounds it commits. On
Toshio's double window a switched window's region answers the rounds of
the windows after it too, so their rounds end at the region's commit. A
stream folds the same samples as it runs.
"""

import dataclasses

import decsim
import decsim.confidence.cluster as cluster
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.stream_command as stream_command
import decsim.experiments.stream_measures as stream_measures
import decsim.machine as machine_module
import decsim.presets.experiment_one as experiment_one
import decsim.presets.machines as machines
import tests.experiments.test_window_final_answer as window_final_answer
import tests.patch_pools as patch_pools
from decsim.decoders.belief_matching import decoder as belief_matching

POINT = "round_to_final_answer"
WINDOW_POINT = window_final_answer.POINT
ROUNDS = window_final_answer.ROUNDS
STREAM_ROUNDS = 150


def _samples_by_round(shot, measurement) -> dict:
    """Each round's sample, as the point lists them: window, then round."""
    rounds = []
    for _key, window in window_final_answer._sorted_windows(shot):
        commit_rounds = measure.commit_round_indices(window)
        rounds.extend(commit_rounds)
    pairs = zip(rounds, measurement.samples[POINT], strict=True)
    return dict(pairs)


def _every_round() -> list:
    last = ROUNDS + 1
    every = range(1, last)
    return list(every)


def test_every_round_counts_once_when_no_window_switches():
    shot, measurement = window_final_answer._shot("never")

    by_round = _samples_by_round(shot, measurement)

    assert sorted(by_round) == _every_round()


def test_every_round_counts_once_when_every_window_switches():
    shot, measurement = window_final_answer._shot("always")
    windows = window_final_answer._sorted_windows(shot)

    by_round = _samples_by_round(shot, measurement)

    assert [window for _key, window in windows if window.is_absorbed]
    assert sorted(by_round) == _every_round()


def _deviations(verdict: str) -> list:
    """Each round's sample less its window's sample and the rounds between.

    The window's clock starts at its last round, buffer included and cut
    at the shot's last round; the round's starts at the round itself.
    """
    shot, measurement = window_final_answer._shot(verdict)
    windows = window_final_answer._sorted_windows(shot)
    period = window_final_answer._round_period_microseconds(shot)
    by_round = _samples_by_round(shot, measurement)
    window_samples = measurement.samples[WINDOW_POINT]
    deviations = []
    pairs = zip(windows, window_samples, strict=True)
    for (_key, window), window_sample in pairs:
        last_round = min(window.buffer_hi, ROUNDS)
        for round_index in measure.commit_round_indices(window):
            rounds_before = last_round - round_index
            expected = window_sample + rounds_before * period
            deviation = by_round[round_index] - expected
            deviations.append(deviation)
    return deviations


def test_a_round_waits_its_windows_time_and_the_rounds_up_to_its_last():
    kept = _deviations("never")
    switched = _deviations("always")

    # up to the rounding of the ticks
    assert max(abs(deviation) for deviation in kept + switched) < 0.01


def _double_window_stream(folds: bool):
    """Experiment 1's union-find stream, escalating under 20 dB."""
    union_find = experiment_one.weak_alone(3, 0.01)
    confidence = cluster.ClusterGap.Settings(walk_microseconds=0.2)
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    strong_window = strong_window_shapes.DoubleWindow.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    times = belief_matching.IterationTime(
        iteration_microseconds=0.5, matching_microseconds=10.0
    )
    algorithm = belief_matching.BeliefMatchingDecoder.Settings(
        max_iterations=30, timing=times
    )
    strong = dataclasses.replace(
        machines.STRONG_BASELINE_POOL, algorithm=algorithm
    )
    machine = machines.with_strong_tier(union_find, strong, switching)
    task = decsim.Task("double", machine)
    graph = stream_command.stream_machine(task, STREAM_ROUNDS, 0)
    observation = dataclasses.replace(graph.observation, folds_windows=folds)
    return task, dataclasses.replace(graph, observation=observation)


def _stream(folds: bool):
    """(task, machine, result, the stream's measurement) of one stream."""
    task, graph = _double_window_stream(folds)
    machine = machine_module.Machine.build(graph, patch_pools.SEED)
    measures = stream_measures.StreamMeasures(machine)
    result = machine.run()
    return task, machine, result, measures.measurement(result)


def test_a_double_window_stream_folds_every_round_as_the_shot_reads_them():
    task, machine, result, folded = _stream(folds=False)
    shot = collect.Shot(task, patch_pools.SEED, machine, result, 0.0)

    measured = measure.measure_shot(shot)

    summary = folded.values[POINT]
    assert folded.escalated_windows > 0
    assert summary["count"] == len(measured.samples[POINT]) == STREAM_ROUNDS
    assert summary["mean"] == measured.means[POINT]
    assert summary["maximum"] == measured.maxes[POINT]


def test_a_stream_that_lets_its_windows_go_folds_the_same_rounds():
    _task, _machine, _result, kept = _stream(folds=False)

    _task, _machine, _result, folded = _stream(folds=True)

    assert folded.values[POINT] == kept.values[POINT]
