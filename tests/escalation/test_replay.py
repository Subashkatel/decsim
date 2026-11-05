"""Replaying a recorded run: its decisions, its decode times, its data.

A study that makes one part ideal keeps every other part of the run it
compares with. The recorded run writes each window's decision
(window_outcomes.csv) and each decode's time (decode_times.csv); the
replay escalates the same windows and holds each unit for the same
time, while its decoders answer on their own input. Replaying the
recorded run on its own decoders then gives the same shots exactly;
replaying it with the truth in the weak pool keeps every window, every
decision and every latency, and changes only the answers.
"""

import pytest

import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.report as report
import decsim.parameters.assignments as assignments_module
import decsim.parameters.tree as tree
import tests.experiments.run_files as run_files

SEEDS = range(8)
LATENCY_POINT = "qpu_last_round_to_frame"


def _machine(*pairs):
    """The test switching machine on Toshio's double window, recording."""
    machine = run_files.switching_machine(rounds_per_shot=30)
    base = (
        ("qpu.source.type", "error_model_stim"),
        ("observation.record_window_outcomes", "true"),
        ("observation.record_decode_times", "true"),
        ("switching.strong_window.type", "double_window"),
        ("windows.boundary_policy.type", "Eager"),
        ("windows.terminal_policy", "lookahead"),
        ("workload.arguments.physical_error_probability", "0.01"),
        # union-find at a stated time, its gap at a stated time, so the
        # recorded run's own times repeat
        ("weak.algorithm.type", "union_find"),
        ("weak.algorithm.timing.type", "PresetTime"),
        ("weak.algorithm.timing.latency", "0.3us"),
        ("switching.confidence.type", "cluster_gap"),
        ("switching.confidence.walk", "1us"),
    )
    texts = [f"{path}={value}" for path, value in base + pairs]
    written = assignments_module.from_texts(texts, "a test")
    applied = tree.apply(machine, written)
    return applied.settings


def _measured(machine):
    task = collect.Task("replay", machine)
    return [_measured_shot(task, seed) for seed in SEEDS]


def _measured_shot(task, seed):
    shot = collect.run_shot(task, seed)
    return measure.measure_shot(shot)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """The recorded run's measurements and its two files."""
    folder = tmp_path_factory.mktemp("recorded")
    machine = _machine()
    measurements = _measured(machine)
    decisions = folder / "window_outcomes.csv"
    times = folder / "decode_times.csv"
    outcome_rows = report.window_outcome_rows(measurements)
    time_rows = report.decode_time_rows(measurements)
    report.write_csv(outcome_rows, decisions)
    report.write_csv(time_rows, times)
    return measurements, str(decisions), str(times)


def _replay_pairs(decisions, times):
    return (
        ("switching.verdict.type", "replay"),
        ("switching.verdict.decisions_path", decisions),
        ("weak.replayed_decode_times", times),
        ("strong.replayed_decode_times", times),
    )


def _windows(measurement):
    """Each window's extent, tier and decision, in order."""
    return [
        (
            row.window_index,
            row.commit_lo,
            row.commit_hi,
            row.tier,
            row.is_escalated,
        )
        for row in measurement.window_outcomes
    ]


def _wrong_weak_windows(measurement):
    """The windows the weak tier kept whose answer is not their label."""
    rows = measurement.window_outcomes
    return [row for row in rows if row.tier == "weak" and not row.is_right]


def test_the_recorded_run_switches_some_windows(recorded):
    measurements, _, _ = recorded
    escalated = sum(
        bool(row.is_escalated)
        for measurement in measurements
        for row in measurement.window_outcomes
    )
    assert escalated > 0


def test_replaying_the_run_on_its_own_decoders_repeats_it_exactly(recorded):
    measurements, decisions, times = recorded
    pairs = _replay_pairs(decisions, times)
    machine = _machine(*pairs)
    replayed = _measured(machine)
    for before, after in zip(measurements, replayed, strict=True):
        assert after.sample_digest == before.sample_digest
        assert after.predictions == before.predictions
        assert after.window_outcomes == before.window_outcomes
        assert after.samples[LATENCY_POINT] == before.samples[LATENCY_POINT]
        assert after.decode_times == before.decode_times


def test_the_truth_in_the_weak_pool_keeps_windows_decisions_and_times(recorded):
    """Only the answers change: the weak ones become the true ones."""
    measurements, decisions, times = recorded
    pairs = _replay_pairs(decisions, times) + (
        ("weak.algorithm.type", "truth"),
        ("switching.confidence.type", "none"),
        ("switching.confidence.walk", "1us"),
    )
    machine = _machine(*pairs)
    replayed = _measured(machine)
    for before, after in zip(measurements, replayed, strict=True):
        assert after.sample_digest == before.sample_digest
        assert _windows(after) == _windows(before)
        assert after.samples[LATENCY_POINT] == before.samples[LATENCY_POINT]
        assert after.decode_times == before.decode_times
        assert _wrong_weak_windows(after) == []


def test_a_seed_the_recording_does_not_hold_is_refused(recorded):
    _, decisions, times = recorded
    pairs = _replay_pairs(decisions, times)
    machine = _machine(*pairs)
    task = collect.Task("replay", machine)

    with pytest.raises(
        ValueError, match="no shot of seed 99|no decode of seed 99"
    ):
        collect.run_shot(task, 99)
