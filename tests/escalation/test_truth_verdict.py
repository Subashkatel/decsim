"""The truth verdict: the perfect switching decision (escalation/verdicts.py).

The referent is the window record: window_outcomes.csv writes each
window's weak answer beside its label, the parity of the fired errors
whose first round lies in its commit rounds (observe/window_outcomes.py).
A truth verdict escalates exactly the windows whose weak answer is off
that label and keeps the rest. It draws nothing from the run's seed, so
it decodes the same shots as the gap's run. A source that keeps no
errors cannot serve it, so the build refuses the pair.
"""

import dataclasses

import pytest

import decsim.escalation.verdicts as verdicts
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.parameters.assignments as assignments_module
import decsim.parameters.tree as tree
import decsim.records.decoding as decoding_records
import tests.experiments.run_files as run_files

KEEP = decoding_records.Verdict.KEEP
ESCALATE = decoding_records.Verdict.ESCALATE
SEEDS = range(12)


@dataclasses.dataclass(frozen=True)
class _Error:
    first_round: int
    logical_observables: tuple


@dataclasses.dataclass(frozen=True)
class _Operation:
    id: str


@dataclasses.dataclass(frozen=True)
class _Window:
    stream_id: str
    commit_lo: int
    commit_hi: int
    key: tuple = ("stream", 0)


@dataclasses.dataclass(frozen=True)
class _Job:
    window: _Window


def _decided(fired_errors, answer, commit=(3, 5)):
    """The truth verdict's decision on one window of commit rounds COMMIT."""
    settings = verdicts.Truth.Settings()
    verdict = settings.build()
    operation = _Operation("stream")
    verdict.errors_sampled(operation, fired_errors)
    window = _Window("stream", *commit)
    job = _Job(window)
    result = decoding_records.DecodeResult(
        stream_id="stream", window_id=0, logical_observables=answer
    )
    return verdict.decide(job, result, None)


def _machine(*pairs, source="error_model_stim"):
    """The test switching machine on Toshio's double window, recording."""
    machine = run_files.switching_machine(rounds_per_shot=30)
    base = (
        ("qpu.source.type", source),
        ("observation.record_window_outcomes", "true"),
        ("switching.strong_window.type", "double_window"),
        ("windows.boundary_policy.type", "Eager"),
        ("windows.terminal_policy", "lookahead"),
        ("workload.arguments.physical_error_probability", "0.01"),
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
    task = collect.Task("truth_verdict", machine)
    return [_measured_shot(task, seed) for seed in SEEDS]


def _measured_shot(task, seed):
    shot = collect.run_shot(task, seed)
    return measure.measure_shot(shot)


def _decided_rows(measurements):
    """Every window a verdict decided, over every shot."""
    return [
        row
        for measurement in measurements
        for row in measurement.window_outcomes
        if row.is_escalated is not None
    ]


def _weak_is_wrong(row) -> bool:
    """The record's reading: an escalated window keeps its weak answer apart."""
    if row.is_escalated:
        return row.weak_answer != row.weak_label
    return row.answer != row.label


@pytest.fixture(scope="module")
def runs():
    """The gap's run and the truth verdict's run, on the same seeds."""
    gap_machine = _machine()
    truth_machine = _machine(("switching.verdict.type", "truth"))
    gap = _measured(gap_machine)
    truth = _measured(truth_machine)
    return gap, truth


def test_a_weak_answer_on_its_label_is_kept_and_one_off_it_escalates():
    fired = (_Error(4, (1,)), _Error(5, (1,)), _Error(3, (1,)))

    on_label = _decided(fired, (1,))
    off_label = _decided(fired, (0,))

    assert on_label is KEEP
    assert off_label is ESCALATE


def test_errors_outside_the_commit_rounds_do_not_change_the_label():
    """Rounds 2 and 6 lie outside the commit rounds 3 to 5."""
    fired = (_Error(2, (1,)), _Error(6, (1,)))

    decided = _decided(fired, (0,))

    assert decided is KEEP


def test_a_verdict_with_no_fired_errors_is_refused():
    settings = verdicts.Truth.Settings()
    verdict = settings.build()
    window = _Window("stream", 0, 2)
    job = _Job(window)
    result = decoding_records.DecodeResult(
        stream_id="stream", window_id=0, logical_observables=(0,)
    )

    with pytest.raises(RuntimeError, match="holds no fired errors"):
        verdict.decide(job, result, None)


def test_the_truth_verdict_escalates_exactly_the_wrong_weak_windows(runs):
    _, truth = runs
    rows = _decided_rows(truth)

    escalated = [bool(row.is_escalated) for row in rows]
    wrong = [_weak_is_wrong(row) for row in rows]

    assert True in escalated
    assert False in escalated
    assert escalated == wrong


def test_the_gap_escalates_right_windows_that_the_truth_verdict_keeps(runs):
    """The two runs differ, so the test above can tell them apart."""
    gap, _ = runs
    rows = _decided_rows(gap)

    switched_right = [
        row for row in rows if row.is_escalated and not _weak_is_wrong(row)
    ]

    assert switched_right


def test_the_truth_verdict_decodes_the_same_shots_as_the_gap(runs):
    gap, truth = runs
    for before, after in zip(gap, truth, strict=True):
        assert after.sample_digest == before.sample_digest


def test_a_truth_verdict_on_a_source_that_keeps_no_errors_is_refused():
    machine = _machine(
        ("switching.verdict.type", "truth"), source="stim_device"
    )
    task = collect.Task("truth_verdict", machine)

    with pytest.raises(
        ValueError, match="switching.verdict reads the shot's fired errors"
    ):
        collect.run_shot(task, 0)
