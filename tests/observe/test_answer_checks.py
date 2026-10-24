"""Each window's two answers, checked against the truth after it.

The referents are each decoder alone on the same seed: a chain that
keeps every weak answer must end as the weak machine does, and one that
keeps every strong answer as the strong machine does, since each
window's check and the chain's last are the closing law the checkpoints
already hold to separate streams (tests/observe/test_checkpoints.py).
The best-of-two chain is held to its rule on scripted answers: it keeps
the strong answer only where the weak one fails its check and the
strong one passes, so it fails a window only where both answers do.
"""

import dataclasses

import pytest

import decsim.confidence.cluster as cluster
import decsim.escalation.verdicts as verdicts
import decsim.experiments.stream_command as stream_command
import decsim.machine as machine_module
import decsim.observe.answer_checks as answer_checks
import decsim.observe.closing_decisions as closing_decisions
import decsim.observe.settings as observe_settings
import decsim.presets.machines as machines
import decsim.records.checkpoints as checkpoint_records
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import tests.observe.test_checkpoints as checkpoint_tests
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

PATHS = observe_settings.ANSWER_CHECK_PATHS
ROUNDS = checkpoint_tests.SWITCHING_ROUNDS
# both alone pass (0) or fail (3); only one of them fails (11, 20, 33)
SEEDS = (0, 3, 11, 20, 33)
# the scripted stream: d = 3, commit 3, buffer 3, read out after round 12
STREAM = 100
FINAL_ROUND = 12
AFTER_FINAL_ROUND = FINAL_ROUND + 1
SCRIPTED_WINDOWS = (
    window_records.Window(STREAM, 0, 1, 3, 6, 6),
    window_records.Window(STREAM, 1, 4, 6, 9, 6),
    window_records.Window(STREAM, 2, 7, 9, 12, 6),
    window_records.Window(STREAM, 3, 10, 12, 12, 3),
)
# each window's weak and strong flip, its weak gap and verdict: window 1's
# weak answer fails and its strong one passes, window 3's the reverse
SCRIPT = {
    0: ((0,), (0,), 9.0, False),
    1: ((1,), (0,), 1.5, True),
    2: ((0,), (0,), 7.0, False),
    3: ((0,), (1,), 2.5, True),
}


def _checked(task, paths=PATHS):
    machine = task.machine
    observation = dataclasses.replace(
        machine.observation, answer_check_paths=paths
    )
    machine = dataclasses.replace(machine, observation=observation)
    return dataclasses.replace(task, machine=machine)


def _weak_machine():
    machine = machines.weak_baseline(3, checkpoint_tests.PROBABILITY, 1.0)
    return machine.with_settings(
        machines.WEAK_POOL, algorithm=checkpoint_tests.UNION_FIND
    )


def _under_threshold(row: dict) -> bool:
    """Keep at a gap at or above the threshold (escalation/verdicts.py)."""
    gap = row["weak_gap"]
    if gap is None:
        return True
    return gap < checkpoint_tests.FIXED.threshold_nats


def _strong_machine(weak):
    """Belief matching on every window of the weak machine's windows."""
    algorithm = checkpoint_tests.BELIEF_MATCHING
    pool = dataclasses.replace(
        machines.STRONG_BASELINE_POOL, algorithm=algorithm
    )
    return machines.strong_alone(weak, pool)


def _kept(row: dict) -> list:
    windows = row["windows"][0]
    return [window["kept"] for window in windows]


def _outcomes(chains: list, path: str) -> list:
    return [chain[path]["failed"][0] for chain in chains]


def _alone_failed(machine, seed: int) -> bool:
    task = checkpoint_tests._checkpointed(machine, 3, None)
    record = stream_command.stream_record(task, ROUNDS, 0, seed)
    return record["measurement"]["logical_failure"]


def _chains(seed: int) -> dict:
    switching = checkpoint_tests._switching_task(checkpoint_tests.FIXED, None)
    task = _checked(switching)
    record = stream_command.stream_record(task, ROUNDS, 0, seed)
    rows = record["measurement"]["answer_checks"]
    return {row["path"]: row for row in rows}


def test_the_weak_and_strong_chains_end_as_each_decoder_alone_seed_by_seed():
    weak = _weak_machine()
    strong = _strong_machine(weak)

    chains = [_chains(seed) for seed in SEEDS]
    weak_alone = [_alone_failed(weak, seed) for seed in SEEDS]
    strong_alone = [_alone_failed(strong, seed) for seed in SEEDS]

    assert _outcomes(chains, "weak") == weak_alone
    assert _outcomes(chains, "strong") == strong_alone
    assert weak_alone != strong_alone


def test_on_these_seeds_the_best_chain_fails_where_both_alone_fail():
    """A window choice here helps no shot both decoders alone fail."""
    chains = [_chains(seed) for seed in SEEDS]

    weak = _outcomes(chains, "weak")
    strong = _outcomes(chains, "strong")
    both = [w and s for w, s in zip(weak, strong, strict=True)]
    assert _outcomes(chains, "best_of_two") == both


def test_a_real_chain_keeps_a_passing_answer_wherever_one_passes_property():
    for seed in SEEDS:
        chains = _chains(seed)

        rows = chains["best_of_two"]["windows"][0]

        for row in rows:
            either_passes = not row["weak_failed"] or not row["strong_failed"]
            kept_failed = row[f"{row['kept']}_failed"]
            assert kept_failed is not either_passes
        escalates = [row["would_escalate"] for row in rows]
        assert escalates == [_under_threshold(row) for row in rows]


def test_each_path_keeps_its_answer_and_the_best_keeps_the_one_that_passes():
    listener = _scripted_listener()

    rows = {row["path"]: row for row in listener.rows()}

    kept = {path: _kept(rows[path]) for path in PATHS}
    assert kept == {
        "weak": ["weak"] * 4,
        "strong": ["strong"] * 4,
        "best_of_two": ["weak", "strong", "weak", "weak"],
    }
    # the weak chain carries window 1's flip; the strong one makes window 3's
    assert rows["weak"]["failed"] == [True]
    assert rows["strong"]["failed"] == [True]
    assert rows["best_of_two"]["failed"] == [False]


def test_a_row_holds_the_check_round_the_weak_gap_and_the_verdict():
    listener = _scripted_listener()

    rows = {row["path"]: row for row in listener.rows()}

    best = rows["best_of_two"]["windows"][0]
    assert [row["check_round"] for row in best] == [7, 10, 12, 12]
    assert [row["weak_gap"] for row in best] == [9.0, 1.5, 7.0, 2.5]
    assert [row["would_escalate"] for row in best] == [False, True, False, True]
    assert [(row["weak_failed"], row["strong_failed"]) for row in best] == [
        (False, False),
        (True, False),
        (False, False),
        (False, True),
    ]


def test_a_row_says_whether_the_two_answers_share_a_logical_class():
    listener = _scripted_listener()

    rows = {row["path"]: row for row in listener.rows()}

    best = rows["best_of_two"]["windows"][0]
    classes = [row["same_logical_class"] for row in best]
    assert classes == [True, False, True, False]


def test_a_stream_lets_go_of_the_rounds_and_copies_its_checks_passed():
    listener = _scripted_listener()

    stream = listener.streams[STREAM]

    # the last window commits round 12; no later window reads before 13
    assert stream.windows == []
    assert list(stream.snapshots) == [FINAL_ROUND]
    reach = stream.closing.reach
    assert min(stream.packets) == FINAL_ROUND + 1 - reach


def test_answer_checks_on_a_machine_with_no_strong_tier_are_refused():
    machine = _weak_machine()
    weak = checkpoint_tests._checkpointed(machine, 3, None)
    task = _checked(weak)

    with pytest.raises(ValueError) as refusal:
        stream_command.stream_record(task, ROUNDS, 0, 1)

    assert "needs an escalate edge to a pool that decodes" in str(refusal.value)


def test_answer_checks_beside_a_verdict_that_draws_are_refused():
    bernoulli = verdicts.Bernoulli.Settings(rate=0.1)
    switching = checkpoint_tests._switching_task(
        checkpoint_tests.FIXED, None, verdict=bernoulli
    )
    task = _checked(switching)

    with pytest.raises(NotImplementedError) as refusal:
        stream_command.stream_record(task, ROUNDS, 0, 1)

    message = str(refusal.value)
    assert "observation.answer_check_paths decides" in message
    assert "Bernoulli draws from the run's seed" in message


def test_answer_checks_on_a_source_that_copies_nothing_are_refused():
    switching = checkpoint_tests._switching_task(checkpoint_tests.FIXED, None)
    task = _checked(switching)

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(task.machine, 1)

    assert "answer_check_paths needs a source that copies" in str(refusal.value)


@pytest.mark.parametrize(
    "paths", [("weak", "weak"), ("oracle",), ["weak"]], ids=str
)
def test_answer_check_paths_other_than_distinct_known_ones_are_refused(paths):
    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings(answer_check_paths=paths)

    assert "answer_check_paths must be a tuple of distinct paths" in str(
        refusal.value
    )


def _scripted_listener() -> answer_checks.AnswerChecks:
    """A noiseless d = 3 stream of 12 rounds, its answers scripted."""
    closer = _ScriptedCloser()
    listener = answer_checks.AnswerChecks(PATHS, closer)
    windows = [dataclasses.replace(window) for window in SCRIPTED_WINDOWS]
    for window in windows:
        listener.window_planned(window)
    program = memory_programs.memory_program(3, "Z", 0.001)
    rounds = range(1, AFTER_FINAL_ROUND)
    for round_index in rounds:
        is_final = round_index == FINAL_ROUND
        snapshot = checkpoint_records.CheckpointSnapshot(
            STREAM, round_index, (0, 0, 0, 0), (0,), is_final, program
        )
        listener.checkpoint_reached(snapshot)
    checkpoint_tests._emit_noiseless_rounds(listener, FINAL_ROUND)
    for window in windows:
        listener.window_final(window)
    return listener


class _ScriptedCloser:
    """Answers each window from SCRIPT; a tail window passes."""

    def __init__(self) -> None:
        decoder = minimum_weight_perfect_matching.PyMatchingDecoder()
        self.requirement = decoder.fault_model_requirement

    def answer_pair(self, job) -> closing_decisions.AnswerPair:
        weak_flip, strong_flip, gap, escalates = SCRIPT[job.window_id]
        weak = _result(job, weak_flip)
        source = cluster.CLUSTER_GAP_SOURCE
        weak.soft_output = decoding_records.SoftOutput(gap=gap, source=source)
        strong = _result(job, strong_flip)
        return closing_decisions.AnswerPair(weak, strong, escalates)

    def decide(self, job) -> decoding_records.DecodeResult:
        return _result(job, (0,))


def _result(job, flip: tuple) -> decoding_records.DecodeResult:
    return decoding_records.DecodeResult(
        job.stream_id, job.window_id, logical_observables=flip
    )
