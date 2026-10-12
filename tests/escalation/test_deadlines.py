"""An edge's deadline on declared ticks: each rescue timeline by hand.

DART-Q (Bhave, Choudhury and Basu 2605.09142) gives a decode job the
deadline d = a + Delta and treats a later output as unavailable; a
trigger injects a rescue. On the declared switching machine
(tests/escalation/declared_fabric.py) W0's weak job arrives at 15 us,
decodes 20-30 us and meets the verdict at 30 us; W1 and W2 arrive at
18 us and wait for their boundaries. A rescue is the strong decode
started before the verdict: W0's rounds ride
weak_to_strong (3 us), cross to the strong unit (6 us)
and decode for 30 us; the answer rides strong_to_frame (4 us)
and commits 1 us later. W1's boundary leaves with the strong answer,
crosses strong_to_weak (4 us) and W1's input (5 us).
"""

import pytest

import decsim.config as config
import decsim.escalation.deadlines as deadlines
import decsim.machine as machine_module
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric


def test_the_slack_trigger_starts_the_rescue_its_threshold_early():
    trigger = deadlines.SlackTrigger.Settings(threshold_microseconds=4.0)
    slack_ticks = _ticks(10.0)

    delay = trigger.rescue_delay(slack_ticks, 0)

    assert delay == _ticks(6.0)


def test_a_slack_threshold_past_the_budget_rescues_at_arrival():
    trigger = deadlines.SlackTrigger.Settings(threshold_microseconds=12.0)
    slack_ticks = _ticks(10.0)

    delay = trigger.rescue_delay(slack_ticks, 0)

    assert delay == 0


def test_the_backlog_trigger_rescues_only_a_job_that_finds_a_queue():
    trigger = deadlines.BacklogTrigger.Settings(queued_jobs=2)
    slack_ticks = _ticks(10.0)

    short = trigger.rescue_delay(slack_ticks, 1)
    long = trigger.rescue_delay(slack_ticks, 2)

    assert short is None
    assert long == 0


def test_a_deadline_needs_a_positive_slack_budget():
    """DART-Q requires Delta > 0 (Sec. III-A, Eq. 1)."""
    with pytest.raises(ValueError, match="slack_microseconds"):
        deadlines.Deadline.Settings(slack_microseconds=0.0)


def test_a_deadline_refuses_a_late_result_it_does_not_know():
    with pytest.raises(ValueError, match="late_source_result"):
        deadlines.Deadline.Settings(
            slack_microseconds=10.0, late_source_result="dropped"
        )


def test_a_backlog_trigger_needs_a_positive_count():
    with pytest.raises(ValueError, match="queued_jobs"):
        deadlines.BacklogTrigger.Settings(queued_jobs=0)


def test_a_source_on_time_starts_no_rescue_and_runs_the_serial_timeline():
    """W0's deadline is 115 us; its verdict at 30 us meets it.

    The serial escalation follows: the strong decode starts at 39 us,
    the answer lands at 73 us and commits at 74 us
    (test_switching_mode.py).
    """
    deadline = deadlines.Deadline.Settings(slack_microseconds=100.0)
    machine = _deadline_run({0}, deadline)

    first = _first_record(machine)
    strong_start = _log_tick(machine, "START DECODE strong(mem1 W0)")
    counts = _deadline_counts(machine)

    assert strong_start == _ticks(39.0)
    assert first.tier == "strong"
    assert first.accepted_ticks == _ticks(73.0)
    assert first.committed_ticks == _ticks(74.0)
    assert counts == {"rescues": 0, "misses": 0}


def test_a_late_source_is_answered_by_the_rescue_started_at_its_deadline():
    """W0's deadline is 25 us; its result at 30 us is unavailable.

    The rescue starts at 25 us: the rounds land at 28 us, the strong
    decode runs 34-64 us, and the weak result, though confident, is
    late, so the verdict escalates and selects the rescue. The answer
    lands at 68 us and commits at 69 us; W1 starts at 73 us. Every
    window misses its deadline: W1 and W2 wait for their boundaries.
    """
    deadline = deadlines.Deadline.Settings(slack_microseconds=10.0)
    machine = _deadline_run(set(), deadline)

    first = _first_record(machine)
    rescue = _log_tick(machine, "mem1 W0 [commit 1-3]: deadline trigger")
    strong_start = _log_tick(machine, "START DECODE strong(mem1 W0)")
    released_start = _log_tick(machine, "START DECODE mem1 W1")
    counts = _deadline_counts(machine)

    assert rescue == _ticks(25.0)
    assert strong_start == _ticks(34.0)
    assert first.tier == "strong"
    assert first.accepted_ticks == _ticks(68.0)
    assert first.committed_ticks == _ticks(69.0)
    assert released_start == _ticks(73.0)
    assert counts == {"rescues": 3, "misses": 3}


def test_a_slack_threshold_starts_the_rescue_before_the_deadline():
    """A 4 us threshold starts W0's rescue at 21 us, not 25 us.

    The rounds land at 24 us, the strong decode runs 30-60 us, the
    answer lands at 64 us and commits at 65 us; W1 starts at 69 us.
    """
    trigger = deadlines.SlackTrigger.Settings(threshold_microseconds=4.0)
    deadline = deadlines.Deadline.Settings(
        slack_microseconds=10.0, trigger=trigger
    )
    machine = _deadline_run(set(), deadline)

    first = _first_record(machine)
    rescue = _log_tick(machine, "mem1 W0 [commit 1-3]: deadline trigger")
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert rescue == _ticks(21.0)
    assert first.accepted_ticks == _ticks(64.0)
    assert first.committed_ticks == _ticks(65.0)
    assert released_start == _ticks(69.0)


def test_a_late_source_taken_as_a_revision_keeps_its_answer():
    """late_source_result revision: the late weak result meets the verdict.

    W0 misses its 25 us deadline, but the verdict keeps its confident
    result, so the frame takes it at 32 us and the rescue is cancelled.
    """
    deadline = deadlines.Deadline.Settings(
        slack_microseconds=10.0, late_source_result="revision"
    )
    machine = _deadline_run(set(), deadline)

    first = _first_record(machine)
    tiers = _tiers(machine)
    counts = _deadline_counts(machine)

    assert first.accepted_ticks == _ticks(32.0)
    assert first.committed_ticks == _ticks(33.0)
    assert tiers == {"weak"}
    assert counts == {"rescues": 3, "misses": 3}


def test_the_backlog_trigger_rescues_the_job_that_finds_one_waiting():
    """W2 arrives at 18 us behind W1, so only W2 is rescued, at arrival.

    Every result meets its 100 us deadline and is kept.
    """
    trigger = deadlines.BacklogTrigger.Settings(queued_jobs=1)
    deadline = deadlines.Deadline.Settings(
        slack_microseconds=100.0, trigger=trigger
    )
    machine = _deadline_run(set(), deadline)

    rescue = _log_tick(machine, "deadline trigger")
    rescued = declared_run.log_lines_containing(machine, "deadline trigger")
    tiers = _tiers(machine)
    counts = _deadline_counts(machine)

    assert rescue == _ticks(18.0)
    assert "mem1 W2" in rescued[0]
    assert tiers == {"weak"}
    assert counts == {"rescues": 1, "misses": 0}


def _deadline_run(doubts: set, deadline) -> machine_module.Machine:
    """The declared switching machine with a deadline on its edge, run."""
    switching = declared_run.declared_switching(deadline=deadline)
    machine = fabric.switching_machine(
        rounds=9, escalated_windows=doubts, switching=switching
    )
    machine.run()
    return machine


def _deadline_counts(machine: machine_module.Machine) -> dict:
    """The edge deadline's statistics, by name."""
    statistics = machine.observation.statistics
    counts = {}
    for row in statistics.rows():
        if row.path == "switching.deadline":
            counts[row.statistic] = row.value
    return counts


def _tiers(machine: machine_module.Machine) -> set:
    snapshot = machine.control.pauli_frame.snapshot()
    return {record.tier for record in snapshot.records}


def _first_record(machine: machine_module.Machine):
    snapshot = machine.control.pauli_frame.snapshot()
    return snapshot.records[0]


def _log_tick(machine: machine_module.Machine, text: str) -> int:
    log_lines = machine.observation.log.lines
    return declared_run.log_tick(log_lines, text)


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)
