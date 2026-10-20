"""The verdict rows an escalate edge decides by (escalation/verdicts.py).

The referents: the gap row is today's switching rule (Toshio et al.
2510.25222 Sec. III A), so a result at the threshold is kept and one
below it, or with no soft output, escalates; the status row reads the
backend's own report; always and never decide alone. On the declared
switching machine, a never verdict leaves every window on the weak
tier and an always verdict sends every window to the strong one. The
bernoulli row is Toshio et al.'s i.i.d. switching (Sec. III C): its
draws come from the run's seed, so a seed repeats them. The
backlog_bernoulli row reads the weak queue the build binds it to, and
steps its share down once that queue grows.
"""

import decsim.escalation.target_steps as target_steps
import decsim.escalation.verdicts as verdicts
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric

KEEP = decoding_records.Verdict.KEEP
ESCALATE = decoding_records.Verdict.ESCALATE
STATUS = decoding_records.BackendDecodeStatus
ALL_WINDOWS = {0, 1, 2}
WEAK_TIERS = [((1, 0), "weak"), ((1, 1), "weak"), ((1, 2), "weak")]
STRONG_TIERS = [((1, 0), "strong"), ((1, 1), "strong"), ((1, 2), "strong")]
# a soft output the gap row only asks is there
SOFT_OUTPUT = object()


class _Threshold:
    """A threshold source whose answer is given."""

    def __init__(self, keeps: bool) -> None:
        self.keeps = keeps

    def decide_keep(self, job, result) -> bool:
        """The given answer."""
        del job, result
        return self.keeps


def _result(**fields) -> decoding_records.DecodeResult:
    return decoding_records.DecodeResult(
        stream_id=1, window_id=0, logical_observables=(0,), **fields
    )


def _bernoulli_tiers(seed: int) -> list:
    """The frame tiers of a 90-window run escalating half its windows."""
    bernoulli = verdicts.Bernoulli.Settings(rate=0.5)
    switching = declared_run.declared_switching(verdict=bernoulli)
    graph = fabric.switching_graph(
        rounds=270, escalated_windows=set(), switching=switching
    )
    machine = machine_module.Machine.build(graph, seed)
    machine.run()
    return declared_run.frame_tiers(machine)


def _switching_run(verdict, escalated_windows):
    switching = declared_run.declared_switching(verdict=verdict)
    machine = fabric.switching_machine(
        rounds=9, escalated_windows=escalated_windows, switching=switching
    )
    machine.run()
    return machine


def test_the_gap_row_keeps_what_its_threshold_keeps():
    verdict = verdicts.GapThreshold()
    confident = _result(soft_output=SOFT_OUTPUT)
    keeping = _Threshold(keeps=True)
    escalating = _Threshold(keeps=False)

    kept = verdict.decide(None, confident, keeping)
    escalated = verdict.decide(None, confident, escalating)

    assert kept is KEEP
    assert escalated is ESCALATE


def test_the_gap_row_escalates_a_result_with_no_soft_output():
    verdict = verdicts.GapThreshold()
    bare = _result()
    threshold = _Threshold(keeps=True)

    decided = verdict.decide(None, bare, threshold)

    assert decided is ESCALATE


def test_the_status_row_keeps_a_succeeded_result_and_escalates_the_rest():
    verdict = verdicts.Status()
    threshold = _Threshold(keeps=True)
    succeeded = _result(decode_status=STATUS.SUCCEEDED)
    unstated = _result()
    nonconverged = _result(decode_status=STATUS.NONCONVERGED)
    invalid = _result(decode_status=STATUS.INVALID_CORRECTION)
    uncorrected = decoding_records.DecodeResult(stream_id=1, window_id=0)

    decided = [
        verdict.decide(None, result, threshold)
        for result in (succeeded, unstated, nonconverged, invalid, uncorrected)
    ]

    assert decided == [KEEP, KEEP, ESCALATE, ESCALATE, ESCALATE]


def test_always_escalates_and_never_keeps():
    result = _result(soft_output=SOFT_OUTPUT)
    threshold = _Threshold(keeps=True)
    always = verdicts.Always()
    never = verdicts.Never()

    escalated = always.decide(None, result, threshold)
    kept = never.decide(None, result, threshold)

    assert escalated is ESCALATE
    assert kept is KEEP


def test_a_never_verdict_keeps_every_window_on_the_weak_tier():
    """The declared decoder asks for every window; the verdict keeps it."""
    never = verdicts.Never.Settings()

    machine = _switching_run(never, ALL_WINDOWS)

    counts = machine.decoders.decoder_manager.strong_requests.counts
    assert declared_run.frame_tiers(machine) == WEAK_TIERS
    assert counts.needed == 0


def test_an_always_verdict_sends_every_window_to_the_strong_tier():
    """The declared decoder is confident everywhere; the verdict escalates."""
    always = verdicts.Always.Settings()

    machine = _switching_run(always, set())

    counts = machine.decoders.decoder_manager.strong_requests.counts
    assert declared_run.frame_tiers(machine) == STRONG_TIERS
    assert counts.needed == 3


def test_bernoulli_at_rate_zero_keeps_and_at_rate_one_escalates():
    never = verdicts.Bernoulli(rate=0.0)
    always = verdicts.Bernoulli(rate=1.0)
    result = _result()

    kept = never.decide(None, result, None)
    escalated = always.decide(None, result, None)

    assert kept is KEEP
    assert escalated is ESCALATE


def test_a_bernoulli_run_repeats_its_seed_and_not_another():
    """Seed 0 twice escalates the same windows; seed 1 others."""
    first = _bernoulli_tiers(0)
    again = _bernoulli_tiers(0)
    other = _bernoulli_tiers(1)

    strong = []
    for _window_key, tier in first:
        is_strong = tier == "strong"
        strong.append(is_strong)
    assert first == again
    assert first != other
    # 90 windows at rate 0.5: 45 escalate on average, sd 4.7
    assert 30 < sum(strong) < 60


def test_a_bernoulli_takes_its_later_rate_from_its_window_on():
    bernoulli = verdicts.Bernoulli.Settings(
        rate=0.0, later_rate=1.0, from_window=45
    )
    switching = declared_run.declared_switching(verdict=bernoulli)
    graph = fabric.switching_graph(
        rounds=270, escalated_windows=set(), switching=switching
    )
    machine = machine_module.Machine.build(graph, 0)

    machine.run()

    tiers = [tier for _window_key, tier in declared_run.frame_tiers(machine)]
    assert set(tiers[:45]) == {"weak"}
    assert set(tiers[45:]) == {"strong"}


def test_a_backlog_bernoulli_steps_down_once_its_queue_backs_up():
    """All escalated, the weak queue grows and the share drops to zero.

    The second check, at the twentieth window and before its draw, sees
    the mean waiting rise over the first, so nineteen windows escalate.
    """
    steps = target_steps.TargetSteps.Settings(
        start=1.0,
        levels=(0.0, 1.0),
        check_every_windows=10,
        rise_checks=1,
        rise_margin_windows=0.0,
        calm_checks=100,
    )
    verdict = verdicts.BacklogBernoulli.Settings(steps=steps)
    switching = declared_run.declared_switching(verdict=verdict)
    graph = fabric.switching_graph(
        rounds=270, escalated_windows=set(), switching=switching
    )
    machine = machine_module.Machine.build(graph, 0)

    machine.run()

    tiers = [tier for _window_key, tier in declared_run.frame_tiers(machine)]
    assert tiers[:19] == ["strong"] * 19
    assert set(tiers[19:]) == {"weak"}
