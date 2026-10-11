"""The verdict rows an escalate edge decides by (escalation/verdicts.py).

The referents: the gap row is today's switching rule (Toshio et al.
2510.25222 Sec. III A), so a result at the threshold is kept and one
below it, or with no soft output, escalates; the status row reads the
backend's own report; always and never decide alone. On the declared
switching machine, a never verdict leaves every window on the weak
tier and an always verdict sends every window to the strong one.
"""

import decsim.escalation.verdicts as verdicts
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
        operation_id=1, window_id=0, logical_observables=(0,), **fields
    )


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
    uncorrected = decoding_records.DecodeResult(operation_id=1, window_id=0)

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
    assert fabric.frame_tiers(machine) == WEAK_TIERS
    assert counts.needed == 0


def test_an_always_verdict_sends_every_window_to_the_strong_tier():
    """The declared decoder is confident everywhere; the verdict escalates."""
    always = verdicts.Always.Settings()

    machine = _switching_run(always, set())

    counts = machine.decoders.decoder_manager.strong_requests.counts
    assert fabric.frame_tiers(machine) == STRONG_TIERS
    assert counts.needed == 3
