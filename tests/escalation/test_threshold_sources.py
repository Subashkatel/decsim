"""The threshold sources' laws: fixed compares; online tracks and audits.

FixedThreshold is Toshio et al. 2510.25222 Sec. III A, step 3: keep at
g >= g_th. OnlineThreshold's rate tracker is the adaptive conformal
recursion of Gibbs and Candes, arXiv:2106.00170, Eq. (2) at line 143,
which two tests here run beside the tracker; its audit lane is the
inverse-propensity estimate over a random sample of kept windows, and
its outer loop raises the target on one bad audit and relaxes it on a
rule-of-three clean quota.
"""

import random
import types

import pytest

import decsim.escalation.threshold_sources as threshold_sources
import decsim.records.decoding as decoding_records

SOURCE = object()  # opaque: the sources compare gaps, never sources


def _result(
    gap: float, logical_observables=(0,)
) -> decoding_records.DecodeResult:
    soft_output = decoding_records.SoftOutput(gap=gap, source=SOURCE)
    return decoding_records.DecodeResult(
        1, 4, logical_observables=logical_observables, soft_output=soft_output
    )


def _job(window_id: int = 4) -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1, window_id=window_id, round_count=3
    )


def _controller(
    *,
    target=0.5,
    threshold=2.0,
    step=0.1,
    audit_rate=1.0,
    kept_bad_budget=0.5,
    max_escalation_rate=0.9,
) -> threshold_sources.OnlineThresholdController:
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=threshold, step=step
    )
    audit = threshold_sources.AuditLane(audit_rate=audit_rate)
    adjustment = threshold_sources.TargetAdjustment(
        kept_bad_budget=kept_bad_budget,
        adjust_factor=2.0,
        min_escalation_rate=1e-5,
        max_escalation_rate=max_escalation_rate,
    )
    return threshold_sources.OnlineThresholdController(
        tracker, audit, adjustment
    )


def test_a_fixed_threshold_keeps_at_or_above_and_escalates_below():
    fixed = threshold_sources.FixedThreshold(2.0)
    job = _job()
    at_threshold = _result(2.0)
    above = _result(2.5)
    below = _result(1.999)
    assert fixed.decide_keep(job, at_threshold) is True
    assert fixed.decide_keep(job, above) is True
    assert fixed.decide_keep(job, below) is False
    assert fixed.audits_by_escalating is False


def test_the_rate_tracker_pins_the_target_rate_over_a_random_gap_stream():
    """A property test: 20000 Gaussian gaps against the recursion's target."""
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=0.1, threshold=4.6, step=0.05
    )
    gap_stream = random.Random(7)
    for _ in range(20000):
        gap = gap_stream.gauss(9.0, 3.0)
        tracker.observe(gap)
    realized_rate = tracker.escalation_rate()
    assert realized_rate == pytest.approx(0.1, abs=0.01)
    assert tracker.threshold > 0.0


def test_the_audit_lane_estimate_is_inverse_propensity_weighted():
    """Each audited bad outcome stands for 1/audit_rate kept windows."""
    lane = threshold_sources.AuditLane(audit_rate=0.5)
    for _ in range(100):
        lane.record_kept()
    lane.record_audit(weak_was_bad=True)
    lane.record_audit(weak_was_bad=True)
    lane.record_audit(weak_was_bad=False)
    assert lane.audited_count == 3
    assert lane.audited_bad_count == 2
    assert lane.kept_bad_rate(total_window_count=200) == (2 / 0.5) / 200


def test_one_bad_audit_raises_the_target_and_a_clean_quota_relaxes_it():
    """Raising is immediate; a relax needs ceil(3 / 0.5) = 6 clean audits."""
    controller = _controller(target=0.1, kept_bad_budget=0.5)
    assert controller.relax_audit_quota() == 6
    controller.record_audit_outcome(weak_was_bad=True)
    assert controller.tracker.target_escalation_rate == pytest.approx(0.2)
    assert controller.raise_count == 1
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    assert controller.relax_count == 0
    controller.record_audit_outcome(weak_was_bad=False)
    assert controller.relax_count == 1
    assert controller.tracker.target_escalation_rate == pytest.approx(0.1)


def test_the_raised_target_respects_the_backlog_cap():
    """The Theorem 1 duty cap bounds the target whatever the audits say."""
    controller = _controller(
        target=0.6, kept_bad_budget=0.5, max_escalation_rate=0.9
    )
    controller.record_audit_outcome(weak_was_bad=True)
    assert controller.tracker.target_escalation_rate == pytest.approx(0.9)


def test_an_online_threshold_audits_kept_windows_and_learns_from_the_strong():
    """An audited window escalates; the strong result labels it.

    The label is whether the strong observables revised the kept weak
    ones.
    """
    controller = _controller(target=0.0, threshold=0.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    assert online.audits_by_escalating is True
    # audit_rate 1.0: every kept window is audited, so it escalates
    fourth = _job(4)
    confident = _result(5.0, (1, 0))
    kept = online.decide_keep(fourth, confident)
    assert kept is False
    audited = online.summary()
    assert audited["pending_audits"] == 1
    clean_strong = decoding_records.DecodeResult(
        1, 4, logical_observables=(1, 0)
    )
    online.learn_from_strong_result((1, 4), clean_strong)
    assert controller.raise_count == 0
    labeled = online.summary()
    assert labeled["pending_audits"] == 0
    fifth = _job(5)
    online.decide_keep(fifth, confident)
    revised_strong = decoding_records.DecodeResult(
        1, 5, logical_observables=(0, 0)
    )
    online.learn_from_strong_result((1, 5), revised_strong)
    assert controller.raise_count == 1
    # a strong result that answers no audit (an ordinary escalation) is
    # ignored rather than mislabeled
    online.learn_from_strong_result((1, 6), clean_strong)
    assert controller.audit.audited_count == 2


def test_an_online_threshold_records_its_trajectory_for_the_front():
    controller = _controller(target=0.0, threshold=10.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    assert online.trajectory == [(0, 10.0, "start")]
    fourth = _job(4)
    audited = _result(15.0)
    online.decide_keep(fourth, audited)
    assert online.trajectory[-1] == (1, 10.0, "audit")


def test_an_audit_needs_the_weak_observables_for_its_label():
    controller = _controller(target=0.0, threshold=0.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    soft_output = decoding_records.SoftOutput(gap=5.0, source=SOURCE)
    timing_only = types.SimpleNamespace(
        soft_output=soft_output, logical_observables=None
    )
    fourth = _job(4)
    with pytest.raises(ValueError, match="timing-only"):
        online.decide_keep(fourth, timing_only)


def test_the_rate_tracker_follows_the_papers_recursion_step_for_step():
    """The referent run beside the row.

    Gibbs and Candes arXiv:2106.00170 Eq. (2), line 143:
    alpha_{t+1} = alpha_t + gamma (alpha - err_t), with err_t the
    indicator of the event whose rate is being pinned. Written out here
    over the same gap stream, it reproduces the tracker's threshold at
    every one of five hundred steps.
    """
    target = 0.1
    step = 0.05
    start = 4.6
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=start, step=step
    )
    gap_stream = random.Random(11)
    reference = start
    tracked = []
    expected = []
    for _ in range(500):
        gap = gap_stream.gauss(9.0, 3.0)
        tracker.observe(gap)
        tracked.append(tracker.threshold)
        escalated = gap < reference
        error = float(escalated)
        reference = reference + step * (target - error)
        expected.append(reference)
    assert tracked == pytest.approx(expected)


def test_the_realized_rate_is_pinned_by_the_distance_the_threshold_moved():
    """Proposition 4.1's identity (2106.00170 lines 309-317).

    Summing the recursion gives
    (1/T) sum err_t - alpha = (alpha_1 - alpha_{T+1}) / (T gamma), so the
    realized escalation rate can only stray from the target as far as the
    threshold itself travelled, whatever the gaps did. That is what makes
    the loop self-correcting under drift, and it holds exactly here since
    the threshold never reached the floor.
    """
    target = 0.1
    step = 0.05
    start = 4.6
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=start, step=step
    )
    gap_stream = random.Random(23)
    window_count = 2000
    for _ in range(window_count):
        gap = gap_stream.gauss(9.0, 3.0)
        tracker.observe(gap)
    assert tracker.threshold > 0.0
    travelled = start - tracker.threshold
    drift = travelled / (window_count * step)
    realized = tracker.escalation_rate()
    assert realized - target == pytest.approx(drift)
