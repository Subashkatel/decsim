"""The threshold sources: where the switching policy's threshold comes from.

FixedThreshold keeps a weak result whose gap is at or above one value,
the paper's constant g_th (Toshio et al. 2510.25222 Sec. III A, step
3). OnlineThreshold starts there and adapts it across a sweep point's
shots with two loops: a rate tracker that pins the escalation fraction
at a target, and an audit lane that strong-decodes a random sample of
kept windows to learn whether the target is safe; it is one instance
per sweep point, shared by every shot. The yaml's third source, table,
is resolved by the front to a fixed threshold per sweep point
(decoders/settings.py, threshold_nats_for), so at run time it is
FixedThreshold. Both rows fill the ThresholdSource port
(decsim/ports.py). Thresholds and gaps are natural-log weight (nats),
the unit the decoder compares in; the yaml converts the paper's
decibels.
"""

import dataclasses
import math

import decsim.records.decoding as decoding_records


class FixedThreshold:
    """The paper's constant g_th: keep at gap >= threshold, escalate below."""

    audits_by_escalating = False

    def __init__(self, threshold_nats: float) -> None:
        self.threshold_nats = threshold_nats

    def decide_keep(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> bool:
        """The gap against the threshold, both in nats."""
        del job
        return result.soft_output.gap >= self.threshold_nats

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """A fixed threshold learns nothing."""
        del window_key
        del result


class EscalationRateTracker:
    """Inner calibration loop: pin the escalation fraction at a target.

    Label-free: every window reveals whether it escalated
    (gap < threshold), so the escalation fraction is fully observable.
    The update is the adaptive conformal recursion (Gibbs and Candes,
    arXiv:2106.00170): raise the threshold a little on every kept
    window, lower it a lot on every escalation, balancing at the target
    quantile of the live gap distribution with a per-sequence long-run
    guarantee under any drift. Hardware precedent: O-GEHL's update
    threshold is servoed by the same event-rate-balancing counter
    (Seznec, CBP-1 2004).
    """

    def __init__(
        self, target_escalation_rate: float, threshold: float, step: float
    ) -> None:
        self.target_escalation_rate = target_escalation_rate
        self.threshold = threshold
        self.step = step
        self.window_count = 0
        self.escalated_count = 0

    def observe(self, gap: float) -> bool:
        """Consume one window's gap; True when it escalates."""
        escalated = gap < self.threshold
        self.window_count += 1
        self.escalated_count += int(escalated)
        move = self.target_escalation_rate - float(escalated)
        self.threshold += self.step * move
        if self.threshold < 0.0:
            self.threshold = 0.0
        return escalated

    def escalation_rate(self) -> float:
        """The escalated fraction of the windows seen so far."""
        if self.window_count == 0:
            return 0.0
        return self.escalated_count / self.window_count


class AuditLane:
    """Randomized strong-decoder audits of kept windows.

    The lane is cache set-dueling's move (Qureshi et al., ISCA 2007):
    dedicate a small fixed sample to the expensive path so ground truth
    keeps flowing whatever threshold is live. Sampling kept windows is
    what breaks the selective-labels bias (Lakkaraju et al., KDD 2017):
    without it every label comes from below the threshold and the kept
    region is pure extrapolation.

    kept_bad_rate estimates P(weak revised and kept) per window, the
    quantity the paper's Eq. 4 bounds with epsilon * PL_strong. Each
    audited bad outcome counts 1/audit_rate kept windows (inverse
    propensity). The label is disagreement with the strong result, the
    reference the paper's protocol also measures against.
    """

    def __init__(self, audit_rate: float) -> None:
        self.audit_rate = audit_rate
        self.audited_count = 0
        self.audited_bad_count = 0
        self.weighted_bad_sum = 0.0
        self.kept_count = 0

    def should_audit(self, unit_random: float) -> bool:
        """Whether this kept window is audited, from one uniform draw."""
        return unit_random < self.audit_rate

    def record_kept(self) -> None:
        """One more window was kept."""
        self.kept_count += 1

    def record_audit(self, weak_was_bad: bool) -> None:
        """One audit's label; a bad one weighs 1/audit_rate windows."""
        self.audited_count += 1
        if weak_was_bad:
            self.audited_bad_count += 1
            self.weighted_bad_sum += 1.0 / self.audit_rate

    def kept_bad_rate(self, total_window_count: int) -> float:
        """The inverse-propensity estimate of P(weak revised and kept)."""
        if total_window_count == 0:
            return 0.0
        return self.weighted_bad_sum / total_window_count


@dataclasses.dataclass(frozen=True)
class TargetAdjustment:
    """How the outer loop moves the escalation target on an audit label.

    kept_bad_budget is the bad rate the target may not exceed; one bad
    audit multiplies the target by adjust_factor, ceil(3 / kept_bad_
    budget) clean audits in a row divide it back; the target stays
    inside [min_escalation_rate, max_escalation_rate].
    """

    kept_bad_budget: float
    adjust_factor: float
    min_escalation_rate: float
    max_escalation_rate: float


class OnlineThresholdController:
    """Both calibration loops together: duty tracking steered by audits.

    The two directions of the outer loop have asymmetric evidence costs
    and are handled asymmetrically. Raise: each audited bad outcome
    carries importance weight one over the audit rate, so one event is
    already strong evidence the budget is blown; the escalation target
    is multiplied by adjust_factor immediately. Relax: certifying the
    bad rate is below budget needs the rule-of-three quota, about
    3 / kept_bad_budget clean audits for a 95% upper bound at the
    budget; only a full clean quota shrinks the target.

    The target always stays inside [min_escalation_rate,
    max_escalation_rate]; the max is the Theorem 1 backlog bound: the
    strong tier's duty cycle may never exceed what its latency can
    absorb, whatever accuracy would prefer.
    """

    def __init__(
        self,
        tracker: EscalationRateTracker,
        audit: AuditLane,
        adjustment: TargetAdjustment,
    ) -> None:
        self.tracker = tracker
        self.audit = audit
        self.adjustment = adjustment
        self.clean_audit_streak = 0
        self.raise_count = 0
        self.relax_count = 0

    def relax_audit_quota(self) -> int:
        """Clean audits needed before a relax is statistically earned."""
        quota = 3.0 / self.adjustment.kept_bad_budget
        rounded_up = math.ceil(quota)
        return int(rounded_up)

    def observe(self, gap: float, unit_random: float) -> tuple:
        """One window: (escalated, audited)."""
        escalated = self.tracker.observe(gap)
        audited = False
        if not escalated:
            self.audit.record_kept()
            audited = self.audit.should_audit(unit_random)
        return escalated, audited

    def record_audit_outcome(self, weak_was_bad: bool) -> None:
        """One audit's label moves the target: up at once, down on a quota."""
        self.audit.record_audit(weak_was_bad)
        if weak_was_bad:
            self.clean_audit_streak = 0
            self.raise_count += 1
            self._scale_target(self.adjustment.adjust_factor)
            return
        self.clean_audit_streak += 1
        quota = self.relax_audit_quota()
        if self.clean_audit_streak >= quota:
            self.clean_audit_streak = 0
            self.relax_count += 1
            relax_factor = 1.0 / self.adjustment.adjust_factor
            self._scale_target(relax_factor)

    def _scale_target(self, factor: float) -> None:
        proposed = self.tracker.target_escalation_rate * factor
        if proposed < self.adjustment.min_escalation_rate:
            proposed = self.adjustment.min_escalation_rate
        if proposed > self.adjustment.max_escalation_rate:
            proposed = self.adjustment.max_escalation_rate
        self.tracker.target_escalation_rate = proposed


class OnlineThreshold:
    """The controller at Switching's decision point (threshold_source online).

    One instance persists across every shot of a sweep point, so the
    controller learns over the point's whole window stream; its random
    stream is seeded once at construction, so a rerun of the point
    reproduces the same audits. An audited window still escalates (the
    strong result commits, so the audit costs latency, never accuracy)
    and is remembered here; when its strong result arrives, the label is
    whether the strong answer revised the weak committed observables.
    """

    audits_by_escalating = True

    def __init__(
        self, controller: OnlineThresholdController, random_generator
    ) -> None:
        self.controller = controller
        self.random_generator = random_generator
        # (operation id, window id) -> the weak observables under audit
        self._pending_audits: dict = {}
        # (window count, threshold, event) at the start, every hundredth
        # window, and every audit and its label
        self.trajectory: list = []
        self._record("start")

    def decide_keep(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> bool:
        """True keeps the weak result, False escalates.

        An audit escalates with the decision recorded for labeling.
        """
        unit_random = self.random_generator.random()
        escalated, audited = self.controller.observe(
            result.soft_output.gap, unit_random
        )
        if self.controller.tracker.window_count % 100 == 0:
            self._record("sample")
        if audited:
            if result.logical_observables is None:
                raise ValueError(
                    "online threshold calibration needs the weak decoder "
                    "to produce logical observables for audit labels; "
                    "the configured weak card is timing-only"
                )
            key = (job.op_id, job.window_id)
            self._pending_audits[key] = tuple(result.logical_observables)
            self._record("audit")
            return False
        return not escalated

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """A strong result arrived; if it answers an audit, label it."""
        weak_observables = self._pending_audits.pop(window_key, None)
        if weak_observables is None:
            return
        if result.logical_observables is None:
            raise ValueError(
                f"audited window {window_key}: the strong result carries no "
                "logical observables to compare against"
            )
        strong_observables = tuple(result.logical_observables)
        weak_was_bad = strong_observables != weak_observables
        self.controller.record_audit_outcome(weak_was_bad)
        event = "audit_clean"
        if weak_was_bad:
            event = "audit_bad"
        self._record(event)

    def summary(self) -> dict:
        """The source's counters and its live threshold."""
        tracker = self.controller.tracker
        audit = self.controller.audit
        return {
            "windows": tracker.window_count,
            "escalated": tracker.escalated_count,
            "escalation_rate": tracker.escalation_rate(),
            "threshold": tracker.threshold,
            "target_escalation_rate": tracker.target_escalation_rate,
            "audited": audit.audited_count,
            "audited_bad": audit.audited_bad_count,
            "kept_bad_rate": audit.kept_bad_rate(tracker.window_count),
            "raises": self.controller.raise_count,
            "relaxes": self.controller.relax_count,
            "pending_audits": len(self._pending_audits),
        }

    def _record(self, event: str) -> None:
        tracker = self.controller.tracker
        self.trajectory.append((tracker.window_count, tracker.threshold, event))
