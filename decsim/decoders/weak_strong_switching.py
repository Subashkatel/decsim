"""The escalation policies: Baseline, StrongOnly and Switching.

Baseline decodes every window on the weak tier and keeps every result;
StrongOnly decodes every window once on the strong tier; Switching
decodes weak first and escalates a window whose confidence falls below
the threshold (Toshio et al. 2510.25222 Sec. III A). The threshold is a
fixed value, a per-code register, or the online calibrator that adapts
it across a sweep point's shots. The routing between the tiers is the
SwitchingRouter's (decoders.py); the decoder manager owns the units,
hold-or-deliver and cancellation.

Wide state recorded: ThresholdRegister sets eight attributes,
OnlineThresholdController nine and Switching eight. Slice 7's
structural commits make the threshold a ThresholdSource row and trim
Switching to the EscalationPolicy port (design note
docs/rewrite/notes/slice_07_escalation.md, section 3), so the width is
recorded, not split.
"""

import math
from typing import Optional

import decsim.controller.policies as boundary_policies
import decsim.message as message
import decsim.windows.windowing_schemes as windowing_schemes


class ThresholdRegister:
    """Per-code confidence thresholds, updatable at runtime.

    The actuator half of a calibration loop: get(code) serves the
    policy (falling back to `default`), set(code, threshold) updates a
    lane and records (sequence number, code, old, new, source) in
    `history`. The loop that computes new thresholds is not here.
    """

    def __init__(
        self,
        default: float,
        expected_source: message.SoftOutputSource,
        per_code: Optional[dict] = None,
    ) -> None:
        initial_per_code = {}
        if per_code is not None:
            initial_per_code = dict(per_code)
        for code in initial_per_code:
            if not code:
                raise ValueError(
                    "threshold-register code identities must be nonempty"
                )
        self.default = float(default)
        self.expected_source = expected_source
        self.per_code = initial_per_code
        self._initial_default = self.default
        self._initial_per_code = dict(self.per_code)
        self.history: list = []
        self._sequence_number = 0

    def get(self, code) -> float:
        """The code's threshold, or the default."""
        return self.per_code.get(code, self.default)

    def set(self, code, threshold: float) -> None:
        """Replace one code's threshold and record the change."""
        if not code:
            raise ValueError(
                "threshold-register code identities must be nonempty"
            )
        old = self.get(code)
        new = float(threshold)
        self.per_code[code] = new
        self._sequence_number += 1
        self.history.append(
            (self._sequence_number, code, old, new, self.expected_source)
        )


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

    Threshold and step are in the gap's own unit, natural-log weight
    (nats), the unit Switching compares in; the yaml layer converts
    from the paper's decibels.
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
        kept_bad_budget: float,
        adjust_factor: float,
        min_escalation_rate: float,
        max_escalation_rate: float,
    ) -> None:
        self.tracker = tracker
        self.audit = audit
        self.kept_bad_budget = kept_bad_budget
        self.adjust_factor = adjust_factor
        self.min_escalation_rate = min_escalation_rate
        self.max_escalation_rate = max_escalation_rate
        self.clean_audit_streak = 0
        self.raise_count = 0
        self.relax_count = 0

    def relax_audit_quota(self) -> int:
        """Clean audits needed before a relax is statistically earned."""
        quota = 3.0 / self.kept_bad_budget
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
            self._scale_target(self.adjust_factor)
            return
        self.clean_audit_streak += 1
        quota = self.relax_audit_quota()
        if self.clean_audit_streak >= quota:
            self.clean_audit_streak = 0
            self.relax_count += 1
            relax_factor = 1.0 / self.adjust_factor
            self._scale_target(relax_factor)

    def _scale_target(self, factor: float) -> None:
        proposed = self.tracker.target_escalation_rate * factor
        if proposed < self.min_escalation_rate:
            proposed = self.min_escalation_rate
        if proposed > self.max_escalation_rate:
            proposed = self.max_escalation_rate
        self.tracker.target_escalation_rate = proposed


class OnlineGapCalibrator:
    """The controller wired to Switching's decision point.

    Owns the live threshold when threshold_source is online (the
    ThresholdRegister stays the actuator for externally computed
    thresholds and is refused alongside this). One instance persists
    across every shot of a sweep point, so the controller learns over
    the point's whole window stream; its random stream is seeded once
    at construction, so a rerun of the point reproduces the same
    audits.

    An audited window still escalates (the strong result commits, so
    the audit costs latency, never accuracy) and is remembered here;
    when its strong result arrives, the label is whether the strong
    answer revised the weak committed observables.
    """

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

    def decide_keep(self, result, job) -> bool:
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

    def absorb_strong_result(self, key: tuple, strong_result) -> None:
        """A strong result arrived; if it answers an audit, label it."""
        weak_observables = self._pending_audits.pop(key, None)
        if weak_observables is None:
            return
        if strong_result.logical_observables is None:
            raise ValueError(
                f"audited window {key}: the strong result carries no "
                "logical observables to compare against"
            )
        strong_observables = tuple(strong_result.logical_observables)
        weak_was_bad = strong_observables != weak_observables
        self.controller.record_audit_outcome(weak_was_bad)
        event = "audit_clean"
        if weak_was_bad:
            event = "audit_bad"
        self._record(event)

    def summary(self) -> dict:
        """The calibrator's counters and its live threshold."""
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


class Baseline:
    """Plain windowed decoding: submit the weak job, accept every outcome."""

    requires_strong_context = False
    bulk_strong = False
    double_window = False
    primary_tier = message.DecoderTier.WEAK

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams,
        static_decode_plan_selected,
        has_frontend,
    ) -> None:
        """Every run shape decodes weakly."""
        del scheme
        del boundary_policy
        del has_dynamic_streams
        del static_decode_plan_selected
        del has_frontend

    def validate_operations(self, operations) -> None:
        """Every workload decodes weakly."""
        del operations

    def validate_code_geometry(self, geometry) -> None:
        """Every window geometry decodes weakly."""
        del geometry

    def on_window_ready(self, window, weak_job, services) -> list:
        """The weak job, alone."""
        del window
        del services
        return [message.Submission(weak_job)]

    def on_decode_outcome(self, outcome, services) -> message.OutcomeDirective:
        """Every result is final."""
        del services
        if outcome.job.strong_decode_for is not None:
            return message.OutcomeDirective(message.Directive.FINALIZE_STRONG)
        return message.OutcomeDirective(message.Directive.FINALIZE)


class StrongOnly:
    """The strong tier decodes the plan's windows directly.

    No weak decode, no verdict, no escalation. The machine is
    data-woken: syndrome buffer 1 stores a round and its signal drives
    window readiness, the shape of LILLIPUT's FIFO-fed decoder and
    Google's streaming decoder. Every window job carries tier STRONG,
    reads its rounds from syndrome buffer 1 over
    strong_buffer_to_strong_decoder, and rides strong_decoder_to_frame
    home; the escalation machinery (the selection link, the ledger, the
    context windows, the strong windows) is never engaged.
    """

    requires_strong_context = False
    bulk_strong = False
    double_window = False
    primary_tier = message.DecoderTier.STRONG

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams,
        static_decode_plan_selected,
        has_frontend,
    ) -> None:
        """A static plan; dynamic streams re-point live window reads."""
        del scheme
        del boundary_policy
        del static_decode_plan_selected
        del has_frontend
        if has_dynamic_streams:
            raise ValueError(
                "strong-only runs support static plans; dynamic streams "
                "re-point live window reads and are not wired to the "
                "room-side store yet"
            )

    def validate_operations(self, operations) -> None:
        """Every workload decodes on the strong tier."""
        del operations

    def validate_code_geometry(self, geometry) -> None:
        """Every window geometry decodes on the strong tier."""
        del geometry

    def on_window_ready(self, window, weak_job, services) -> list:
        """The pre-built job is the window's job.

        The submit path already stamped it with the primary tier, the
        primary store's payloads and the strong_buffer_to_strong_decoder
        transfer.
        """
        del window
        del services
        return [message.Submission(weak_job)]

    def on_decode_outcome(self, outcome, services) -> message.OutcomeDirective:
        """Every result is final."""
        del outcome
        del services
        return message.OutcomeDirective(message.Directive.FINALIZE)


class Switching:
    """Weak decoder first; escalate to a strong decoder on low confidence.

    The confidence threshold gates keep-weak only for the exact
    configured source (soft_output.gap >= threshold); serial mode
    escalates after the weak_decoder_to_strong_decoder hop;
    run_both_at_once starts the strong sibling with the weak and cancels
    it on confidence (the paper's Step 1); bulk_strong batches queued
    serial redos (timing-only). The redo covers commit + 2 buffer rounds
    (the paper's two-sided context).

    With double_window, the strong window contains the suspicious
    commit region plus one buffer on each side. It starts at the
    suspicious commit and extends forward; the weak chain skips the
    windows it absorbs and restarts past it; the strong result owns the
    whole extent; the strong job starts only after both of its
    boundaries are weak-determined (left: the commits before it, right:
    the restart window's commit, or the terminal boundary). The weak
    pipeline never waits on strong work.

    Seam modelling: both faces of the strong window are decoded as
    two-sided windows: one buffer of raw context per face, exact
    fault-ownership partition, no folded decoded defects (folding at a
    raw-read face double-counts). Unlike the paper's exactly-r_strong
    read with weak-pinned faces, the context reads are extra: seam-edge
    accuracy is slightly optimistic, and the strong window is priced
    for the whole context it reads rather than the r_strong rounds it
    commits, so its decode cost is conservative against Theorem 1
    rather than optimistic.
    """

    requires_strong_context = True
    primary_tier = message.DecoderTier.WEAK

    def __init__(
        self,
        confidence_threshold: float,
        expected_source: message.SoftOutputSource,
        run_both_at_once: bool = False,
        bulk_strong: bool = False,
        threshold_register: Optional[ThresholdRegister] = None,
        threshold_calibrator: Optional[OnlineGapCalibrator] = None,
        double_window: bool = False,
    ) -> None:
        _refuse_contradicting_modes(
            run_both_at_once, bulk_strong, double_window
        )
        _refuse_register_mismatch(
            confidence_threshold, expected_source, threshold_register
        )
        if threshold_calibrator is not None:
            _refuse_calibrator_contradictions(
                threshold_register, run_both_at_once, double_window
            )
        self.confidence_threshold = confidence_threshold
        self.expected_source = expected_source
        self.threshold_register = threshold_register  # None: the scalar
        self.threshold_calibrator = threshold_calibrator
        self.run_both_at_once = run_both_at_once
        self.bulk_strong = bulk_strong
        self.double_window = double_window

    def run_seed_children(self) -> tuple:
        """The optional register that selects confidence thresholds."""
        if self.threshold_register is None:
            return ()
        segment = message.RunSeedPathSegment("field", "threshold_register")
        child = message.RunSeedChild((segment,), self.threshold_register)
        return (child,)

    # ---- the hooks

    def on_window_ready(self, window, weak_job, services) -> list:
        """The weak job, and the strong sibling when both run at once."""
        del window
        submissions = [message.Submission(weak_job)]
        if self.run_both_at_once:
            label = _strong_label_of(weak_job)
            strong = services.make_strong_job(weak_job, label)
            submissions.append(strong)
        return submissions

    def on_decode_outcome(self, outcome, services) -> message.OutcomeDirective:
        """Keep a confident weak result; otherwise await the strong one."""
        job = outcome.job
        if job.strong_decode_for is not None:
            if self.threshold_calibrator is not None:
                self.threshold_calibrator.absorb_strong_result(
                    job.strong_decode_for, outcome.result
                )
            return message.OutcomeDirective(message.Directive.FINALIZE_STRONG)
        if job.attempt != 0:
            return message.OutcomeDirective(message.Directive.FINALIZE)
        if self._keep_weak_outcome(outcome.result, job):
            # the pool cancels the parallel sibling
            return message.OutcomeDirective(message.Directive.FINALIZE)
        extra = None
        strong_request_key = None
        if self.double_window:
            # the window manager submits the strong window after the
            # far-side weak boundary is ready
            strong_request_key = services.defer_strong_escalation(job)
        elif not self.run_both_at_once:
            label = _strong_label_of(job)
            extra = services.make_strong_job(job, label)
            strong_request_key = extra.job.request_key
        return message.OutcomeDirective(
            message.Directive.AWAIT_STRONG,
            extra=extra,
            strong_request_key=strong_request_key,
        )

    # ---- validation

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams,
        static_decode_plan_selected,
        has_frontend,
    ) -> None:
        """Refuse a run shape the escalation cannot serve."""
        _refuse_flush_terminal(scheme)
        if not self.double_window:
            _refuse_eager_serial_boundaries(boundary_policy)
            return
        _refuse_double_window_scheme(scheme, boundary_policy)
        if has_dynamic_streams or static_decode_plan_selected:
            raise ValueError(
                "double_window skips statically planned windows when a "
                "strong window is assigned; stream windows created or "
                "folded at runtime (dynamic_streams/decode_ops) are not "
                "supported yet"
            )
        if has_frontend:
            raise ValueError(
                "double_window is validated for explicit ops= workloads; "
                "frontend-built operation chains are not supported yet"
            )

    def validate_operations(self, operations) -> None:
        """A double window needs one single-patch stream per operation."""
        if not self.double_window:
            return
        for operation in operations:
            if operation.decoder_boundary_predecessors:
                raise ValueError(
                    "double_window supports one single-patch stream per "
                    "operation; decoder-boundary chains would let a strong "
                    "window cross an operation seam before its far "
                    "boundary exists"
                )

    def validate_code_geometry(self, geometry) -> None:
        """Every window geometry switches."""
        del geometry

    # ---- the keep decision

    def keep_weak_result(self, result, job) -> bool:
        """True when the weak result is committed: gap at or above threshold.

        With a threshold_register configured and a job carrying a code,
        the register's per-code value replaces the scalar.
        """
        threshold = self.confidence_threshold
        if self._has_register_for(job):
            threshold = self.threshold_register.get(job.code)
        if result is None or result.soft_output is None:
            return False
        if result.soft_output.source != self.expected_source:
            raise ValueError(
                "decoder confidence source does not match the switching "
                "threshold source"
            )
        return result.soft_output.gap >= threshold

    def _has_register_for(self, job) -> bool:
        if self.threshold_register is None:
            return False
        if job is None:
            return False
        code = getattr(job, "code", None)
        return code is not None

    def _keep_weak_outcome(self, result, job) -> bool:
        """The keep decision, made exactly once per window.

        The calibrator's when one is configured (it learns from every
        call), the fixed threshold's otherwise.
        """
        if self.threshold_calibrator is None:
            return self.keep_weak_result(result, job)
        if result is None or result.soft_output is None:
            return False
        if result.soft_output.source != self.expected_source:
            raise ValueError(
                "decoder confidence source does not match the switching "
                "threshold source"
            )
        return self.threshold_calibrator.decide_keep(result, job)


def _strong_label_of(job) -> str:
    label = getattr(job, "strong_label", None)
    if label is None:
        return f"strong({job.label})"
    return label


def _refuse_contradicting_modes(
    run_both_at_once: bool, bulk_strong: bool, double_window: bool
) -> None:
    if bulk_strong and run_both_at_once:
        raise ValueError(
            "bulk_strong is only meaningful in serial mode "
            "(run_both_at_once=False)"
        )
    if double_window and run_both_at_once:
        raise ValueError(
            "double_window defers the strong start until the far weak "
            "boundary exists; run_both_at_once starts it immediately "
            "(the two policies contradict; pick one)"
        )
    if double_window and bulk_strong:
        raise ValueError(
            "double_window + bulk_strong is not supported: deferred "
            "strong windows are submitted one per escalation"
        )


def _refuse_register_mismatch(
    confidence_threshold: float,
    expected_source: message.SoftOutputSource,
    threshold_register: Optional[ThresholdRegister],
) -> None:
    if threshold_register is None:
        return
    if confidence_threshold != threshold_register.default:
        raise ValueError(
            "Switching confidence threshold must equal the threshold "
            "register default"
        )
    if expected_source != threshold_register.expected_source:
        raise ValueError(
            "Switching expected source must equal the threshold register source"
        )


def _refuse_calibrator_contradictions(
    threshold_register: Optional[ThresholdRegister],
    run_both_at_once: bool,
    double_window: bool,
) -> None:
    if threshold_register is not None:
        raise ValueError(
            "an online threshold calibrator and a threshold "
            "register are two owners for the same threshold; "
            "configure one"
        )
    if run_both_at_once:
        raise ValueError(
            "online threshold calibration is meaningless with "
            "run_both_at_once: the strong decoder already runs "
            "for every window, so there is nothing to audit"
        )
    if double_window:
        raise ValueError(
            "online threshold calibration is serial-only: an "
            "audit label compares one window's weak and strong "
            "committed observables, and a double-window strong "
            "result owns a larger extent than the audited window"
        )


def _refuse_flush_terminal(scheme) -> None:
    """Switching needs the lookahead terminal policy on sliding windows."""
    if type(scheme) is not windowing_schemes.SlidingWindowScheme:
        return
    lookahead = windowing_schemes.SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
    if scheme.terminal_policy is not lookahead:
        raise ValueError(
            "switching and strong-window recovery require the explicit "
            "REGULAR_STRIDE_LOOKAHEAD terminal policy; the literature-exact "
            "QUITS/Tan all-core flush has no trailing tail context"
        )


def _refuse_eager_serial_boundaries(boundary_policy) -> None:
    if isinstance(boundary_policy, boundary_policies.Eager):
        raise ValueError(
            "serial switching requires Held boundaries: an eagerly "
            "shipped provisional boundary is never corrected when "
            "the strong result later revises the window"
        )


def _refuse_double_window_scheme(scheme, boundary_policy) -> None:
    if type(scheme) is not windowing_schemes.SlidingWindowScheme:
        raise ValueError(
            "double_window requires the exact shipped serial "
            "SlidingWindowScheme"
        )
    if isinstance(boundary_policy, boundary_policies.Held):
        raise ValueError(
            "double_window requires the weak chain to keep committing "
            "(the far boundary IS the restart window's weak commit); "
            "the Held boundary policy would make later windows wait for "
            "the strong result and deadlock the strong window"
        )
