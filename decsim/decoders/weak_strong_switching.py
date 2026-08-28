"""The two escalation policies: Baseline and Switching.

Baseline is the default (weak decoder only). Switching escalates weak decodes to strong; the
weak/strong routing itself stays in the router (SwitchingRouter), and the
decoder manager owns unit bookkeeping, hold-or-deliver, and cancellation.
"""

from __future__ import annotations

import math
from typing import Optional

from ..message import (
    RunSeedChild,
    RunSeedPathSegment,
    SoftOutputSource,
)
from ..message import DecoderTier, Directive, OutcomeDirective, Submission


class ThresholdRegister:
    """Per-code confidence thresholds, updatable at runtime (P17).

    The actuator half of a calibration loop: get(code) serves the
    router (falling back to `default`), set(code, g) updates a lane
    and records (seq, code, old, new) in `history` for audit. The
    loop that COMPUTES new thresholds is out of scope (row 11's
    remaining open item)."""

    def __init__(
        self,
        default: float,
        expected_source: SoftOutputSource,
        per_code: dict = None,
    ):
        initial_per_code = dict(per_code or {})
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
        self._seq = 0

    def get(self, code) -> float:
        return self.per_code.get(code, self.default)

    def set(self, code, threshold: float) -> None:
        if not code:
            raise ValueError(
                "threshold-register code identities must be nonempty"
            )
        old = self.get(code)
        self.per_code[code] = float(threshold)
        self._seq += 1
        self.history.append(
            (
                self._seq,
                code,
                old,
                float(threshold),
                self.expected_source,
            )
        )


class EscalationRateTracker:
    """Inner calibration loop: pin the escalation fraction at a target.

    Label-free: every window reveals whether it escalated
    (gap < threshold), so the escalation FRACTION is fully observable.
    The update is the adaptive conformal recursion (Gibbs and Candes,
    arXiv:2106.00170): raise the threshold a little on every kept
    window, lower it a lot on every escalation, balancing at the target
    quantile of the live gap distribution with a per-sequence long-run
    guarantee under any drift. Hardware precedent: O-GEHL's
    update threshold is servoed by the same event-rate-balancing
    counter (Seznec, CBP-1 2004).

    Threshold and step are in the gap's own unit, natural-log weight
    (nats), the unit ``Switching`` compares in; the yaml layer converts
    from the paper's decibels."""

    def __init__(self, target_escalation_rate: float, threshold: float,
                 step: float):
        self.target_escalation_rate = target_escalation_rate
        self.threshold = threshold
        self.step = step
        self.window_count = 0
        self.escalated_count = 0

    def observe(self, gap: float) -> bool:
        """Consume one window's gap; return True when it escalates."""
        escalated = gap < self.threshold
        self.window_count += 1
        self.escalated_count += int(escalated)
        move = self.target_escalation_rate - float(escalated)
        self.threshold += self.step * move
        if self.threshold < 0.0:
            self.threshold = 0.0
        return escalated

    def escalation_rate(self) -> float:
        if self.window_count == 0:
            return 0.0
        return self.escalated_count / self.window_count


class AuditLane:
    """Randomized strong-decoder audits of kept windows.

    The lane is cache set-dueling's move (Qureshi et al., ISCA 2007):
    dedicate a small fixed sample to the expensive path so ground truth
    keeps flowing whatever threshold is live. Sampling KEPT windows is
    what breaks the selective-labels bias (Lakkaraju et al., KDD 2017):
    without it every label comes from below the threshold and the kept
    region is pure extrapolation.

    ``kept_bad_rate`` estimates P(weak revised AND kept) per window,
    the quantity the paper's Eq. 4 bounds with epsilon * PL_strong.
    Each audited bad outcome counts 1/audit_rate kept windows (inverse
    propensity). The label is disagreement with the strong result, the
    reference the paper's protocol also measures against."""

    def __init__(self, audit_rate: float):
        self.audit_rate = audit_rate
        self.audited_count = 0
        self.audited_bad_count = 0
        self.weighted_bad_sum = 0.0
        self.kept_count = 0

    def should_audit(self, unit_random: float) -> bool:
        return unit_random < self.audit_rate

    def record_kept(self) -> None:
        self.kept_count += 1

    def record_audit(self, weak_was_bad: bool) -> None:
        self.audited_count += 1
        if weak_was_bad:
            self.audited_bad_count += 1
            self.weighted_bad_sum += 1.0 / self.audit_rate

    def kept_bad_rate(self, total_window_count: int) -> float:
        if total_window_count == 0:
            return 0.0
        return self.weighted_bad_sum / total_window_count


class OnlineThresholdController:
    """Both calibration loops together: duty tracking steered by audits.

    The two directions of the outer loop have asymmetric evidence costs
    and are handled asymmetrically:

      raise    each audited bad outcome carries importance weight one
               over the audit rate, so one event is already strong
               evidence the budget is blown. The escalation target is
               multiplied by ``adjust_factor`` immediately.
      relax    certifying the bad rate is BELOW budget needs the
               rule-of-three quota, about 3 / kept_bad_budget clean
               audits for a 95% upper bound at the budget. Only a full
               clean quota shrinks the target.

    The target always stays inside [min_escalation_rate,
    max_escalation_rate]; the max is the Theorem 1 backlog bound: the
    strong tier's duty cycle may never exceed what its latency can
    absorb, whatever accuracy would prefer."""

    def __init__(self, tracker: EscalationRateTracker, audit: AuditLane,
                 kept_bad_budget: float, adjust_factor: float,
                 min_escalation_rate: float, max_escalation_rate: float):
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
        return int(math.ceil(3.0 / self.kept_bad_budget))

    def observe(self, gap: float, unit_random: float) -> tuple:
        """One window: (escalated, audited)."""
        escalated = self.tracker.observe(gap)
        audited = False
        if not escalated:
            self.audit.record_kept()
            audited = self.audit.should_audit(unit_random)
        return escalated, audited

    def record_audit_outcome(self, weak_was_bad: bool) -> None:
        self.audit.record_audit(weak_was_bad)
        if weak_was_bad:
            self.clean_audit_streak = 0
            self.raise_count += 1
            self._scale_target(self.adjust_factor)
            return
        self.clean_audit_streak += 1
        if self.clean_audit_streak >= self.relax_audit_quota():
            self.clean_audit_streak = 0
            self.relax_count += 1
            self._scale_target(1.0 / self.adjust_factor)

    def _scale_target(self, factor: float) -> None:
        proposed = self.tracker.target_escalation_rate * factor
        if proposed < self.min_escalation_rate:
            proposed = self.min_escalation_rate
        if proposed > self.max_escalation_rate:
            proposed = self.max_escalation_rate
        self.tracker.target_escalation_rate = proposed


class OnlineGapCalibrator:
    """The controller wired to Switching's decision point.

    Owns the live threshold when ``switching.threshold_source: online``
    (the ThresholdRegister stays the actuator for externally computed
    thresholds and is refused alongside this). One instance persists
    across every shot of a sweep point, so the controller learns over
    the point's whole window stream, the same continuous stream the
    drift-replay validation ran on; its random stream is seeded once at
    construction, so a rerun of the point reproduces the same audits.

    An audited window still escalates (the strong result commits, so
    the audit costs latency, never accuracy) and is remembered here;
    when its strong result arrives, the label is whether the strong
    answer revised the weak committed observables."""

    def __init__(self, controller: OnlineThresholdController, rng):
        self.controller = controller
        self.rng = rng
        self._pending_audits: dict = {}   # (op_id, window_id) -> weak observables
        self.trajectory: list = []        # (window_count, threshold, event)
        self._record("start")

    def _record(self, event: str) -> None:
        self.trajectory.append(
            (self.controller.tracker.window_count,
             self.controller.tracker.threshold, event))

    def decide_keep(self, result, job) -> bool:
        """One weak outcome: True keeps the weak result, False escalates
        (an audit escalates with the decision recorded for labeling)."""
        escalated, audited = self.controller.observe(
            result.soft_output.gap, self.rng.random())
        if self.controller.tracker.window_count % 100 == 0:
            self._record("sample")
        if audited:
            if result.logical_observables is None:
                raise ValueError(
                    "online threshold calibration needs the weak decoder "
                    "to produce logical observables for audit labels; "
                    "the configured weak card is timing-only")
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
                "logical observables to compare against")
        weak_was_bad = (
            tuple(strong_result.logical_observables) != weak_observables)
        self.controller.record_audit_outcome(weak_was_bad)
        self._record("audit_bad" if weak_was_bad else "audit_clean")

    def summary(self) -> dict:
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


class Baseline:
    """Plain windowed decoding: submit the weak job, accept every outcome."""

    requires_strong_context = False
    bulk_strong = False
    double_window = False
    primary_tier = DecoderTier.WEAK

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams,
        static_decode_plan_selected,
        has_frontend,
    ) -> None:
        pass

    def validate_operations(self, operations) -> None:
        pass

    def validate_code_geometry(self, geometry) -> None:
        pass

    def on_window_ready(self, window, weak_job, services) -> list:
        return [Submission(weak_job)]

    def on_decode_outcome(self, outcome, services) -> OutcomeDirective:
        if outcome.job.strong_decode_for is not None:
            return OutcomeDirective(Directive.FINALIZE_STRONG)
        return OutcomeDirective(Directive.FINALIZE)


class StrongOnly:
    """The strong tier decodes the plan's windows directly: no weak decode,
    no verdict, no escalation. The machine is data-woken: syndrome buffer 1
    stores a round and its signal drives window readiness, the shape of
    LILLIPUT's FIFO-fed decoder and Google's streaming decoder. Every window
    job carries tier STRONG, reads its rounds from syndrome buffer 1 over
    SBD, and rides DO home; the escalation machinery (WSD, ledger, context
    windows, strong windows) is never engaged."""

    requires_strong_context = False
    bulk_strong = False
    double_window = False
    primary_tier = DecoderTier.STRONG

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams,
        static_decode_plan_selected,
        has_frontend,
    ) -> None:
        if has_dynamic_streams:
            raise ValueError(
                "strong-only runs support static plans; dynamic streams "
                "re-point live window reads and are not wired to the "
                "room-side store yet")

    def validate_operations(self, operations) -> None:
        pass

    def validate_code_geometry(self, geometry) -> None:
        pass

    def on_window_ready(self, window, weak_job, services) -> list:
        # the pre-built job IS the window's job; the submit path already
        # stamped it with the primary tier, the primary store's payloads,
        # and the SBD transfer
        return [Submission(weak_job)]

    def on_decode_outcome(self, outcome, services) -> OutcomeDirective:
        return OutcomeDirective(Directive.FINALIZE)


class Switching:
    """Weak decoder first; escalate to a strong decoder on low confidence.

    The confidence threshold gates keep-weak only for the exact configured
    source (`soft_output.gap >= threshold`); serial mode escalates after the
    ws hop; run_both_at_once starts
    the strong sibling with the weak and cancels it on confidence; bulk_strong
    batches queued serial redos (timing-only). Redo covers commit + 2*buffer
    rounds (the paper's two-sided context).

    With ``double_window=True``, the strong window contains the
    suspicious commit region plus one buffer on each side. It starts at
    the suspicious commit and extends forward; the weak chain skips the
    windows it absorbs and restarts past it; the strong result owns the
    whole extent; the strong job starts only after both of its
    boundaries are weak-determined (left: the commits before it, right:
    the restart window's commit, or the terminal boundary). The weak
    pipeline never waits on strong work.

    Seam modelling: both faces of the strong window are decoded as
    two-sided B-windows:
    one buffer of raw context per face, exact fault-ownership partition,
    no folded decoded defects (folding at a raw-read face double-counts;
    see test_parallel_two_sided_windows_match_global_decoding). Unlike the
    paper's exactly-r_strong read with weak-pinned faces, the context
    reads are extra: seam-edge accuracy is slightly optimistic, and the
    strong window is priced for the whole context it reads rather than the
    r_strong rounds it commits, so its decode cost is conservative
    against Theorem 1 rather than optimistic. The transfer cost of the
    extra context still belongs to the strong-data-path backlog item."""

    requires_strong_context = True
    primary_tier = DecoderTier.WEAK

    def __init__(self, confidence_threshold: float,
                 expected_source: SoftOutputSource,
                 run_both_at_once: bool = False,
                 weak_keepup_ratio: Optional[float] = None,
                 bulk_strong: bool = False,
                 threshold_register: Optional["ThresholdRegister"] = None,
                 threshold_calibrator: Optional["OnlineGapCalibrator"] = None,
                 double_window: bool = False):
        if weak_keepup_ratio is not None and not 0 < weak_keepup_ratio < 1:
            raise ValueError(f"weak_keepup_ratio must be between 0 and 1 "
                             f"(got {weak_keepup_ratio})")
        if bulk_strong and run_both_at_once:
            raise ValueError("bulk_strong is only meaningful in serial mode "
                             "(run_both_at_once=False)")
        if double_window and run_both_at_once:
            raise ValueError(
                "double_window defers the strong start until the far weak "
                "boundary exists; run_both_at_once starts it immediately "
                "(the two policies contradict; pick one)")
        if double_window and bulk_strong:
            raise ValueError(
                "double_window + bulk_strong is not supported: deferred "
                "strong windows are submitted one per escalation")
        if threshold_register is not None:
            if confidence_threshold != threshold_register.default:
                raise ValueError(
                    "Switching confidence threshold must equal the threshold "
                    "register default"
                )
            if expected_source != threshold_register.expected_source:
                raise ValueError(
                    "Switching expected source must equal the threshold "
                    "register source"
                )
        if threshold_calibrator is not None:
            if threshold_register is not None:
                raise ValueError(
                    "an online threshold calibrator and a threshold "
                    "register are two owners for the same threshold; "
                    "configure one")
            if run_both_at_once:
                raise ValueError(
                    "online threshold calibration is meaningless with "
                    "run_both_at_once: the strong decoder already runs "
                    "for every window, so there is nothing to audit")
            if double_window:
                raise ValueError(
                    "online threshold calibration is serial-only: an "
                    "audit label compares one window's weak and strong "
                    "committed observables, and a double-window strong "
                    "result owns a larger extent than the audited window")
        self.confidence_threshold = confidence_threshold
        self.expected_source = expected_source
        self.threshold_register = threshold_register   # P17 (None = scalar)
        self.threshold_calibrator = threshold_calibrator
        self.run_both_at_once = run_both_at_once
        self.weak_keepup_ratio = weak_keepup_ratio
        self.bulk_strong = bulk_strong
        self.double_window = double_window

    def run_seed_children(self):
        """Expose the optional register that selects confidence thresholds."""
        if self.threshold_register is None:
            return ()
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "threshold_register"),),
                self.threshold_register,
            ),
        )

    # ------------------------------------------------------------ the hooks

    def on_window_ready(self, window, weak_job, services) -> list:
        submissions = [Submission(weak_job)]
        if self.run_both_at_once:              # parallel: no ws delay (dm:109-110)
            strong = services.make_strong_job(
                weak_job, self.strong_redo_rounds(window),
                getattr(weak_job, "strong_label", f"strong({weak_job.label})"))
            submissions.append(Submission(strong, delay_ticks=0))
        return submissions

    def on_decode_outcome(self, outcome, services) -> OutcomeDirective:
        job = outcome.job
        if job.strong_decode_for is not None:
            if self.threshold_calibrator is not None:
                self.threshold_calibrator.absorb_strong_result(
                    job.strong_decode_for, outcome.result)
            return OutcomeDirective(Directive.FINALIZE_STRONG)
        if job.attempt != 0 or self._keep_weak_outcome(outcome.result, job):
            return OutcomeDirective(Directive.FINALIZE)   # pool cancels sibling
        extra = None
        strong_request_key = None
        if self.double_window:
            # Register now. The window manager submits the strong window
            # after the far-side weak boundary is ready.
            strong_request_key = services.defer_strong_escalation(job)
        elif not self.run_both_at_once:        # serial: redo after ws (dm:153-154)
            strong = services.make_strong_job(
                job, self.strong_redo_rounds(job.window),
                getattr(job, "strong_label", f"strong({job.label})"))
            extra = Submission(strong)
            strong_request_key = strong.request_key
        return OutcomeDirective(
            Directive.AWAIT_STRONG,
            extra=extra,
            strong_request_key=strong_request_key,
        )

    # ------------------------------------------------------------ validation

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams,
        static_decode_plan_selected,
        has_frontend,
    ) -> None:
        from ..controller.policies import Eager, Held
        from ..windows.windowing_schemes import SlidingTerminalPolicy, SlidingWindowScheme

        if (
            type(scheme) is SlidingWindowScheme
            and scheme.terminal_policy
            is not SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
        ):
            raise ValueError(
                "switching and strong-window recovery require the explicit "
                "REGULAR_STRIDE_LOOKAHEAD terminal policy; the literature-exact "
                "QUITS/Tan all-core flush has no trailing tail context"
            )
        if self.weak_keepup_ratio is not None and (
            type(scheme) is not SlidingWindowScheme
        ):
            raise ValueError(
                "weak_keepup_ratio implements the exact shipped serial "
                "sliding keep-up contract and requires SlidingWindowScheme"
            )
        if not self.double_window:
            if isinstance(boundary_policy, Eager):
                raise ValueError(
                    "serial switching requires Held boundaries: an eagerly "
                    "shipped provisional boundary is never corrected when "
                    "the strong result later revises the window"
                )
            return
        if type(scheme) is not SlidingWindowScheme:
            raise ValueError(
                "double_window requires the exact shipped serial "
                "SlidingWindowScheme"
            )
        if isinstance(boundary_policy, Held):
            raise ValueError(
                "double_window requires the weak chain to keep committing "
                "(the far boundary IS the restart window's weak commit); "
                "the Held boundary policy would make later windows wait for "
                "the strong result and deadlock the strong window")
        if has_dynamic_streams or static_decode_plan_selected:
            raise ValueError(
                "double_window skips statically planned windows when a "
                "strong window is assigned; stream windows created or folded at runtime "
                "(dynamic_streams/decode_ops) are not supported yet")
        if has_frontend:
            raise ValueError(
                "double_window is validated for explicit ops= workloads; "
                "frontend-built operation chains are not supported yet")

    def validate_operations(self, operations) -> None:
        if self.double_window and any(
            operation.decoder_boundary_predecessors
            for operation in operations
        ):
            raise ValueError(
                "double_window supports one single-patch stream per operation; "
                "decoder-boundary chains would let a strong window cross "
                "an operation seam before its far boundary exists")

    def validate_code_geometry(self, geometry) -> None:
        if self.weak_keepup_ratio is None:
            return
        ratio = self.weak_keepup_ratio
        commit_rounds = geometry.commit_round_count
        buffer_rounds = geometry.buffer_round_count
        if ratio > commit_rounds / (commit_rounds + buffer_rounds):
            needed = math.ceil(ratio / (1 - ratio) * buffer_rounds)
            raise ValueError(
                f"commit region of {commit_rounds} rounds too short for "
                f"weak_keepup_ratio={ratio} (needs >= {needed}); use a bigger "
                f"commit region or lower weak_keepup_ratio."
            )

    # ---------------------------------------------------------- policy knobs

    def _keep_weak_outcome(self, result, job) -> bool:
        """The run's keep decision for one weak outcome, made exactly
        once per window: the calibrator's when one is configured (it
        learns from every call), the fixed threshold's otherwise."""
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

    def keep_weak_result(self, result, job) -> bool:
        """True when the weak result should be committed (switching.py:28-31).

        With a threshold_register configured and a job carrying a
        code, the register's per-code value replaces the scalar
        (P17); job=None or register=None preserves the original
        scalar semantics bit-identically."""
        threshold = self.confidence_threshold
        if (self.threshold_register is not None and job is not None
                and getattr(job, "code", None) is not None):
            threshold = self.threshold_register.get(job.code)
        if result is None or result.soft_output is None:
            return False
        if result.soft_output.source != self.expected_source:
            raise ValueError(
                "decoder confidence source does not match the switching "
                "threshold source"
            )
        return result.soft_output.gap >= threshold

    @staticmethod
    def strong_redo_rounds(window) -> int:
        """Rounds the strong decoder reprocesses: commit + 2*buffer."""
        commit = window.commit_hi - window.commit_lo + 1
        buffer = window.buffer_hi - window.commit_hi
        return commit + 2 * buffer
