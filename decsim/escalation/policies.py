"""The escalation policies: Baseline, StrongOnly and Switching.

Baseline decodes every window on the weak tier and keeps every result;
StrongOnly decodes every window once on the strong tier; Switching
decodes weak first and escalates a window whose confidence falls below
its threshold (Toshio et al. 2510.25222 Sec. III A). A policy decides
and is told (gem5's conditional predictor, src/cpu/pred/conditional.hh):
it builds no job; the window side plans and submits the strong re-decode
(strong_redecode.py, strong_window_shapes.py) and the decoder manager
owns the units, hold-or-deliver and cancellation. Where Switching's
threshold comes from is its ThresholdSource (threshold_sources.py).
"""

import decsim.controller.policies as boundary_policies
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.windows.windowing_schemes as windowing_schemes


class EscalationPolicyBase:
    """The defaults a policy row inherits (sinter's Decoder shape).

    A row declares primary_tier and requires_strong_context and answers
    verdict_for_weak_result; the rest has a default here: every run
    shape is served, the primary tier alone decodes a ready window, and
    nothing is learned from a strong result.
    """

    def check_plan(self, plan: decoding_records.RunShape) -> None:
        """Every run shape is served."""
        del plan

    def tiers_for_ready_window(self, window: window_records.Window) -> tuple:
        """The primary tier alone."""
        del window
        return (self.primary_tier,)

    def learn_from_strong_result(self, window_key: tuple, result) -> None:
        """Nothing is learned."""
        del window_key
        del result


class Baseline(EscalationPolicyBase):
    """Plain windowed decoding: every window once on the weak tier, kept."""

    requires_strong_context = False
    primary_tier = window_records.DecoderTier.WEAK

    def verdict_for_weak_result(self, job, result) -> decoding_records.Verdict:
        """Every result is final."""
        del job
        del result
        return decoding_records.Verdict.KEEP


class StrongOnly(EscalationPolicyBase):
    """The strong tier decodes the plan's windows directly.

    No weak decode, no escalation. The machine is data-woken: syndrome
    buffer 1 stores a round and its signal drives window readiness, the
    shape of LILLIPUT's FIFO-fed decoder and Google's streaming decoder.
    Every window job carries tier STRONG, reads its rounds from syndrome
    buffer 1 over strong_buffer_to_strong_decoder, and rides
    strong_decoder_to_frame home; the escalation machinery (the
    selection link, the ledger, the context windows, the strong windows)
    is never engaged.
    """

    requires_strong_context = False
    primary_tier = window_records.DecoderTier.STRONG

    def check_plan(self, plan: decoding_records.RunShape) -> None:
        """A static plan; dynamic streams re-point live window reads."""
        if plan.has_dynamic_streams:
            raise ValueError(
                "strong-only runs support static plans; dynamic streams "
                "re-point live window reads and are not wired to the "
                "room-side store yet"
            )

    def verdict_for_weak_result(self, job, result) -> decoding_records.Verdict:
        """Every result is final: the strong tier decoded it."""
        del job
        del result
        return decoding_records.Verdict.KEEP


class Switching(EscalationPolicyBase):
    """Weak decoder first; escalate to a strong decoder on low confidence.

    The threshold source decides keep from the weak result's soft
    output, whose source must be the one the policy expects; a result
    without a soft output escalates. run_both_at_once starts the strong
    sibling with the weak job and cancels it on confidence (the paper's
    Step 1, Toshio et al. 2510.25222 Sec. III A); otherwise the strong
    re-decode starts at the verdict, after the weak_decoder_to_strong_
    decoder hop (the serial modification of the same section). How the
    strong window is laid out is the run's shape (escalation.double_
    window: decsim's own two-sided context, or the forward window of
    Sec. III C, strong_window_shapes.py), and whether queued re-decodes are batched
    is the decoder manager's (bulk_strong); check_plan holds the policy's
    knobs and its threshold source against both once, at build.
    """

    requires_strong_context = True
    primary_tier = window_records.DecoderTier.WEAK

    def __init__(
        self,
        threshold,
        expected_source: decoding_records.SoftOutputSource,
        run_both_at_once: bool = False,
    ) -> None:
        if run_both_at_once and threshold.audits_by_escalating:
            raise ValueError(
                "online threshold calibration is meaningless with "
                "run_both_at_once: the strong decoder already runs "
                "for every window, so there is nothing to audit"
            )
        self.threshold = threshold
        self.expected_source = expected_source
        self.run_both_at_once = run_both_at_once

    def check_plan(self, plan: decoding_records.RunShape) -> None:
        """Refuse a run shape the escalation cannot serve."""
        _refuse_flush_terminal(plan.scheme)
        if plan.is_bulk_strong and self.run_both_at_once:
            raise ValueError(
                "bulk_strong is only meaningful in serial mode "
                "(run_both_at_once=False)"
            )
        if not plan.is_double_window:
            _refuse_eager_serial_boundaries(plan.boundary_policy)
            return
        self._refuse_double_window_contradictions(plan)
        _refuse_double_window_scheme(plan.scheme, plan.boundary_policy)
        _refuse_crossing_strong_region(plan)
        _refuse_double_window_run(plan)

    def tiers_for_ready_window(self, window: window_records.Window) -> tuple:
        """The weak tier, and the strong tier too when both run at once."""
        del window
        if self.run_both_at_once:
            return (
                window_records.DecoderTier.WEAK,
                window_records.DecoderTier.STRONG,
            )
        return (window_records.DecoderTier.WEAK,)

    def verdict_for_weak_result(self, job, result) -> decoding_records.Verdict:
        """Keep a confident weak result; otherwise escalate its window.

        The decision is made exactly once per window, since an online
        source learns from every call.
        """
        soft_output = result.soft_output
        if soft_output is None:
            return decoding_records.Verdict.ESCALATE
        if soft_output.source != self.expected_source:
            raise ValueError(
                "decoder confidence source does not match the switching "
                "threshold source"
            )
        if self.threshold.decide_keep(job, result):
            # a kept result cancels the parallel sibling
            return decoding_records.Verdict.KEEP
        return decoding_records.Verdict.ESCALATE

    def learn_from_strong_result(self, window_key: tuple, result) -> None:
        """The threshold source hears the strong tier's answer."""
        self.threshold.learn_from_strong_result(window_key, result)

    def _refuse_double_window_contradictions(
        self, plan: decoding_records.RunShape
    ) -> None:
        """A forward strong window starts late and alone; these knobs do not."""
        if self.run_both_at_once:
            raise ValueError(
                "double_window defers the strong start until the far weak "
                "boundary exists; run_both_at_once starts it immediately "
                "(the two policies contradict; pick one)"
            )
        if plan.is_bulk_strong:
            raise ValueError(
                "double_window + bulk_strong is not supported: deferred "
                "strong windows are submitted one per escalation"
            )
        if self.threshold.audits_by_escalating:
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


def _refuse_crossing_strong_region(plan: decoding_records.RunShape) -> None:
    """The forward strong window must end where a weak commit region ends.

    The shipped interaction's strong region is commit plus two buffers
    from the escalated window's commit start (window_interactions.py),
    and the sliding scheme commits in strides of commit_rounds, so the
    region ends on a stride edge exactly when twice buffer_rounds is a
    multiple of commit_rounds. Otherwise a later window commits across
    the region's end and has no owner; the shape stops the run there
    (strong_window_shapes.py), and the settings say so at build.
    """
    commit_round_count = plan.commit_round_count
    buffer_round_count = plan.buffer_round_count
    strong_round_count = commit_round_count + 2 * buffer_round_count
    if strong_round_count % commit_round_count == 0:
        return
    raise ValueError(
        f"windows.commit_rounds {commit_round_count} with "
        f"windows.buffer_rounds {buffer_round_count} gives the double "
        f"window a strong region of {strong_round_count} rounds that ends "
        "inside a later window's commit region; the double window's "
        "strong region, commit plus two buffers, must end inside its own "
        "commit region, so twice buffer_rounds must be a multiple of "
        "commit_rounds"
    )


def _refuse_double_window_run(plan: decoding_records.RunShape) -> None:
    """A double window needs static, explicit, single-patch operations."""
    if plan.has_dynamic_streams or plan.has_static_decode_plan:
        raise ValueError(
            "double_window skips statically planned windows when a "
            "strong window is assigned; stream windows created or "
            "folded at runtime (dynamic_streams/decode_ops) are not "
            "supported yet"
        )
    if plan.has_frontend:
        raise ValueError(
            "double_window is validated for explicit ops= workloads; "
            "frontend-built operation chains are not supported yet"
        )
    for operation in plan.operations:
        if operation.decoder_boundary_predecessors:
            raise ValueError(
                "double_window supports one single-patch stream per "
                "operation; decoder-boundary chains would let a strong "
                "window cross an operation seam before its far "
                "boundary exists"
            )
