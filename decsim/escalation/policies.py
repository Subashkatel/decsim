"""The switching policy: weak first, escalate on low confidence.

Switching decodes weak first and escalates a window whose confidence
falls below its threshold (Toshio et al. 2510.25222 Sec. III A). A run
with no switching slot has no policy: every window is decoded once, on
the decoder that decodes the plan's windows, and kept. A policy decides
and is told (gem5's conditional predictor, src/cpu/pred/conditional.hh):
it builds no job; the window side plans and submits the strong re-decode
(strong_redecode.py, strong_window_shapes.py) and the decoder manager
owns the units, hold-or-deliver and cancellation. Where Switching's
threshold comes from is its ThresholdSource (threshold_sources.py).
"""

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


class Switching:
    """Weak decoder first; escalate to a strong decoder on low confidence.

    The threshold source decides keep from the weak result's soft output, whose
    source must be the one the policy expects; a result without a soft output
    escalates. run_both_at_once starts a speculative strong decode with the weak
    job and cancels it on confidence (the paper's Step 1, Toshio et al.
    2510.25222 Sec. III A); otherwise the strong re-decode starts at the
    verdict, after the weak_decoder_to_strong_decoder hop (the serial
    modification of the same section). How the strong window is laid out is the
    run's shape (the row escalation.strong_window names in STRONG_WINDOW_SHAPES,
    strong_window_shapes.py, which every refusal here names back), and whether
    queued re-decodes are batched is the decoder manager's (bulk_strong);
    check_plan holds the policy's knobs and its threshold source against both
    once, at build. threshold is the ThresholdSource it decides keep on,
    expected_source the soft output source the run's confidence signal
    reports.
    """

    # The burst detector: while one of its flags meets a window, the
    # machine is in burst mode and the window escalates whatever its
    # confidence. Unbound when burst_detector.kind is none, the machine
    # in normal mode throughout.
    burst_detector = ports.Port(ports.BurstDetector, optional=True)

    def __init__(
        self,
        threshold: ports.ThresholdSource,
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
        if not plan.is_absorbing_strong_window:
            _refuse_eager_serial_boundaries(plan.boundary_policy)
            return
        self._refuse_absorbing_window_contradictions(plan)
        _refuse_absorbing_window_scheme(
            plan.strong_window, plan.scheme, plan.boundary_policy
        )
        _refuse_crossing_strong_region(plan)
        _refuse_absorbing_window_run(plan)

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

        A window a published burst flag meets escalates first, without
        asking the threshold: Q3DE decodes the anomalous span again once
        the anomaly is detected (Suzuki et al. 2501.00331 lines
        966-981), and the strong tier is that decode here. The decision
        is made exactly once per window, since an online source learns
        from every call it is asked.
        """
        if self._is_in_a_burst(job.window):
            return decoding_records.Verdict.ESCALATE
        soft_output = result.soft_output
        if soft_output is None:
            return decoding_records.Verdict.ESCALATE
        if soft_output.source != self.expected_source:
            raise ValueError(
                "decoder confidence source does not match the switching "
                "threshold source"
            )
        if self.threshold.decide_keep(job, result):
            # a kept result cancels the speculative strong decode
            return decoding_records.Verdict.KEEP
        return decoding_records.Verdict.ESCALATE

    def learn_from_strong_result(self, window_key: tuple, result) -> None:
        """The threshold source hears the strong tier's answer."""
        self.threshold.learn_from_strong_result(window_key, result)

    def _is_in_a_burst(self, window: window_records.Window) -> bool:
        if self.burst_detector is None:
            return False
        return self.burst_detector.is_burst_window(window)

    def _refuse_absorbing_window_contradictions(
        self, plan: decoding_records.RunShape
    ) -> None:
        """An absorbing strong window starts late and alone; these do not."""
        row_name = plan.strong_window
        if self.run_both_at_once:
            raise ValueError(
                f"switching.strong_window {row_name} defers the strong "
                "start until the far weak boundary exists; "
                "run_both_at_once starts it immediately (the two "
                "policies contradict; pick one)"
            )
        if plan.is_bulk_strong:
            raise ValueError(
                f"switching.strong_window {row_name} with bulk_strong is "
                "not supported: deferred strong windows are submitted "
                "one per escalation"
            )
        if self.threshold.audits_by_escalating:
            raise ValueError(
                "online threshold calibration is serial-only: an "
                "audit label compares one window's weak and strong "
                "committed observables, and an absorbing strong window "
                "owns a larger extent than the audited window"
            )


def _refuse_flush_terminal(scheme) -> None:
    """Switching needs a last window that still reads past its commit."""
    if scheme.has_trailing_tail_context:
        return
    raise ValueError(
        "switching and strong-window recovery require a windowing scheme "
        "whose last window carries buffer context past its commit; this "
        "scheme has no trailing tail context"
    )


def _refuse_eager_serial_boundaries(boundary_policy) -> None:
    if boundary_policy.ships_provisional_boundaries:
        raise ValueError(
            "serial switching requires held boundaries: an eagerly "
            "shipped provisional boundary is never corrected when "
            "the strong result later revises the window"
        )


def _refuse_absorbing_window_scheme(
    row_name: str, scheme, boundary_policy
) -> None:
    if not scheme.commits_in_one_serial_chain:
        raise ValueError(
            f"switching.strong_window {row_name} requires a windowing "
            "scheme whose windows commit in one serial chain, since an "
            "absorbing strong window takes over the weak windows its "
            "region covers"
        )
    if not boundary_policy.ships_provisional_boundaries:
        raise ValueError(
            f"switching.strong_window {row_name} requires the weak chain "
            "to keep committing "
            "(the far boundary IS the restart window's weak commit); "
            "a boundary policy that holds provisional boundaries would "
            "make later windows wait for "
            "the strong result and deadlock the strong window"
        )


def _refuse_crossing_strong_region(plan: decoding_records.RunShape) -> None:
    """An absorbing strong region must end where a weak commit region ends.

    The shipped interaction's strong region is commit plus two buffers
    from the escalated window's commit start (window_interactions.py),
    and the sliding scheme commits in strides of commit_rounds, so the
    region ends on a stride edge exactly when twice buffer_rounds is a
    multiple of commit_rounds. Otherwise a later window commits across
    the region's end and has no owner; the shape stops the run there
    (strong_window_shapes.py), and the settings say so at build.
    """
    row_name = plan.strong_window
    commit_round_count = plan.commit_round_count
    buffer_round_count = plan.buffer_round_count
    strong_round_count = window_records.strong_region_round_count(
        commit_round_count, buffer_round_count
    )
    if strong_round_count % commit_round_count == 0:
        return
    raise ValueError(
        f"windows of {commit_round_count} commit rounds and "
        f"{buffer_round_count} buffer rounds give "
        f"switching.strong_window {row_name} a strong region of "
        f"{strong_round_count} rounds that ends inside a later window's "
        f"commit region; the strong region of {row_name}, commit plus two "
        "buffers, must end inside its own commit region, so twice "
        "buffer_rounds must be a multiple of commit_rounds"
    )


def _refuse_absorbing_window_run(plan: decoding_records.RunShape) -> None:
    """An absorbing window needs static, explicit, single-patch operations."""
    row_name = plan.strong_window
    if plan.has_dynamic_streams or plan.has_static_decode_plan:
        raise ValueError(
            f"switching.strong_window {row_name} skips statically planned "
            "windows when a "
            "strong window is assigned; stream windows created or "
            "folded at runtime (dynamic_streams/decode_ops) are not "
            "supported yet"
        )
    for operation in plan.operations:
        if operation.decoder_boundary_predecessors:
            raise ValueError(
                f"switching.strong_window {row_name} supports one "
                "single-patch stream per "
                "operation; decoder-boundary chains would let a strong "
                "window cross an operation seam before its far "
                "boundary exists"
            )
