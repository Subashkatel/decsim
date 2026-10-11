"""The switching policy: weak first, escalate on low confidence.

Toshio et al. 2510.25222 Sec. III A. A run with no switching slot has no
policy: every window is decoded once and kept. A policy decides and is
told, as gem5's conditional predictor is (src/cpu/pred/conditional.hh):
it builds no job; the window side plans and submits the strong
re-decode, and the decoder manager owns the units and cancellation.
"""

from typing import Optional

import decsim.escalation.verdicts as verdicts
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


class Switching:
    """Weak decoder first; escalate to a strong decoder on the verdict.

    verdict keeps or escalates each weak result (verdicts.py), by
    default on its gap against the threshold, where a result without a
    soft output escalates. run_both_at_once starts a speculative strong
    decode with the weak job and cancels it on a keep (Step 1);
    otherwise the strong re-decode starts at the verdict, after the
    weak_decoder_to_strong_decoder hop. check_plan refuses, once at
    build, a run shape the escalation cannot serve.
    """

    def __init__(
        self,
        threshold: ports.ThresholdSource,
        run_both_at_once: bool = False,
        verdict: Optional[ports.EscalationVerdict] = None,
        reads_threshold: bool = True,
    ) -> None:
        if run_both_at_once and threshold.audits_by_escalating:
            raise ValueError(
                "online threshold calibration is meaningless with "
                "run_both_at_once: the strong decoder already runs "
                "for every window, so there is nothing to audit"
            )
        if not reads_threshold and threshold.audits_by_escalating:
            raise ValueError(
                "online threshold calibration learns from the windows the "
                "gap escalates, and switching.verdict reads no threshold; "
                "use a fixed threshold or the gap_threshold verdict"
            )
        if verdict is None:
            verdict = verdicts.GapThreshold()
        self.threshold = threshold
        self.run_both_at_once = run_both_at_once
        self.verdict = verdict

    def check_plan(self, plan: decoding_records.RunShape) -> None:
        """Refuse a run shape the escalation cannot serve."""
        _refuse_flush_terminal(plan.scheme)
        if plan.is_bulk_strong and self.run_both_at_once:
            raise ValueError(
                "bulk_strong is only meaningful in serial mode "
                "(run_both_at_once=False)"
            )
        if not plan.is_absorbing_strong_window:
            # the boundaries the policy ships are the entry pool's
            if plan.leaves_the_entry_pool:
                _refuse_eager_serial_boundaries(plan.boundary_policy)
            return
        self._refuse_absorbing_window_contradictions(plan)
        _refuse_absorbing_window_scheme(
            plan.strong_window, plan.scheme, plan.boundary_policy
        )
        _refuse_absorbing_window_run(plan)

    def decodes_both_at_once(self, window: window_records.Window) -> bool:
        """Whether the strong tier starts beside the weak one (Step 1)."""
        del window
        return self.run_both_at_once

    def verdict_for_weak_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> decoding_records.Verdict:
        """The verdict row's decision on one weak result.

        The decision is made exactly once per window, since an online
        source learns from every call.
        """
        return self.verdict.decide(job, result, self.threshold)

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """The threshold source hears the strong tier's answer."""
        self.threshold.learn_from_strong_result(window_key, result)

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
