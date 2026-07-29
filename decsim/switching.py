"""Decoder switching strategies (port 10): Baseline and Switching.

Part module: the strategy seam implementations (Contract 2c). Baseline is the
default. Switching escalates weak decodes to strong; the weak/strong
routing itself stays in the router (SwitchingRouter), and the pool owns unit
bookkeeping, hold-or-deliver, and cancellation mechanics.
"""

from __future__ import annotations

import math
from typing import Optional

from .message import (
    RunSeedChild,
    RunSeedPathSegment,
    SoftOutputSource,
)
from .protocols import Directive, OutcomeDirective, Submission


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
        if not isinstance(expected_source, SoftOutputSource):
            raise TypeError(
                "ThresholdRegister expected_source must be a SoftOutputSource"
            )
        initial_per_code = dict(per_code or {})
        for code in initial_per_code:
            if type(code) is not str:
                raise TypeError(
                    "threshold-register code identities must be exact strings"
                )
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
        if type(code) is not str:
            raise TypeError(
                "threshold-register code identities must be exact strings"
            )
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


class Baseline:
    """Plain windowed decoding: submit the weak job, accept every outcome."""

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

    def metrics(self) -> dict:
        return {}


class Switching:
    """Weak decoder first; escalate to a strong decoder on low confidence.

    The confidence threshold gates keep-weak only for the exact configured
    source (`soft_output.gap >= threshold`); serial mode escalates after the
    ws hop; run_both_at_once starts
    the strong sibling with the weak and cancels it on confidence; bulk_strong
    batches queued serial redos (timing-only). Redo covers commit + 2*buffer
    rounds (the paper's two-sided context).

    double_window=True is the faithful double-window protocol (arXiv:
    2510.25222 Sec. III C, Fig. 12): the slab of commit + 2*buffer rounds
    starts at the suspicious commit and extends forward; the weak chain
    skips the windows the slab absorbs and restarts past the slab; the
    strong result owns the whole slab; the strong job starts only after
    both slab boundaries are weak-determined (left: pre-slab commits,
    right: the restart window's commit, or the terminal boundary). The
    weak pipeline never waits on strong work.

    Seam modelling: both slab faces are decoded as two-sided B-windows:
    one buffer of raw context per face, exact fault-ownership partition,
    no folded decoded defects (folding at a raw-read face double-counts;
    see test_parallel_two_sided_windows_match_global_decoding). Unlike the
    paper's exactly-r_strong read with weak-pinned faces, the context
    reads are extra: seam-edge accuracy is slightly optimistic, and the
    slab is priced for the whole context it reads rather than the
    r_strong rounds it commits, so its decode cost is conservative
    against Theorem 1 rather than optimistic. The transfer cost of the
    extra context still belongs to the strong-data-path backlog item."""

    def __init__(self, confidence_threshold: float,
                 expected_source: SoftOutputSource,
                 run_both_at_once: bool = False,
                 weak_keepup_ratio: Optional[float] = None,
                 bulk_strong: bool = False,
                 threshold_register: Optional["ThresholdRegister"] = None,
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
                "slabs are submitted one per escalation")
        if not isinstance(expected_source, SoftOutputSource):
            raise TypeError(
                "Switching expected_source must be a SoftOutputSource"
            )
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
        self.confidence_threshold = confidence_threshold
        self.expected_source = expected_source
        self.threshold_register = threshold_register   # P17 (None = scalar)
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
            return OutcomeDirective(Directive.FINALIZE_STRONG)
        if job.attempt != 0 or self.keep_weak_result(outcome.result, job):
            return OutcomeDirective(Directive.FINALIZE)   # pool cancels sibling
        extra = None
        if self.double_window:
            # Faithful protocol: register the escalation only. The window
            # manager builds and submits the slab once the far-side weak
            # boundary is determined (paper Fig. 12 start condition).
            services.defer_strong_escalation(
                job)
        elif not self.run_both_at_once:        # serial: redo after ws (dm:153-154)
            strong = services.make_strong_job(
                job, self.strong_redo_rounds(job.window),
                getattr(job, "strong_label", f"strong({job.label})"))
            extra = Submission(strong)
        return OutcomeDirective(Directive.AWAIT_STRONG, extra=extra)

    def metrics(self) -> dict:
        return {"confidence_threshold": self.confidence_threshold,
                "run_both_at_once": self.run_both_at_once,
                "double_window": self.double_window}

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
        from .policies import Eager, Held
        from .schemes import SlidingWindowScheme

        if self.weak_keepup_ratio is not None and (
            type(scheme) is not SlidingWindowScheme
        ):
            raise ValueError(
                "weak_keepup_ratio implements the exact shipped serial "
                "sliding keep-up contract and requires SlidingWindowScheme"
            )
        if not self.double_window:
            if has_dynamic_streams and isinstance(boundary_policy, Eager):
                raise ValueError(
                    "Eager speculative recovery needs a statically planned "
                    "replay cone; dynamic streams create future windows at "
                    "runtime. Use Held boundaries for dynamic streams."
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
                "the strong result and deadlock the slab")
        if has_dynamic_streams or static_decode_plan_selected:
            raise ValueError(
                "double_window skips statically planned windows when a slab "
                "is assigned; stream windows created or folded at runtime "
                "(dynamic_streams/decode_ops) are not supported yet")
        if has_frontend:
            raise ValueError(
                "double_window is validated for explicit ops= workloads; "
                "frontend-built operation chains are not supported yet")

    def validate_operations(self, operations) -> None:
        if self.double_window and any(
            operation.predecessors or operation.has_successor
            for operation in operations
        ):
            raise ValueError(
                "double_window models one single-patch stream per operation "
                "(arXiv:2510.25222 Fig. 12); operation chains with "
                "predecessors/successors would let a slab cross an op seam "
                "where no far-boundary gate exists yet")

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
