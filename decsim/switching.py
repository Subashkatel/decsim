"""Decoder switching strategies (port 10): Baseline and Switching.

Part module: the strategy seam implementations (Contract 2c). Baseline is the
default. Switching escalates weak decodes to strong; the weak/strong
routing itself stays in the router (SwitchingRouter), and the pool owns unit
bookkeeping, hold-or-deliver, and cancellation mechanics.
"""

from __future__ import annotations

import math
from typing import Optional

from .message import RunSeedChild, RunSeedPathSegment
from .protocols import Directive, OutcomeDirective, Submission


class ThresholdRegister:
    """Per-code confidence thresholds, updatable at runtime (P17).

    The actuator half of a calibration loop: get(code) serves the
    router (falling back to `default`), set(code, g) updates a lane
    and records (seq, code, old, new) in `history` for audit. The
    loop that COMPUTES new thresholds is out of scope (row 11's
    remaining open item)."""

    def __init__(self, default: float, per_code: dict = None):
        self.default = float(default)
        self.per_code = dict(per_code or {})
        self._initial_default = self.default
        self._initial_per_code = dict(self.per_code)
        self.history: list = []
        self._seq = 0

    def run_manifest_config(self):
        return {
            "kind": "threshold_register",
            "initial_default": self._initial_default,
            "initial_per_code": dict(self._initial_per_code),
        }

    def get(self, code) -> float:
        return self.per_code.get(code, self.default)

    def set(self, code, threshold: float) -> None:
        old = self.get(code)
        self.per_code[code] = float(threshold)
        self._seq += 1
        self.history.append((self._seq, code, old, float(threshold)))


class Baseline:
    """Plain windowed decoding: submit the weak job, accept every outcome."""

    def run_manifest_config(self):
        return {"kind": "baseline"}

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

    Port of switching.py: confidence_threshold gates keep-weak (soft_output >=
    threshold); serial mode escalates after the ws hop; run_both_at_once starts
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
        self.confidence_threshold = confidence_threshold
        self.threshold_register = threshold_register   # P17 (None = scalar)
        self.run_both_at_once = run_both_at_once
        self.weak_keepup_ratio = weak_keepup_ratio
        self.bulk_strong = bulk_strong
        self.double_window = double_window

    def run_manifest_config(self):
        return {
            "kind": "switching",
            "confidence_threshold": self.confidence_threshold,
            "run_both_at_once": self.run_both_at_once,
            "weak_keepup_ratio": self.weak_keepup_ratio,
            "bulk_strong": self.bulk_strong,
            "double_window": self.double_window,
        }

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
                job, self.strong_redo_rounds(job.window),
                getattr(job, "strong_label", f"strong({job.label})"))
        elif not self.run_both_at_once:        # serial: redo after ws (dm:153-154)
            strong = services.make_strong_job(
                job, self.strong_redo_rounds(job.window),
                getattr(job, "strong_label", f"strong({job.label})"))
            extra = Submission(strong, delay_ticks=services.ws_delay())
        return OutcomeDirective(Directive.AWAIT_STRONG, extra=extra)

    def metrics(self) -> dict:
        return {"confidence_threshold": self.confidence_threshold,
                "run_both_at_once": self.run_both_at_once,
                "double_window": self.double_window}

    # ------------------------------------------------------------ validation

    def validate(self, spec, planning) -> None:
        """Reject RunSpec combinations that would deadlock, bypass the
        faithful start condition, or need skip semantics the runtime does
        not model yet."""
        if not self.double_window:
            from .policies import Eager
            boundary_policy = spec.boundary_policy
            if (spec.dynamic_streams
                    and (boundary_policy is None
                         or isinstance(boundary_policy, Eager))):
                raise ValueError(
                    "Eager speculative recovery needs a statically planned replay "
                    "cone; dynamic streams create future windows at runtime. Use "
                    "Held boundaries for dynamic streams.")
            return
        from .policies import Held
        if isinstance(spec.boundary_policy, Held):
            raise ValueError(
                "double_window requires the weak chain to keep committing "
                "(the far boundary IS the restart window's weak commit); "
                "the Held boundary policy would make later windows wait for "
                "the strong result and deadlock the slab")
        if hasattr(planning.scheme, "wire_deps"):
            raise ValueError(
                "double_window is defined for linearly-chained sliding "
                "windows (arXiv:2510.25222 Fig. 12); two-layer parallel "
                "window schemes are not supported")
        if spec.dynamic_streams or spec.decode_ops:
            raise ValueError(
                "double_window skips statically planned windows when a slab "
                "is assigned; stream windows created or folded at runtime "
                "(dynamic_streams/decode_ops) are not supported yet")
        if spec.frontend is not None:
            raise ValueError(
                "double_window is validated for explicit ops= workloads; "
                "frontend-built operation chains are not supported yet")
        if any(op.predecessors or op.has_successor for op in spec.ops or ()):
            raise ValueError(
                "double_window models one single-patch stream per operation "
                "(arXiv:2510.25222 Fig. 12); operation chains with "
                "predecessors/successors would let a slab cross an op seam "
                "where no far-boundary gate exists yet")

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
        return (result is not None and result.soft_output is not None
                and result.soft_output >= threshold)

    @staticmethod
    def strong_redo_rounds(window) -> int:
        """Rounds the strong decoder reprocesses: commit + 2*buffer."""
        commit = window.commit_hi - window.commit_lo + 1
        buffer = window.buffer_hi - window.commit_hi
        return commit + 2 * buffer

    def check_window_size(self, commit_rounds: int, buffer_rounds: int) -> None:
        """Raise if the weak decoder cannot keep up with this window size."""
        if self.weak_keepup_ratio is None:
            return
        ratio = self.weak_keepup_ratio
        weak_decode_rounds = ratio * (commit_rounds + buffer_rounds)
        if weak_decode_rounds > commit_rounds + 1e-9:
            needed = math.ceil(ratio / (1 - ratio) * buffer_rounds - 1e-9)
            raise ValueError(
                f"commit region of {commit_rounds} rounds too short for "
                f"weak_keepup_ratio={ratio} (needs >= {needed}); use a bigger "
                f"commit region or lower weak_keepup_ratio.")
