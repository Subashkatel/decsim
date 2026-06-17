from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .message import Window, Operation
    from .protocols import CodeModel, LayoutModel

# ===============================================================================
# SCHEMES
# This is the default decoding scheme. It answers two questions:
# 1. how an operation's syndrome rounds are grouped into commit and buffer windows
# 2. when a window has accumulated enough syndrome rounds to be decoded safely.
# It is a pure POLICY -- no engine, no clock -- so it is a clean swap point for
# other windowing schemes: adaptive windowing (ADaPT, arXiv:2605.01149), parallel
# A/B-layer windowing (arXiv:2511.10633 Sec II.4), speculative windowing, double-
# window decoder switching (arXiv:2510.25222 Sec III.3), and so on.
# ===============================================================================

class SlidingWindowScheme:
    """The standard SEQUENTIAL (forward) sliding-window decoder: each operation is chopped
    into windows that commit C rounds behind a B-round look-ahead buffer, where C and B come
    from the code (both default to the code distance d). If the buffer spills beyond the end
    of the operation's own rounds, the overflow comes from the successor operation's early
    rounds (or idle memory rounds). Once a window commits, error strings crossing into its
    buffer become "artificial defects" handed to the NEXT window as its boundary -- so the
    windows of one operation form a serial dependency chain.

    Grounding. This is the (W, C)-sliding window of the literature: window size W = C + B
    with the commit-then-carry-forward artificial-defect rule (ADaPT, arXiv:2605.01149
    Sec II-C; used on FPGA hardware for the gross code with W = d and runtime-configurable C,
    arXiv:2510.21600 Sec 4.1). It is NOT the parallel windowing of arXiv:2511.10633 Sec II.4,
    where 3d-round windows have buffer/commit/buffer sub-regions and alternate in two layers
    (independent layer-A windows decode concurrently, then layer-B windows consume their
    boundaries; memory reaction time gamma_mem = 6d*tau_d(d^2) + t_com, Eq. 13). The serial
    chain here is throughput-limited by one window's decode per C rounds; the parallel scheme
    exists precisely to break that chain -- see ParallelWindowScheme below, which implements
    it through the scheme's own wire_deps hook with no planner change."""

    # Human-readable decoding strategy, reported verbatim in the cluster's execution-plan log
    # line: a scheme describes ITSELF (ground truth), so the trace never has to infer
    # "is this windowed?" from the window count. Override in subclasses.
    scheme_label = "sliding-window (serial commit/buffer chain)"
    windowed = True            # decodes in sliding windows: commit + buffer, inter-window boundaries

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int]]:
        """Lay out the windows for an operation: commit C rounds behind a B-round look-ahead buffer."""
        import math
        C,B,R = code.commit_rounds(), code.buffer_rounds(), n_rounds
        nwin = max(1, math.ceil(R / C))
        plan = []
        for k in range(nwin):
            commit_lo = k * C + 1
            commit_hi = min((k + 1) * C, R)
            buffer_hi = commit_hi + B
            plan.append((commit_lo, commit_hi, buffer_hi))
        return plan
    
    def data_complete(self, window: "Window", rounds_arrived: int, successor_rounds: int,
                      memory_rounds: int, n_rounds: int, has_successor: bool,
                      op: "Operation" = None, layout: "LayoutModel" = None) -> bool:
        """Return True once the commit+buffer rounds have arrived (including spillover from successor or memory rounds if needed).
        This default is a purely temporal rule; op/layout are available for more complex schemes (ignored here)."""
        in_op_need = min(window.buffer_hi, n_rounds)       # commit + in-op buffer rounds
        if rounds_arrived < in_op_need:
            return False
        overflow = window.buffer_hi - n_rounds  # buffer rounds beyond the end of the operation
        if overflow > 0:
            if not has_successor:
                return True # no successor to provide overflow rounds, so just go with what we have
            return (successor_rounds >= overflow) or (memory_rounds >= overflow) # successor or memory rounds can provide the overflow
        return True


class NaiveOnlineScheme(SlidingWindowScheme):
    """The NAIVE online decoding of arXiv:2510.25222 Sec III.C (Fig 9): all of an
    operation's syndrome data is collected first and then decoded COLLECTIVELY, as one
    batch -- no sliding, no look-ahead buffer. This is the baseline the windowing
    schemes exist to beat: decoding cannot even start until the last round has arrived,
    so the reaction wait carries the full batch decode, and a too-slow decoder shows
    the paper's backlog growth at its starkest (their Fig 10/11 use exactly this
    scheme). One window per operation; cross-op dependencies still apply (the planner's
    DAG wiring is scheme-independent).

    Do NOT expect it to lose on a single operation's reaction time: one batch has no
    serial window chain, no t_dd boundary hops, and no buffer-spillover wait, so it can
    beat the sliding scheme there -- the paper itself prefers batch-without-buffer when
    affordable (Sec III.C), and d/d sliding windows violate its Eq. 7 once tau_dec >
    tau_gen/2. The naive scheme's real cost is across a STREAM: decode never overlaps
    data collection, which is what drives the Fig 10/11 backlog growth.

    Inherits data_complete: with buffer_hi = n_rounds there is no overflow, so the
    window is ready exactly when all its own rounds have arrived.

    `batches_idle_rounds_into_next_op`: under this scheme a batch is the whole
    feedback-to-feedback SEGMENT -- the rounds a patch idled before the gate plus the
    gate's own rounds (the r_i of Eq. 5; Terhal's backlog argument: the record
    generated while waiting "needs to have been processed" before the next feedback).
    The cluster reads this flag in prepend_idle_rounds; continuously-windowed schemes
    leave it False and decode idle stretches concurrently instead (see
    docs/DESIGN-idle-stream-windows.md for why merging is exactly what makes this
    scheme reproduce Eq. 5 and concurrent windows would not)."""

    batches_idle_rounds_into_next_op = True
    scheme_label = "naive online -- GLOBAL batch decode (no windowing)"
    windowed = False           # one batch decode of the whole operation: no windows, no boundaries

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int]]:
        """One batch window: commit every round, look ahead none."""
        return [(1, n_rounds, n_rounds)]


class ParallelWindowScheme(SlidingWindowScheme):
    """The PARALLEL (two-layer) windowing of arXiv:2511.10633 Sec II.4. Windows have "3d
    temporal size" with "three d-sized sub-regions: a buffer region, a commit region, and
    another buffer region"; layer-A windows "are separated by a gap of d rounds"; each
    layer-B window covers "the two buffer regions and the gap region between two commit
    regions from layer A". Layer-A windows have NO dependencies on each other, so "all
    tasks in a single layer can, in principle, be decoded in parallel" -- with enough
    decoder units the decode latency is two window decodes, gamma_mem = 6d*tau_d(d^2) +
    t_com (Eq. 13), instead of the sequential scheme's one-decode-per-commit-stride chain.
    After a layer-A decode, the error strings crossing into its buffers become artificial
    defects at the buffer boundaries -- the t_dd boundary message each B window waits for.

    Layout, generically (C = commit_rounds, B = buffer_rounds, gap = C; the paper uses
    C = B = gap = d). Period S = 2C + 2B per A window:
      A_k (k >= 0): commit [1 + kS, kS + C], leading buffer B (none for A_0 -- the stream
                    has no rounds before round 1), trailing buffer B.
      B_k: commits everything between A_k's and A_{k+1}'s commit regions (trailing buffer
           + gap + leading buffer = 2B + C rounds); its lookahead on both sides is the A
           windows' data, enforced by its dependencies rather than extra rounds.
      tail: if rounds remain after the last A's commit, one final window commits them with
            a B-round lookahead (spillover from the successor, as in the sequential
            scheme). The stream-end handling is this implementation's choice; the paper
            describes the steady state only.

    Windows are emitted interleaved in commit order [A_0, B_0, A_1, B_1, ...], so EVEN
    indices are layer A and ODD indices are layer B / the tail -- which wire_deps uses.

    Inherits data_complete (a window is decodable once its own rounds arrived, with
    successor/memory spillover for lookahead past the operation's end)."""

    scheme_label = "parallel A/B two-layer window (arXiv:2511.10633 Sec II.4)"

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int, int]]:
        """Lay out interleaved A/B windows: A commits every 2C+2B rounds, B commits the
        rounds in between, a tail window commits any stream-end remainder."""
        C, B, R = code.commit_rounds(), code.buffer_rounds(), n_rounds
        S = 2 * C + 2 * B
        a_windows = []                                     # (buffer_lo, commit_lo, commit_hi, buffer_hi)
        k = 0
        while 1 + k * S <= R:
            commit_lo = 1 + k * S
            commit_hi = min(commit_lo + C - 1, R)
            buffer_lo = max(1, commit_lo - B)              # A_0 has no leading rounds
            a_windows.append((buffer_lo, commit_lo, commit_hi, commit_hi + B))
            k += 1
        plan = []
        for i, a in enumerate(a_windows):
            plan.append(a)
            if i + 1 < len(a_windows):                     # B window between A_i and A_{i+1}
                lo, hi = a[2] + 1, a_windows[i + 1][1] - 1
                plan.append((lo, lo, hi, hi))              # pure commit; lookahead = A data
            elif a[2] < R:                                 # tail: commit the remainder
                lo = a[2] + 1
                plan.append((lo, lo, R, R + B))            # B-round lookahead (spillover)
        return plan

    def wire_deps(self, windows: list) -> None:
        """Layer-B windows (odd indices) depend on their neighbouring layer-A windows
        (boundary artificial defects from both sides); layer-A windows are independent."""
        for k in range(1, len(windows), 2):
            w = windows[k]
            w.deps.append((w.op_id, k - 1))                # A on the left
            if k + 1 < len(windows):
                w.deps.append((w.op_id, k + 1))            # A on the right (absent for tail)


class ThresholdSwitch:
    """The default switch decision for DoubleWindowScheme: escalate a window to the strong
    decoder when the weak decoder's soft output is below a fixed threshold g_th
    (arXiv:2510.25222 Sec III.A). This is the simplest possible policy; it is a separate object
    precisely so the decision can be changed without touching the scheme or the cluster -- swap
    in any other SwitchPolicy (a distance-dependent or adaptive threshold, a rule on the raw gap
    value, etc.) and the rest of the switching machinery is unchanged."""
    def __init__(self, g_th: float):
        """Escalate any window whose soft output is below g_th."""
        self.g_th = g_th

    def should_escalate(self, job, result) -> bool:
        """True when the weak result carries a soft output below the threshold."""
        return (result is not None and result.soft_output is not None
                and result.soft_output < self.g_th)


class DoubleWindowScheme(SlidingWindowScheme):
    """The double-window decoder-switching scheme of arXiv:2510.25222 (Sec III.C, Fig 12).

    A fast WEAK decoder runs ordinary sliding windows and emits a soft output (its confidence)
    each window. When that confidence is too low, the window is ALSO handed to a slow STRONG
    decoder over `strong_rounds = commit + 2*buffer` rounds. The weak decoder never waits for
    the strong one, so the weak stream never backs up (the scheme's defining property); only
    the strong decoder accumulates a backlog, which Theorem 1 bounds.

    Three pieces, each swappable on its own:
      - the WHEN: a SwitchPolicy (`switch_policy`); the default ThresholdSwitch(g_th) escalates
        on soft output < g_th. The cluster asks should_escalate(job, result) and never sees the
        rule itself.
      - the WHERE: decoders.SwitchingRouter sends escalated (hint="strong") jobs to the strong
        decoder; a {"default", "strong"} unit-pool split runs the two on separate units, so the
        strong pool's queue IS the backlog.
      - the LAYOUT and resume rule: inherited from SlidingWindowScheme unchanged (a continuous
        weak stream makes "resume after commit + buffer rounds" just the next window's turn).

    Pass `f_weak = tau_weak / tau_gen` to enable the Eq. 7 keep-up guard in plan_windows.

    Scope: the SERIAL variant (decode weak, then maybe strong), modelled at the timing/backlog
    level -- the whole of the paper's double-window analysis. The parallel-feed variant and
    real-decode refinement of the logical outcome are documented follow-ups."""

    scheme_label = "double-window (weak sliding + strong escalation, arXiv:2510.25222)"

    def __init__(self, g_th: float = None, switch_policy=None, f_weak: float = None):
        """Configure how switching is decided and, optionally, the Eq. 7 keep-up check.

        Pass exactly one of: `g_th` (the simple soft-output threshold, wrapped in a
        ThresholdSwitch) or `switch_policy` (any object with should_escalate(job, result) --
        see the SwitchPolicy protocol). `f_weak = tau_weak / tau_gen` enables the Eq. 7 guard."""
        if (g_th is None) == (switch_policy is None):
            raise ValueError("provide exactly one of g_th (a threshold) or switch_policy "
                             "(a custom SwitchPolicy)")
        self.switch_policy = switch_policy if switch_policy is not None else ThresholdSwitch(g_th)
        self.g_th = getattr(self.switch_policy, "g_th", None)   # exposed for logs / config
        if f_weak is not None and not 0 < f_weak < 1:
            raise ValueError(f"f_weak must be in (0, 1) -- the weak decoder must be faster "
                             f"than syndrome generation (got {f_weak})")
        self.f_weak = f_weak

    def should_escalate(self, job, result) -> bool:
        """Delegate the switch decision to the policy (the cluster calls this per weak window)."""
        return self.switch_policy.should_escalate(job, result)

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int]]:
        """The weak decoder's sliding-window layout, after checking the Eq. 7 keep-up bound."""
        self._require_keep_up(code.commit_rounds(), code.buffer_rounds())
        return super().plan_windows(op_id, n_rounds, code)

    def _require_keep_up(self, commit_rounds: int, buffer_rounds: int) -> None:
        """Raise if the commit region is too small for the weak decoder to keep pace with
        syndrome generation (Eq. 7 of arXiv:2510.25222). Skipped when f_weak is None."""
        import math
        if self.f_weak is None:
            return
        minimum = math.ceil(self.f_weak / (1 - self.f_weak) * buffer_rounds)
        if commit_rounds < minimum:
            raise ValueError(
                f"DoubleWindowScheme: a commit region of {commit_rounds} rounds is too small "
                f"for a weak decoder with tau_weak/tau_gen = {self.f_weak}. Eq. 7 of "
                f"arXiv:2510.25222 requires commit >= {minimum} rounds (for buffer = "
                f"{buffer_rounds}), or the weak decoder falls behind. Raise the commit rounds, "
                f"lower f_weak, or pass f_weak=None to skip this check.")

    def strong_rounds(self, window: "Window") -> int:
        """The escalated region size r_strong = commit + 2*buffer: the suspect commit region
        plus one buffer on each side (Fig 12 of arXiv:2510.25222)."""
        commit_rounds = window.commit_hi - window.commit_lo + 1
        buffer_rounds = window.buffer_hi - window.commit_hi
        return commit_rounds + 2 * buffer_rounds
