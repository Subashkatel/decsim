"""The windowing runtime: tracks each decode window from ready to committed.

When a window's rounds are all present it builds the decode job (submission
itself is DecodingStrategy.on_window_ready), hands the result to the
orchestrator, ships boundary data to dependent windows at commit (gated by
BoundaryPolicy.on_commit; Eager, the default, always ships), and declares
the op finished once every window is committed. The windows themselves are
laid out either up front by planner.WindowPlanner (ops of known length) or
at runtime by DynamicWindows (streams of unknown length); this hub runs
whatever both produce. Two helpers own adjacent state: PayloadStore (how
long raw syndrome rounds stay retained) and DynamicWindows (dynamic-stream
growth and sealing).

Event ordering here is load-bearing: same-tick events run FIFO, so the
frozen timing goldens pin the engine.schedule call order inside every
handler. Invariants to know before editing:
  - on_decode_done runs after the decode layer has set
    job.awaiting_strong_result, so the escalation check sees it.
  - a strong (redo) result only revises the op's logical accumulator;
    committed windows are never rolled back.
  - op delivery waits on the per-op pending-strong count and the stream
    seal; requires_strong_commit marks the op but never gates delivery.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Callable, Optional

from .message import (DecodeJob, DecodeResult, Operation, SyndromePayload, Window,
                    WindowPlan)
from .payload_store import PayloadStore
from .dynamic_windows import DynamicWindows


class WindowManager:
    """Own window state, readiness, commits, and boundary handoff."""

    def __init__(self, engine, *, scheme, layout, rounds_policy, code,
                 deadline_policy, links, orchestrator, boundary_policy,
                 syndrome_source=None, switching_active: bool = False,
                 store: Optional[PayloadStore] = None):
        self.engine = engine
        self.scheme = scheme
        self.layout = layout
        self.rounds_policy = rounds_policy
        self.code = code
        self.deadline_policy = deadline_policy
        self.links = links
        self.orchestrator = orchestrator
        self.boundary_policy = boundary_policy
        self.syndrome_source = syndrome_source
        #: retains rounds for possible strong re-decodes (today: switching set).
        self.switching_active = switching_active

        # Wired post-construction by the composition root:
        self.strategy = None
        self.services = None
        self.submit_fn: Optional[Callable] = None    # (job, delay_ticks) -> None
        self.needs_hyperedges = False
        self.on_workload_complete: Optional[Callable[[], None]] = None

        self.d = code.distance
        self.commit = code.commit_rounds()
        self.buffer = code.buffer_rounds()

        self.store = store if store is not None else PayloadStore()
        self.lifecycle = DynamicWindows(self)

        self.ops: dict[int, Operation] = {}
        self.rounds_arrived: dict[int, int] = {}
        self.memory_rounds: dict[int, int] = {}
        self.memory_rounds_total = 0

        self.windows: dict[tuple, Window] = {}
        self.op_windows: dict[int, list] = {}
        self.window_count: dict[int, int] = {}
        self.successors: dict[int, list] = {}
        self.committed_windows: set = set()
        self._committed_per_op: dict[int, int] = {}
        self.blocking_ops: set[int] = set()
        self.op_results: dict[int, int] = {}
        self._window_logical_values: dict[tuple, int] = {}
        self._committed_boundary_defects: dict[tuple, Optional[dict]] = {}
        self._held_boundary: dict[tuple, tuple] = {}      # Held policy deferrals
        self._pending_strong_windows: set[tuple] = set()
        self._pending_strong_per_op: dict[int, int] = {}
        #: faithful double-window escalations waiting for their far-side weak
        #: boundary; slab key -> pending record (see defer_strong_escalation)
        self._deferred_strong: dict[tuple, dict] = {}
        self._deferred_by_far: dict[tuple, tuple] = {}   # restart key -> slab key
        self._deferred_terminal: dict[int, tuple] = {}   # op id -> slab key
        self.absorbed_windows: set[tuple] = set()        # skipped by the weak chain
        self.op_strong_commit_time: dict[int, int] = {}
        self._finalize_gates: dict[int, Callable] = {}    # gate_finalize seam
        self._finished_ops: set[int] = set()
        self._workload_complete_sent = False
        self.window_models: dict = {}
        self.total_windows = 0
        self._windows_built = False
        self._plan_spatial = None

    # ---------------------------------------------------------- registration

    def register_op(self, op: Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        if op.id not in self.ops:
            self.rounds_arrived[op.id] = 0
            self.memory_rounds[op.id] = 0
            self.store.register_op(op.id)
        self.ops[op.id] = op
        if op.blocked_by is not None:
            self.blocking_ops.add(op.blocked_by)

    def register_dynamic_stream(self, stream_op: Operation, code) -> None:
        """Register a stream whose windows are created at runtime."""
        stream_id = stream_op.id
        self.ops[stream_id] = stream_op
        self.rounds_arrived.setdefault(stream_id, 0)
        self.memory_rounds.setdefault(stream_id, 0)
        self.store.register_op(stream_id)
        self.window_count[stream_id] = 0
        self.op_windows[stream_id] = []
        self.successors.setdefault(stream_id, [])
        source_round_limit = None
        if self.syndrome_source is not None:
            source_round_limit = self.syndrome_source.register_dynamic_stream(
                stream_op, self.rounds_for(stream_op),
                belief_matching=self.needs_hyperedges)
        self.lifecycle.register(stream_op, code, source_round_limit)

    def rounds_for(self, op: Operation) -> int:
        """Rounds this operation runs for under its code/layout."""
        return self.rounds_policy.rounds_for(op, self.layout.code_for_op(op))

    def _spatial_nodes(self, op: Operation) -> int:
        if self._plan_spatial is not None and op.id in self._plan_spatial:
            return self._plan_spatial[op.id]
        return self.layout.spatial_nodes_for(op)

    # ----------------------------------------------------------------- plan

    def load_execution_plan(self, plan: WindowPlan) -> None:
        """Install the pre-computed compile-time window plan."""
        if self._windows_built:
            return
        self._windows_built = True
        self.windows = plan.windows
        self.window_count = plan.window_count
        self.op_windows = plan.op_windows
        self.successors = plan.successors
        self._plan_spatial = plan.spatial_nodes
        self.total_windows = plan.total_windows
        self._build_window_error_models()
        self._build_round_leases()

    def _build_window_error_models(self) -> None:
        """Ask the syndrome source for per-window detector error models."""
        if self.syndrome_source is None:
            return
        for op_id, op in self.ops.items():
            keys = [(op_id, k) for k in self.op_windows.get(op_id, [])]
            wins = [self.windows[key] for key in keys]
            if not wins:
                continue
            models = self.syndrome_source.window_models_for_operation(
                op, wins, self.rounds_for(op),
                belief_matching=self.needs_hyperedges)
            if not models:
                continue
            for key, model in zip(keys, models):
                self.window_models[key] = model

    # ----------------------------------------------------------- read leases

    def _build_round_leases(self) -> None:
        for key, window in self.windows.items():
            self._add_window_read_refs(key, window)

    def _add_window_read_refs(self, key: tuple, window: Window) -> None:
        """Retain rounds needed by the weak window and possible strong re-decode."""
        weak = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window)
        strong = self._strong_context_read_keys(window, weak)
        self.store.lease(key, weak)
        self.store.lease((key, "strong"), strong)

    def _read_keys_for_bounds(self, op_id, start_round: int, buffer_hi: int,
                              window: Optional[Window] = None) -> list:
        """Retained payload round keys for a possibly cross-operation range."""
        operation_rounds = self._effective_round_count_for_window(op_id, window)
        reads = [(op_id, r)
                 for r in range(start_round, min(buffer_hi, operation_rounds) + 1)]
        overflow = buffer_hi - operation_rounds
        if overflow <= 0:
            return reads
        for successor_id in self.successors.get(op_id, []):
            reads += [(successor_id, r) for r in range(1, overflow + 1)]
        return reads

    def _strong_context_read_keys(self, window: Window, weak_reads: list) -> list:
        """Rounds retained until we know whether the strong decoder needs them."""
        if not self.switching_active:
            return []
        buffer_lo, _cl, _ch, buffer_hi = self._strong_context_bounds(window)
        weak = set(weak_reads)
        strong = self._read_keys_for_bounds(
            window.op_id, buffer_lo, buffer_hi, window)
        return [rk for rk in strong if rk not in weak]

    def _replace_window_read_refs(self, key: tuple, window: Window) -> None:
        """Replace retained-round references for an unqueued window.

        A closed boundary shrinks the weak set, so rounds can MOVE from the
        weak lease to the strong lease; a temporary guard lease keeps them
        referenced across the two per-lease replacements (the original did
        one merged old-vs-new diff, which never dropped a moved round)."""
        weak = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window)
        strong = self._strong_context_read_keys(window, weak)
        self.store.lease((key, "replace-guard"), weak + strong)
        self.store.replace(key, weak)
        self.store.replace((key, "strong"), strong)
        self.store.release((key, "replace-guard"))

    def refresh_unqueued_stream_windows(self, stream_id) -> None:
        """Refresh retained reads after a stream boundary closes future context."""
        for window_index in self.op_windows.get(stream_id, []):
            key = (stream_id, window_index)
            window = self.windows[key]
            if window.queued or window.committed:
                continue
            self._replace_window_read_refs(key, window)

    def reset_dynamic_window_reads(self, stream_id, window_index: int,
                                   window: Window) -> None:
        """Refresh read references after a live stream tail is clipped:
        weak reads only, over the window's commit range."""
        key = (stream_id, window_index)
        new_reads = sorted(
            (stream_id, r)
            for r in range(window.start_round, window.commit_hi + 1))
        self.store.replace(key, new_reads)

    # ------------------------------------------------------- dynamic windows

    def create_dynamic_window(self, stream_id, window_index, commit_lo,
                              commit_hi, buffer_hi, *, is_last) -> None:
        """Create one dynamic-stream window and wire it into the live plan.
        If the previous window already committed and shipped its boundary,
        the defects fold in right here — no dependency wait, no delay event."""
        buffer_lo = commit_lo
        window = Window(op_id=stream_id, k=window_index, commit_lo=commit_lo,
                        commit_hi=commit_hi, buffer_hi=buffer_hi,
                        n_rounds=buffer_hi - buffer_lo + 1, buffer_lo=buffer_lo)
        if window_index > 0:
            previous_key = (stream_id, window_index - 1)
            if (previous_key in self.committed_windows
                    and previous_key not in self._held_boundary):
                # boundary already shipped (a held one is NOT available yet)
                self._store_boundary(
                    window, stream_id,
                    self._committed_boundary_defects.get(previous_key))
            else:
                window.deps.append(previous_key)
                window.deps_remaining = 1
                self.windows[previous_key].dependents.append(
                    (stream_id, window_index))
        self.windows[(stream_id, window_index)] = window
        self.op_windows[stream_id].append(window_index)
        self.window_count[stream_id] += 1
        self.total_windows += 1
        if self.syndrome_source is not None:
            model = self.syndrome_source.window_model_for_stream(
                stream_id, window, is_last=is_last)
            if model is not None:
                self.window_models[(stream_id, window_index)] = model
        self._add_window_read_refs((stream_id, window_index), window)
        self.check_window((stream_id, window_index))

    def validate_stream_length(self, stream_id, stream_round_count: int) -> None:
        if self.syndrome_source is None:
            return
        self.syndrome_source.validate_stream_length(
            self.ops[stream_id], stream_round_count)

    # -------------------------------------------------------------- arrivals

    def on_syndrome_arrival(self, payload: SyndromePayload) -> None:
        """Store an arriving syndrome round and re-check affected windows."""
        op = self.ops[payload.operation_id]
        self._store_payload(payload, op)
        self.lifecycle.maybe_update(op.id)
        self._check_deferred_strong_after_arrival(op.id)
        self.check_windows_for_operation(op.id)
        for predecessor_id in op.predecessors:
            self.check_windows_for_operation(predecessor_id)

    def _store_payload(self, payload: SyndromePayload, op: Operation) -> None:
        if not self.store.has_op(op.id):
            raise RuntimeError(
                f"round {payload.round_index} of {op.name} arrived after the op's "
                f"last window committed and its syndrome RAM was freed. The device "
                f"emitted more rounds than the execution plan expects.")
        self.store.store(op.id, payload.round_index, payload,
                         fragment_index=payload.patch_id)
        fragments = self.store.fragments(op.id, payload.round_index)
        if len(fragments) >= payload.n_fragments:
            self.rounds_arrived[op.id] = max(self.rounds_arrived[op.id],
                                             payload.round_index)
        self.engine.log("DecoderCluster",
                        f"round {payload.round_index} of {op.name} arrived "
                        f"(op now has rounds 1..{self.rounds_arrived[op.id]})")

    def on_memory_round(self, op_id: int) -> None:
        """Record an idle/memory round and re-check waiting windows."""
        self.memory_rounds[op_id] += 1
        self.memory_rounds_total += 1
        self.engine.log("DecoderCluster",
                        f"memory round for {self.ops[op_id].name} "
                        f"(idle buffer rounds: {self.memory_rounds[op_id]})")
        for k in range(self.window_count[op_id]):
            self.check_window((op_id, k))

    def prepend_idle_rounds(self, op_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style op when the scheme asks."""
        if round_count <= 0 or not getattr(
                self.scheme, "batches_idle_rounds_into_next_op", False):
            return
        w = self.windows[(op_id, 0)]
        w.n_rounds += round_count

    # ------------------------------------------------------------- readiness

    def check_windows_for_operation(self, op_id: int) -> None:
        for window_index in range(self.window_count[op_id]):
            self.check_window((op_id, window_index))

    def check_window(self, key: tuple) -> None:
        """If a window has its data and dependencies, submit via the strategy."""
        window = self.windows[key]
        if window.queued or window.committed:
            return
        if (window.t_first_round is None
                and self.rounds_arrived[window.op_id] >= window.start_round):
            window.t_first_round = self.engine.now
        if not self._window_data_complete(window):
            return
        if window.t_data_complete is None:
            window.t_data_complete = self.engine.now
        op = self.ops[window.op_id]
        if window.deps_remaining > 0:
            if not window.blocked_logged:
                window.blocked_logged = True
            return
        self._submit_window_decode(key, window, op)

    def _window_data_complete(self, w: Window) -> bool:
        op = self.ops[w.op_id]
        succ_rounds = self._successor_rounds_available(w)
        round_count = self._effective_round_count_for_window(w.op_id, w)
        has_successor = op.has_successor and not self._window_has_closed_boundary(w)
        return self.scheme.data_complete(
            w, rounds_arrived=self.rounds_arrived[w.op_id],
            successor_rounds=succ_rounds, memory_rounds=self.memory_rounds[w.op_id],
            round_count=round_count, has_successor=has_successor,
            op=op, layout=self.layout)

    @property
    def _windowed(self) -> bool:
        return getattr(self.scheme, "windowed", True)

    def _job_desc(self, w: Window, op: Operation) -> str:
        """Human decode-job label."""
        if self._windowed:
            return f"{op.name} W{w.k} [commit {w.commit_lo}-{w.commit_hi}]"
        body_rounds = self._round_count_for_window(op.id, w)
        idle_rounds = max(0, w.n_rounds - body_rounds)
        if idle_rounds:
            return (f"{op.name} [whole op, {w.n_rounds} rounds: "
                    f"{idle_rounds} idle + {body_rounds} body]")
        return f"{op.name} [whole op, {w.n_rounds} rounds]"

    def _round_count_for_window(self, op_id, window: Optional[Window] = None) -> int:
        return self.lifecycle.round_count_for_window(
            op_id, window,
            fallback_rounds=self.rounds_for(self.ops[op_id])
            if not self.lifecycle.has(op_id) else 0)

    def _effective_round_count_for_window(self, op_id,
                                          window: Optional[Window]) -> int:
        round_count = self._round_count_for_window(op_id, window)
        if window is None:
            return round_count
        closed = self._closed_boundary_round_for_window(window)
        if closed is None:
            return round_count
        return min(round_count, closed)

    def _window_has_closed_boundary(self, window: Window) -> bool:
        return self._closed_boundary_round_for_window(window) is not None

    def _closed_boundary_round_for_window(self, window: Window) -> Optional[int]:
        stream_boundary = self.lifecycle.closed_boundary_for_window(window)
        if stream_boundary is not None:
            return stream_boundary
        op = self.ops[window.op_id]
        if op.feedback_boundary_mode != "measurement_closed":
            return None
        if op.id not in self.blocking_ops:
            return None
        round_count = self._round_count_for_window(window.op_id, window)
        if window.commit_hi <= round_count < window.buffer_hi:
            return round_count
        return None

    def _successor_rounds_available(self, window: Window) -> int:
        successor_ids = self.successors[window.op_id]
        successor_rounds = max((self.rounds_arrived[s] for s in successor_ids),
                               default=0)
        overflow = window.buffer_hi - self._round_count_for_window(
            window.op_id, window)
        if overflow <= 0 or not successor_ids or successor_rounds >= overflow:
            return successor_rounds
        successors_exhausted = all(
            self.rounds_arrived[s] >= self._round_count_for_window(s)
            for s in successor_ids)
        return overflow if successors_exhausted else successor_rounds

    # ------------------------------------------------------------ job build

    def _submit_window_decode(self, key: tuple, window: Window,
                              op: Operation) -> None:
        """Build the weak job, ask the strategy, and enqueue its submissions."""
        window.t_queued = self.engine.now
        deadline = self.deadline_policy.deadline(
            op, window, self.engine.now,
            on_reaction_path=(op.id in self.blocking_ops))
        job = DecodeJob(op_id=window.op_id, window_id=window.k,
                        n_rounds=window.n_rounds,
                        ready_time=self.engine.now, deadline=deadline,
                        spatial_nodes=self._spatial_nodes(op),
                        payloads=self._assemble_payloads(window),
                        dem=self.window_models.get(key),
                        code=self.layout.code_for_op(op).name,
                        window=window, label=self._job_desc(window, op))
        job.strong_label = f"strong({op.name} W{window.k})"
        window.queued = True
        for submission in self.strategy.on_window_ready(window, job,
                                                        self.services):
            self.submit_fn(submission.job, submission.delay_ticks)

    # ------------------------------------------------------- payload assembly

    @staticmethod
    def _xor_mask(previous_mask, incoming_mask) -> list:
        previous_bits = [int(b) for b in previous_mask] \
            if previous_mask is not None else []
        incoming_bits = [int(b) for b in incoming_mask]
        if len(previous_bits) < len(incoming_bits):
            previous_bits += [0] * (len(incoming_bits) - len(previous_bits))
        for i, bit in enumerate(incoming_bits):
            previous_bits[i] ^= bit
        return previous_bits

    def _apply_boundary(self, w: Window, payload: SyndromePayload,
                        round_key: Optional[int] = None) -> SyndromePayload:
        """Return a payload copy with received artificial defects XORed in."""
        r = payload.round_index if round_key is None else round_key
        mask = w.boundary_in.get((r, payload.patch_id), w.boundary_in.get(r))
        if mask is None:
            return payload
        bits = [int(m) for m in mask] if payload.bits is None \
            else self._xor_mask(payload.bits, mask)
        return dc_replace(payload, bits=bits)

    def _assemble_payloads(self, w: Window) -> list:
        """Collect this window's payloads, including successor overflow rounds."""
        operation_rounds = self._effective_round_count_for_window(w.op_id, w)
        end_round = min(w.buffer_hi, operation_rounds)
        payloads = []
        for round_index in range(w.start_round, end_round + 1):
            frags = self.store.fragments(w.op_id, round_index)
            if frags is not None:
                payloads += [self._apply_boundary(w, frags[patch_id])
                             for patch_id in sorted(frags)]
        overflow = w.buffer_hi - operation_rounds
        if overflow > 0:
            for successor_id in self.successors.get(w.op_id, []):
                for round_index in range(1, overflow + 1):
                    frags = self.store.fragments(successor_id, round_index)
                    if frags is not None:
                        payloads += [
                            self._apply_boundary(
                                w, frags[patch_id],
                                round_key=operation_rounds + round_index)
                            for patch_id in sorted(frags)]
        return payloads

    # ------------------------------------------------------------ strong jobs

    def make_strong_decode_job(self, weak_job: DecodeJob, round_count: int,
                               label: str) -> DecodeJob:
        """Build the two-sided strong re-decode job for an escalated window."""
        key = (weak_job.op_id, weak_job.window_id)
        weak_window = self.windows[key]
        op = self.ops[weak_job.op_id]
        strong_window = self._strong_context_window(weak_window)
        dem = None
        if self.syndrome_source is not None:
            dem = self.syndrome_source.strong_window_model_for_operation(
                op, strong_window,
                self._round_count_for_window(op.id, strong_window),
                belief_matching=self.needs_hyperedges)
        return DecodeJob(
            op_id=weak_job.op_id, window_id=weak_job.window_id,
            n_rounds=round_count, ready_time=self.engine.now,
            deadline=self.engine.now, label=label, hint="strong",
            spatial_nodes=weak_job.spatial_nodes, code=weak_job.code,
            dem=dem, payloads=self._assemble_payloads(strong_window),
            attempt=1, window=strong_window, strong_decode_for=key)

    def _strong_context_window(self, weak_window: Window) -> Window:
        buffer_lo, commit_lo, commit_hi, buffer_hi = \
            self._strong_context_bounds(weak_window)
        strong_window = Window(
            op_id=weak_window.op_id, k=weak_window.k, commit_lo=commit_lo,
            commit_hi=commit_hi, buffer_hi=buffer_hi, buffer_lo=buffer_lo,
            n_rounds=buffer_hi - buffer_lo + 1)
        strong_window.boundary_in = dict(weak_window.boundary_in)
        return strong_window

    @staticmethod
    def _strong_context_bounds(window: Window) -> tuple:
        buffer_rounds = max(0, window.buffer_hi - window.commit_hi)
        buffer_lo = max(1, window.commit_lo - buffer_rounds)
        buffer_hi = window.commit_hi + buffer_rounds
        return buffer_lo, window.commit_lo, window.commit_hi, buffer_hi

    # ------------------------------------------- faithful double window (III C)
    #
    # arXiv:2510.25222 Fig. 12: on a switching event the slab of
    # r_strong = r_com + 2*r_buf rounds starts AT the suspicious commit and
    # extends FORWARD; the weak chain skips the windows whose commit regions
    # the slab absorbs and restarts on the first window past the slab; the
    # strong decoder commits the entire slab, and it may start only once the
    # weak decoder has determined the boundary conditions at both slab ends
    # (left: the escalated window's own entry defects; right: the restart
    # window's weak commit, or the terminal boundary at the stream end).

    def defer_strong_escalation(self, weak_job: DecodeJob, n_rounds: int,
                                label: str) -> None:
        """Register a switching event: lay out the forward slab, absorb the
        weak windows it covers (they are never weak-decoded), and hold the
        strong job until the restart window's weak commit fixes the far-side
        boundary (state waiting_far_boundary), or, for a terminal slab, until
        every clamped slab round has been stored (waiting_terminal_data).
        Exactly one strong job per escalation (duplicates raise)."""
        key = (weak_job.op_id, weak_job.window_id)
        if key in self._deferred_strong:
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one switching "
                f"event creates exactly one strong job")
        weak_window = self.windows[key]
        op_id, escalated_index = key
        round_count = self._round_count_for_window(op_id, weak_window)
        slab_lo = weak_window.commit_lo
        slab_hi = min(slab_lo + n_rounds - 1, round_count)
        trailing_rounds = max(0, weak_window.buffer_hi - weak_window.commit_hi)
        context_hi = min(slab_hi + trailing_rounds, round_count)

        absorbed, restart_key = [], None
        for window_index in self.op_windows[op_id]:
            if window_index <= escalated_index:
                continue
            later = self.windows[(op_id, window_index)]
            if later.commit_hi <= slab_hi:
                absorbed.append((op_id, window_index))
                continue
            if later.commit_lo <= slab_hi:
                raise RuntimeError(
                    f"window ({op_id},{window_index}) commits "
                    f"{later.commit_lo}-{later.commit_hi} across the slab "
                    f"edge {slab_hi}; double_window needs commit regions "
                    f"that tile the slab (commit == buffer layouts do)")
            restart_key = (op_id, window_index)
            break

        self._deferred_strong[key] = {
            "weak_job": weak_job, "label": label, "slab_lo": slab_lo,
            "slab_hi": slab_hi, "context_hi": context_hi,
            "restart_key": restart_key, "state": "waiting_far_boundary"}
        # The standing strong lease held only the legacy side-context rounds;
        # the deferred slab is assembled after later weak commits release
        # their leases, so it must hold every slab + context round until then.
        self.store.replace((key, "strong"),
                           [(op_id, r) for r in range(slab_lo, context_hi + 1)])
        for absorbed_key in absorbed:
            self._absorb_window(absorbed_key, restart_key)
        self.engine.log("DecoderCluster",
                        f"{label}: slab rounds {slab_lo}-{slab_hi} assigned; "
                        f"weak chain skips {len(absorbed)} window(s); strong "
                        f"start deferred until the far-side weak boundary")
        if restart_key is None:
            # Terminal slab: the far side is the stream end, but Fig. 12's
            # blocks are STORED data; a clamped slab may still be waiting
            # for its final rounds to be generated.
            if self.rounds_arrived[op_id] >= context_hi:
                self._submit_deferred_strong(key)
            else:
                self._deferred_strong[key]["state"] = "waiting_terminal_data"
                self._deferred_terminal[op_id] = key
        else:
            self._deferred_by_far[restart_key] = key
            self.check_window(restart_key)      # its absorbed dep is gone

    def _absorb_window(self, key: tuple, restart_key: Optional[tuple]) -> None:
        """A window whose commit region the slab covers is never decoded by
        the weak chain (Fig. 12 panel 5): it counts as committed with no
        logical contribution, and the restart window stops waiting for it."""
        window = self.windows[key]
        if window.queued or window.committed:
            raise RuntimeError(f"cannot absorb window {key}: already "
                               f"{'queued' if window.queued else 'committed'}")
        window.queued = True                  # keeps check_window() away
        window.committed = True
        self.committed_windows.add(key)
        self._committed_per_op[key[0]] = self._committed_per_op.get(key[0], 0) + 1
        self.absorbed_windows.add(key)
        if restart_key is not None:
            restart = self.windows[restart_key]
            if key in restart.deps:
                restart.deps.remove(key)
                restart.deps_remaining -= 1
            if restart_key in window.dependents:
                window.dependents.remove(restart_key)
        self.store.release(key)
        self.store.release((key, "strong"))
        self.engine.log("DecoderCluster",
                        f"window {key} absorbed into the strong slab "
                        f"(weak chain skips it)")

    def _submit_deferred_strong(self, key: tuple) -> None:
        """Both slab boundaries are now weak-determined: build the slab job
        and submit it. The slab owns (commits) all r_strong rounds; the
        trailing rounds up to context_hi are read-only context, the same
        role a buffer plays for every weak window."""
        pending = self._deferred_strong.pop(key)
        weak_job = pending["weak_job"]
        weak_window = self.windows[key]
        op = self.ops[key[0]]
        slab = Window(op_id=key[0], k=key[1],
                      commit_lo=pending["slab_lo"],
                      commit_hi=pending["slab_hi"],
                      buffer_hi=pending["context_hi"],
                      buffer_lo=pending["slab_lo"],
                      n_rounds=pending["context_hi"] - pending["slab_lo"] + 1)
        # Left face: the slab starts at the escalated commit, so its entry
        # condition is exactly what the escalated window received.
        slab.boundary_in = dict(weak_window.boundary_in)
        dem = None
        if self.syndrome_source is not None:
            dem = self.syndrome_source.strong_window_model_for_operation(
                op, slab, self._round_count_for_window(op.id, weak_window),
                belief_matching=self.needs_hyperedges)
        payloads = self._assemble_payloads(slab)
        covered = {payload.round_index for payload in payloads}
        needed = set(range(pending["slab_lo"], pending["context_hi"] + 1))
        if covered != needed:
            raise RuntimeError(
                f"{pending['label']}: slab submitted with rounds "
                f"{sorted(covered)} but it needs "
                f"{pending['slab_lo']}-{pending['context_hi']}; a slab may "
                f"only start once every stored block exists (Fig. 12)")
        strong_job = DecodeJob(
            op_id=key[0], window_id=key[1],
            n_rounds=pending["slab_hi"] - pending["slab_lo"] + 1,
            ready_time=self.engine.now, deadline=self.engine.now,
            label=pending["label"], hint="strong",
            spatial_nodes=weak_job.spatial_nodes, code=weak_job.code,
            dem=dem, payloads=payloads,
            attempt=1, window=slab, strong_decode_for=key)
        self.services.check_strong_route(weak_job, strong_job)
        self.store.release((key, "strong"))   # slab captured into the job
        self.engine.log("DecoderCluster",
                        f"{pending['label']}: far-side weak boundary "
                        f"determined -> strong slab submitted")
        self.submit_fn(strong_job, self.services.ws_delay())

    def _check_deferred_strong_after_commit(self, committed_key: tuple) -> None:
        """The restart window's weak commit is the far boundary of a slab."""
        slab_key = self._deferred_by_far.pop(committed_key, None)
        if slab_key is not None:
            self._submit_deferred_strong(slab_key)

    def _check_deferred_strong_after_arrival(self, op_id) -> None:
        """A terminal slab waits for its clamped tail rounds to be stored."""
        slab_key = self._deferred_terminal.get(op_id)
        if slab_key is None:
            return
        if self.rounds_arrived[op_id] >= self._deferred_strong[slab_key]["context_hi"]:
            del self._deferred_terminal[op_id]
            self._submit_deferred_strong(slab_key)

    @property
    def pending_escalations(self) -> dict:
        """Deferral state per escalated window key, for tests and metrics
        (entries exist only while waiting for the far boundary)."""
        return {key: record["state"]
                for key, record in self._deferred_strong.items()}

    # ---------------------------------------------------------------- commit

    def on_decode_done(self, job: DecodeJob, res: DecodeResult) -> None:
        """Commit a finished window decode. The step order below is frozen
        by the timing goldens — do not reorder."""
        key = (job.op_id, job.window_id)
        window = self.windows[key]
        op = self.ops[job.op_id]
        self._commit_window(job, res, key, window, op)
        self.lifecycle.update_committed_round_count(op.id)
        defects = res.boundary_defects if res is not None else None
        final = not job.awaiting_strong_result
        if self.boundary_policy.on_commit(window, final=final):
            self._send_boundary_defects(window, op, defects)
        else:
            self._held_boundary[key] = (op.id, defects)      # Held: ship at final
        self.store.release(key)
        self._check_deferred_strong_after_commit(key)
        self._finish_operation_if_ready(op)
        self.finish_workload_if_ready()

    def _commit_window(self, job: DecodeJob, res: DecodeResult, key: tuple,
                       window: Window, op: Operation) -> None:
        window.committed = True
        window.t_done = self.engine.now
        self.committed_windows.add(key)
        self._committed_per_op[op.id] = self._committed_per_op.get(op.id, 0) + 1
        self.engine.log("DecoderCluster",
                        f"DECODE DONE {op.name} W{window.k} "
                        f"[commit {window.commit_lo}-{window.commit_hi}]")
        if res is not None and res.logical_value is not None:
            self._replace_window_logical_value(key, op.id, int(res.logical_value))
        if job.awaiting_strong_result:
            self._mark_window_waiting_for_strong(key, op.id)
        if key not in self._deferred_strong:
            # a deferred slab still needs these rounds; released at submission
            self.store.release((key, "strong"))

    def _replace_window_logical_value(self, key: tuple, op_id: int,
                                      value: int) -> None:
        previous = self._window_logical_values.get(key)
        if previous is not None:
            self.op_results[op_id] = self.op_results.get(op_id, 0) ^ previous
        self._window_logical_values[key] = value
        self.op_results[op_id] = self.op_results.get(op_id, 0) ^ value

    def _mark_window_waiting_for_strong(self, key: tuple, op_id: int) -> None:
        if key in self._pending_strong_windows:
            return
        self._pending_strong_windows.add(key)
        self._pending_strong_per_op[op_id] = \
            self._pending_strong_per_op.get(op_id, 0) + 1

    def on_strong_decode_done(self, key: tuple, result: DecodeResult) -> None:
        """Finalize a weak-committed window with the strong logical result:
        only the logical value is revised; dependents are never touched."""
        window = self.windows[key]
        op = self.ops[window.op_id]
        if result is not None and result.logical_value is not None:
            self._replace_window_logical_value(key, op.id,
                                               int(result.logical_value))
        self.op_strong_commit_time[op.id] = max(
            self.op_strong_commit_time.get(op.id, 0), self.engine.now)
        self._resolve_strong_wait(key, op.id)
        if key in self._held_boundary:                        # Held: ship now
            src_op_id, defects = self._held_boundary.pop(key)
            strong_defects = result.boundary_defects if result is not None else None
            self._send_boundary_defects(
                window, self.ops[src_op_id],
                strong_defects if strong_defects is not None else defects)
        self.release_stream_segments_at_commit(
            op.id, self.lifecycle.committed_round_counts.get(op.id, 0))
        self._finish_operation_if_ready(op)
        self.finish_workload_if_ready()

    def _resolve_strong_wait(self, key: tuple, op_id: int) -> None:
        if key not in self._pending_strong_windows:
            return
        self._pending_strong_windows.remove(key)
        remaining = self._pending_strong_per_op.get(op_id, 0) - 1
        if remaining > 0:
            self._pending_strong_per_op[op_id] = remaining
        else:
            self._pending_strong_per_op.pop(op_id, None)

    # -------------------------------------------------------------- handoff

    def _send_boundary_defects(self, window: Window, op: Operation,
                               defects: Optional[dict]) -> None:
        """Send this window's artificial defects to dependent windows (+t_dd)."""
        self._committed_boundary_defects[(window.op_id, window.k)] = defects
        for dep_key in window.dependents:
            self.engine.schedule(
                self.links.dd.cost(),
                lambda dk=dep_key, so=op.id, bd=defects:
                    self._receive_boundary(dk, so, bd),
                label=f"defects {op.name}W{window.k}->W{dep_key}")

    def _store_boundary(self, w: Window, src_op_id: int,
                        defects: Optional[dict]) -> None:
        """Fold artificial defects into a dependent window's round numbering:
        rounds shift into the dependent's frame, and defects that land below
        round 1 are dropped."""
        if not defects:
            return
        shift = 0 if src_op_id == w.op_id \
            else -self.rounds_for(self.ops[src_op_id])
        for key, mask in defects.items():
            r, patch = key if isinstance(key, tuple) else (key, None)
            r += shift
            if r < 1:
                continue
            dst_key = (r, patch) if patch is not None else r
            w.boundary_in[dst_key] = self._xor_mask(w.boundary_in.get(dst_key),
                                                    mask)

    def _receive_boundary(self, key: tuple, src_op_id: int,
                          defects: Optional[dict] = None) -> None:
        w = self.windows[key]
        self._store_boundary(w, src_op_id, defects)
        w.deps_remaining -= 1
        self.check_window(key)

    # --------------------------------------------------------------- finish

    def gate_finalize(self, op_id, predicate: Callable) -> None:
        """Hold this op's result publication (and so its non-Clifford
        feed-forward) until predicate(op) is true, on top of the built-in
        pending-strong gate. Used by the Switching and Speculation parts.
        The check re-runs at every commit and strong completion; a part
        whose predicate flips outside those events calls recheck_finalize()."""
        self._finalize_gates[op_id] = predicate

    def recheck_finalize(self, op_id) -> None:
        """Re-evaluate a gated op's finish check after its predicate flips."""
        op = self.ops.get(op_id)
        if op is not None:
            self._finish_operation_if_ready(op)

    def _finish_operation_if_ready(self, op: Operation) -> None:
        """Deliver an op result once every window is committed, no strong
        redo is pending, and the stream is sealed; delivery costs t_do."""
        if op.id in self._finished_ops:
            return
        if self._pending_strong_per_op.get(op.id, 0) > 0:
            return
        predicate = self._finalize_gates.get(op.id)
        if predicate is not None and not predicate(op):
            return
        if (self._committed_per_op.get(op.id, 0) == self.window_count[op.id]
                and self.lifecycle.sealed(op.id)):
            self._finished_ops.add(op.id)
            self.engine.schedule(self.links.do.cost(),
                                 lambda: self._deliver_result(op),
                                 label=f"result->orch({op.name})")
            self.store.free_op(op.id)

    def finish_workload_if_ready(self) -> None:
        if self._workload_complete_sent:
            return
        if (len(self.committed_windows) == self.total_windows
                and not self._pending_strong_windows
                and not self.lifecycle.unsealed
                and self.on_workload_complete is not None):
            self._workload_complete_sent = True
            self.on_workload_complete()

    def _deliver_result(self, op: Operation) -> None:
        result = DecodeResult(op.id, self.window_count[op.id] - 1,
                              logical_value=self.op_results.get(op.id))
        self.orchestrator.integrate(op, result)

    # -------------------------------------------------------- stream segments

    def release_stream_segments_at_commit(self, stream_id,
                                          committed_round_count: int) -> None:
        """Deliver segment results whose full round range has committed
        (gated the same way as ops: no pending strong may still change it)."""
        for operation in list(self.ops.values()):
            if operation.stream_id != stream_id:
                continue
            if operation.id not in self.blocking_ops:
                continue
            if operation.id in self.lifecycle.segment_results_sent:
                continue
            segment_end = self._stream_segment_end(operation)
            if segment_end is None or segment_end > committed_round_count:
                continue
            if self._segment_waits_for_strong(stream_id, segment_end):
                continue
            self.lifecycle.segment_results_sent.add(operation.id)
            self.engine.schedule(
                self.links.do.cost(),
                lambda op=operation: self.orchestrator.integrate(
                    op, DecodeResult(op.id, -1, logical_value=None)),
                label=f"result->orch({operation.name})")

    def _segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        for key in self._pending_strong_windows:
            if key[0] != stream_id:
                continue
            if self.windows[key].commit_lo <= segment_end:
                return True
        return False

    def _stream_segment_end(self, operation: Operation) -> Optional[int]:
        if operation.stream_offset is None:
            return None
        return operation.stream_offset + self.rounds_for(operation)

    # ------------------------------------------------------------- lifecycle
    # thin delegations kept for the composition root / chip side

    def close_stream_boundary(self, stream_id, stream_round_count: int) -> None:
        self.lifecycle.close_boundary(stream_id, stream_round_count)

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        self.lifecycle.seal(stream_id, stream_round_count)

    def has_dynamic_stream(self, stream_id) -> bool:
        return self.lifecycle.has(stream_id)

    def committed_stream_round_count(self, stream_id) -> int:
        return self.lifecycle.committed_round_counts.get(stream_id, 0)

    @property
    def peak_payloads(self) -> int:
        return self.store.peak_payloads

    @property
    def payloads_held(self) -> int:
        return self.store.payloads_held
