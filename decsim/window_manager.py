"""Runtime window state.

The window manager stores syndrome payloads, creates ready decode jobs, commits
finished windows, forwards boundary defects, and frees payloads. It does not own
decoder queues or unit pools.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from .message import DecodeJob, DecodeResult, Operation, SyndromePayload, Window, WindowPlan
from .planner import WindowPlanner

if TYPE_CHECKING:
    from .decoder_manager import DecoderManager
    from .engine import Engine
    from .links import LinkModel
    from .protocols import (CodeModel, DeadlinePolicy, DecodingScheme, LayoutModel,
                            Orchestrator, SyndromeSource)


class WindowManager:
    """Own window state, syndrome buffers, boundary handoff, and commits."""

    def __init__(self, engine: "Engine", *, scheme: "DecodingScheme",
                 layout: "LayoutModel", rounds_policy, code: "CodeModel",
                 decoder_manager: "DecoderManager", deadline_policy: "DeadlinePolicy",
                 links: "LinkModel", orchestrator: "Orchestrator",
                 syndrome_source: Optional["SyndromeSource"] = None):
        self.engine = engine
        self.scheme = scheme
        self.layout = layout
        self.rounds_policy = rounds_policy
        self.code = code
        self.decoder_manager = decoder_manager
        self.deadline_policy = deadline_policy
        self.links = links
        self.orchestrator = orchestrator
        self.syndrome_source = syndrome_source

        self.d = code.distance
        self.commit = code.commit_rounds()
        self.buffer = code.buffer_rounds()

        self.ops: dict[int, Operation] = {}
        self.rounds_arrived: dict[int, int] = {}
        self.memory_rounds: dict[int, int] = {}
        self.payload_store: dict[int, dict[int, dict]] = {}
        self.payloads_held = 0
        self.peak_payloads = 0
        self._round_refs: dict[tuple, int] = {}
        self._read_sets: dict[tuple, list] = {}
        self._strong_read_sets: dict[tuple, list] = {}

        self.windows: dict[tuple, Window] = {}
        self.op_windows: dict[int, list] = {}
        self.window_count: dict[int, int] = {}
        self.successors: dict[int, list] = {}
        self.committed_windows: set = set()
        self._committed_per_op: dict[int, int] = {}
        self._blocking_ops: set[int] = set()
        self.op_results: dict[int, int] = {}
        self._window_logical_values: dict[tuple, int] = {}
        self._committed_boundary_defects: dict[tuple, Optional[dict]] = {}
        self._pending_strong_windows: set[tuple] = set()
        self._pending_strong_per_op: dict[int, int] = {}
        #: op_id -> latest strong-commit tick across its windows.
        self.op_strong_commit_time: dict[int, int] = {}
        self._finished_ops: set[int] = set()
        self._workload_complete_sent = False
        self.window_models: dict = {}
        self.total_windows = 0
        self._windows_built = False
        self._dynamic_streams: dict = {}
        self._unsealed_streams: set = set()
        self._closed_stream_boundaries: dict = {}
        self._committed_stream_round_counts: dict = {}
        self._stream_segment_results_sent: set[int] = set()
        self._plan_spatial = None
        self.on_workload_complete: Optional[Callable[[], None]] = None

    def register_op(self, op: Operation) -> None:
        """Start tracking an operation's rounds, payload store, and feedback role."""
        if op.id not in self.ops:
            self.rounds_arrived[op.id] = 0
            self.memory_rounds[op.id] = 0
            self.payload_store[op.id] = {}
        self.ops[op.id] = op
        if op.blocked_by is not None:
            self._blocking_ops.add(op.blocked_by)

    def register_dynamic_stream(self, stream_op: Operation, code) -> None:
        """Register a stream whose windows are created from arriving rounds at runtime."""
        stream_id = stream_op.id
        self.ops[stream_id] = stream_op
        self.rounds_arrived.setdefault(stream_id, 0)
        self.memory_rounds.setdefault(stream_id, 0)
        self.payload_store.setdefault(stream_id, {})
        self.window_count[stream_id] = 0
        self.op_windows[stream_id] = []
        self.successors.setdefault(stream_id, [])
        source_round_limit = None
        if self.syndrome_source is not None:
            source_round_limit = self.syndrome_source.register_dynamic_stream(
                stream_op,
                self.rounds_for(stream_op),
                belief_matching=self._wants_belief_matching())
        self._dynamic_streams[stream_id] = {
            "commit_rounds": code.commit_rounds(),
            "buffer_rounds": code.buffer_rounds(),
            "next_window": 0,
            "sealed": False,
            "source_round_limit": source_round_limit,
            "sealed_round_count": None,
        }
        self._unsealed_streams.add(stream_id)
        self._closed_stream_boundaries.setdefault(stream_id, set())

    def has_dynamic_stream(self, stream_id) -> bool:
        """Return True when stream_id is registered for runtime window growth."""
        return stream_id in self._dynamic_streams

    def close_stream_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed feedback boundary."""
        if stream_id not in self._dynamic_streams:
            return
        if stream_round_count < 1:
            raise ValueError(
                f"stream_round_count must be >= 1 (got {stream_round_count})")

        self._reject_unsupported_internal_real_stream_boundary(
            stream_id, stream_round_count)
        self._closed_stream_boundaries.setdefault(stream_id, set()).add(stream_round_count)
        self.grow_stream(stream_id, rounds_to_plan=stream_round_count)
        self._refresh_unqueued_stream_windows(stream_id)
        self._check_windows_for_operation(stream_id)

    def _reject_unsupported_internal_real_stream_boundary(
            self, stream_id, stream_round_count: int) -> None:
        """Reject internal closed boundaries in finite real-syndrome streams."""
        stream_state = self._dynamic_streams[stream_id]
        source_round_limit = stream_state["source_round_limit"]
        if source_round_limit is None:
            return
        if stream_round_count >= source_round_limit:
            return
        raise RuntimeError(
            "measurement_closed live-stream boundaries inside a finite real-syndrome "
            "stream need a source circuit with that destructive boundary. A continuous "
            f"stream model was registered for {source_round_limit} rounds, but the "
            f"feedback boundary closes at round {stream_round_count}. Use timing-only "
            "payloads for this timing study, split the workload into finite operation "
            "circuits, or provide a boundary-aware syndrome source.")

    def _refresh_unqueued_stream_windows(self, stream_id) -> None:
        """Refresh retained reads after a stream boundary closes future context."""
        for window_index in self.op_windows.get(stream_id, []):
            key = (stream_id, window_index)
            window = self.windows[key]
            if window.queued or window.committed:
                continue
            self._replace_window_read_refs(key, window)

    def committed_stream_round_count(self, stream_id) -> int:
        """Number of initial stream rounds whose commit regions have decoded."""
        return self._committed_stream_round_counts.get(stream_id, 0)

    def grow_stream(self, stream_id, rounds_to_plan: Optional[int] = None) -> None:
        """Create every dynamic-stream window whose commit region has begun."""
        stream_state = self._dynamic_streams[stream_id]
        if stream_state["sealed"]:
            return
        commit_rounds = stream_state["commit_rounds"]
        buffer_rounds = stream_state["buffer_rounds"]
        highest_known_round = self.rounds_arrived[stream_id] \
            if rounds_to_plan is None else rounds_to_plan
        while stream_state["next_window"] * commit_rounds + 1 <= highest_known_round:
            window_index = stream_state["next_window"]
            commit_lo = window_index * commit_rounds + 1
            commit_hi = self._dynamic_commit_hi(stream_state, window_index)
            buffer_hi = commit_hi + buffer_rounds
            self._create_window(stream_id, window_index, commit_lo,
                                commit_hi, buffer_hi, is_last=False)
            stream_state["next_window"] += 1

    @staticmethod
    def _dynamic_commit_hi(stream_state: dict, window_index: int) -> int:
        """Commit end for one dynamic window, clipped when a cap is known."""
        commit_rounds = stream_state["commit_rounds"]
        commit_hi = (window_index + 1) * commit_rounds
        known_round_count = stream_state["sealed_round_count"]
        if known_round_count is None:
            known_round_count = stream_state["source_round_limit"]
        if known_round_count is None:
            return commit_hi
        return min(commit_hi, known_round_count)

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        """Close a dynamic stream once its full length has arrived."""
        stream_state = self._dynamic_streams[stream_id]
        if stream_state["sealed"]:
            return
        self._check_stream_source_length(stream_id, stream_round_count)
        stream_state["sealed_round_count"] = stream_round_count
        self.grow_stream(stream_id, rounds_to_plan=stream_round_count)
        self._trim_sealed_stream_tail(stream_id, stream_round_count)
        stream_state["sealed"] = True
        self._unsealed_streams.discard(stream_id)
        self._check_windows_for_operation(stream_id)
        self._finish_workload_if_ready()

    def _check_stream_source_length(self, stream_id, stream_round_count: int) -> None:
        """Let the syndrome source validate fixed-length stream models."""
        if self.syndrome_source is None:
            return
        self.syndrome_source.validate_stream_length(
            self.ops[stream_id], stream_round_count)

    def _trim_sealed_stream_tail(self, stream_id, stream_round_count: int) -> None:
        """Clip the final open-stream commit region to the actual sealed length."""
        stream_state = self._dynamic_streams[stream_id]
        for window_index in self.op_windows.get(stream_id, []):
            window = self.windows[(stream_id, window_index)]
            if window.commit_lo <= stream_round_count <= window.commit_hi:
                window.commit_hi = stream_round_count
                window.buffer_hi = stream_round_count + stream_state["buffer_rounds"]
                window.n_rounds = window.buffer_hi - window.start_round + 1
                self._reset_dynamic_window_reads(stream_id, window_index, window)
                return

    def _reset_dynamic_window_reads(self, stream_id, window_index: int,
                                    window: Window) -> None:
        """Refresh read references after a live stream tail is clipped."""
        key = (stream_id, window_index)
        old_reads = set(self._read_sets.get(key, ()))
        new_reads = {
            (stream_id, round_index)
            for round_index in range(window.start_round, window.commit_hi + 1)
        }
        for round_key in old_reads - new_reads:
            self._round_refs[round_key] = self._round_refs.get(round_key, 0) - 1
            if self._round_refs[round_key] <= 0:
                self._round_refs.pop(round_key, None)
        for round_key in new_reads - old_reads:
            self._round_refs[round_key] = self._round_refs.get(round_key, 0) + 1
        self._read_sets[key] = sorted(new_reads)

    def _create_window(self, stream_id, window_index, commit_lo, commit_hi,
                       buffer_hi, *, is_last) -> None:
        """Create one dynamic-stream window and wire it into the live plan."""
        buffer_lo = commit_lo
        window = Window(op_id=stream_id, k=window_index, commit_lo=commit_lo,
                        commit_hi=commit_hi, buffer_hi=buffer_hi,
                        n_rounds=buffer_hi - buffer_lo + 1,
                        buffer_lo=buffer_lo)
        if window_index > 0:
            previous_key = (stream_id, window_index - 1)
            if previous_key in self.committed_windows:
                self._store_boundary(
                    window, stream_id,
                    self._committed_boundary_defects.get(previous_key))
            else:
                window.deps.append(previous_key)
                window.deps_remaining = 1
                self.windows[previous_key].dependents.append((stream_id, window_index))
        self.windows[(stream_id, window_index)] = window
        self.op_windows[stream_id].append(window_index)
        self.window_count[stream_id] += 1
        self.total_windows += 1
        model = self._dynamic_window_model(stream_id, window, is_last=is_last)
        if model is not None:
            self.window_models[(stream_id, window_index)] = model
        self._add_window_read_refs((stream_id, window_index), window)
        self._check_window((stream_id, window_index))

    def _dynamic_window_model(self, stream_id, window: Window, *, is_last: bool):
        """Ask the syndrome source for this runtime-built window's model."""
        if self.syndrome_source is None:
            return None
        return self.syndrome_source.window_model_for_stream(
            stream_id, window, is_last=is_last)

    def _stream_sealed(self, op_id) -> bool:
        stream_state = self._dynamic_streams.get(op_id)
        return stream_state is None or stream_state["sealed"]

    def _round_count_for_window(self, op_id, window: Optional[Window] = None) -> int:
        """Round count to use when checking or reading one window."""
        stream_state = self._dynamic_streams.get(op_id)
        if stream_state is None:
            return self.rounds_for(self.ops[op_id])

        if stream_state["sealed"]:
            return stream_state["sealed_round_count"]
        if stream_state["source_round_limit"] is not None:
            return stream_state["source_round_limit"]
        if window is not None:
            return window.buffer_hi
        return self.rounds_arrived.get(op_id, 0)

    def rounds_for(self, op: Operation) -> int:
        """Rounds this operation runs for under its code/layout."""
        return self.rounds_policy.rounds_for(op, self.layout.code_for_op(op))

    def _spatial_nodes(self, op: Operation) -> int:
        """Decoding-graph size for an operation."""
        if self._plan_spatial is not None and op.id in self._plan_spatial:
            return self._plan_spatial[op.id]
        return self.layout.spatial_nodes_for(op)

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

        self._log_execution_plan(plan)

        if self.decoder_manager.switching is not None:
            self.decoder_manager.switching.check_window_size(self.commit, self.buffer)
        self._build_window_error_models()
        self._build_round_refcounts()

    def _log_execution_plan(self, plan: WindowPlan) -> None:
        """Log a short description of the installed plan."""
        scheme_label = getattr(self.scheme, "scheme_label", type(self.scheme).__name__)
        job_count = plan.total_windows
        job_label = f"{job_count} decode job" + ("" if job_count == 1 else "s")
        self.engine.log(
            "DecoderCluster",
            f"received execution plan: d={self.d}, scheme={scheme_label}",
        )
        self.engine.log(
            "DecoderCluster",
            f"  -> {job_label}: {self._plan_structure(plan)}",
        )

    def _plan_structure(self, plan: WindowPlan) -> str:
        """Human-readable shape of the first operation's windows."""
        first_operation_id = next(iter(self.ops), None)
        windows_per_operation = plan.window_count.get(first_operation_id, 0)
        rounds_per_operation = plan.summary.get("rounds_per_op", "?")
        first_window = self.windows.get((first_operation_id, 0)) \
            if first_operation_id is not None else None
        if not self._windowed:
            return f"each {rounds_per_operation}-round operation decoded as one batch"
        if first_window is None:
            return f"{windows_per_operation} window(s) per operation"
        commit_size = first_window.commit_hi - first_window.commit_lo + 1
        buffer_size = max(0, first_window.buffer_hi - first_window.commit_hi)
        plural = "" if windows_per_operation == 1 else "s"
        total_size = commit_size + buffer_size
        return (f"each {rounds_per_operation}-round operation split into "
                f"{windows_per_operation} window{plural} of {commit_size} commit "
                f"+ {buffer_size} buffer ({total_size} rounds each)")

    def _build_window_error_models(self) -> None:
        """Ask the syndrome source for per-window detector error models."""
        if self.syndrome_source is None:
            return
        want_bm = self._wants_belief_matching()
        for op_id, op in self.ops.items():
            keys = [(op_id, k) for k in self.op_windows.get(op_id, [])]
            wins = [self.windows[key] for key in keys]
            if not wins:
                continue
            operation_rounds = self.rounds_for(op)
            models = self.syndrome_source.window_models_for_operation(
                op, wins, operation_rounds,
                belief_matching=want_bm)
            if not models:
                continue
            for key, model in zip(keys, models):
                self.window_models[key] = model
            self.engine.log("DecoderCluster",
                            f"{op.name}: built {len(models)} decode error model(s) "
                            f"({sum(m.check.shape[1] for m in models)} fault columns)")

    def _wants_belief_matching(self) -> bool:
        """Return True when any configured decoder needs hyperedge model fields."""
        return getattr(self.decoder_manager.decoder, "needs_hyperedges", False) or any(
            getattr(decoder, "needs_hyperedges", False)
            for decoder in self.decoder_manager.decoders.values())

    def _build_round_refcounts(self) -> None:
        """Count how many unfinished windows still need each retained round."""
        self._round_refs = {}
        self._read_sets = {}
        self._strong_read_sets = {}
        for key, window in self.windows.items():
            self._add_window_read_refs(key, window)

    def _add_window_read_refs(self, key: tuple, window: Window) -> None:
        """Retain rounds needed by the weak window and possible strong re-decode."""
        self._read_sets[key] = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window)
        self._strong_read_sets[key] = self._strong_context_read_keys(window)
        for round_key in self._read_sets[key] + self._strong_read_sets[key]:
            self._round_refs[round_key] = self._round_refs.get(round_key, 0) + 1

    def _read_keys_for_bounds(self, op_id, start_round: int, buffer_hi: int,
                              window: Optional[Window] = None) -> list:
        """Return retained payload round keys for a possibly cross-operation range."""
        operation_rounds = self._effective_round_count_for_window(op_id, window)
        reads = [
            (op_id, round_index)
            for round_index in range(start_round, min(buffer_hi, operation_rounds) + 1)
        ]
        overflow = buffer_hi - operation_rounds
        if overflow <= 0:
            return reads
        for successor_id in self.successors.get(op_id, []):
            reads += [
                (successor_id, round_index)
                for round_index in range(1, overflow + 1)
            ]
        return reads

    def _strong_context_read_keys(self, window: Window) -> list:
        """Rounds retained until we know whether the strong decoder needs them."""
        if self.decoder_manager.switching is None:
            return []
        buffer_lo, _commit_lo, _commit_hi, buffer_hi = self._strong_context_bounds(window)
        weak_reads = set(self._read_sets.get((window.op_id, window.k), ()))
        strong_reads = self._read_keys_for_bounds(
            window.op_id, buffer_lo, buffer_hi, window)
        return [round_key for round_key in strong_reads if round_key not in weak_reads]

    def _replace_window_read_refs(self, key: tuple, window: Window) -> None:
        """Replace retained-round references for an unqueued window."""
        old_reads = set(self._read_sets.get(key, ()))
        old_strong_reads = set(self._strong_read_sets.get(key, ()))
        old_round_keys = old_reads | old_strong_reads

        new_reads = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window)
        if self.decoder_manager.switching is None:
            new_strong_reads = []
        else:
            buffer_lo, _commit_lo, _commit_hi, buffer_hi = \
                self._strong_context_bounds(window)
            strong_candidates = self._read_keys_for_bounds(
                window.op_id, buffer_lo, buffer_hi, window)
            new_strong_reads = [
                round_key
                for round_key in strong_candidates
                if round_key not in set(new_reads)
            ]
        new_round_keys = set(new_reads) | set(new_strong_reads)

        for round_key in old_round_keys - new_round_keys:
            self._drop_round_ref(round_key)
        for round_key in new_round_keys - old_round_keys:
            self._round_refs[round_key] = self._round_refs.get(round_key, 0) + 1

        self._read_sets[key] = list(new_reads)
        self._strong_read_sets[key] = list(new_strong_reads)

    def _drop_round_ref(self, round_key: tuple) -> None:
        """Drop one retained-round reference and free the payload if it was the last."""
        self._round_refs[round_key] = self._round_refs.get(round_key, 0) - 1
        if self._round_refs[round_key] > 0:
            return
        round_op, round_no = round_key
        fragments = self.payload_store.get(round_op, {}).pop(round_no, None)
        if fragments is not None:
            self.payloads_held -= len(fragments)
        self._round_refs.pop(round_key, None)

    def build_windows(self) -> None:
        """Compatibility entry: build and install a plan if none has been loaded."""
        if self._windows_built:
            return
        planner = WindowPlanner(self.scheme, self.layout, self.rounds_policy)
        self.load_execution_plan(planner.plan(list(self.ops.values())))

    def on_syndrome_arrival(self, payload: SyndromePayload) -> None:
        """Store an arriving syndrome round and re-check affected windows."""
        op = self.ops[payload.operation_id]
        self._store_payload(payload, op)
        self._maybe_update_dynamic_stream(op.id)
        self._check_windows_for_operation(op.id)
        self._check_predecessor_windows(op)

    def _store_payload(self, payload: SyndromePayload, op: Operation) -> None:
        """Store one payload fragment and update round-completeness counters."""
        op_store = self.payload_store.get(op.id)
        if op_store is None:
            raise RuntimeError(
                f"round {payload.round_index} of {op.name} arrived after the op's last "
                f"window committed and its syndrome RAM was freed. The device emitted "
                f"more rounds than the execution plan expects."
            )
        fragments = op_store.setdefault(payload.round_index, {})
        if payload.patch_id not in fragments:
            self.payloads_held += 1
        fragments[payload.patch_id] = payload
        if len(fragments) >= payload.n_fragments:
            self.rounds_arrived[op.id] = max(self.rounds_arrived[op.id],
                                             payload.round_index)
        self.peak_payloads = max(self.peak_payloads, self.payloads_held)
        self.engine.log("DecoderCluster",
                        f"round {payload.round_index} of {op.name} arrived "
                        f"(op now has rounds 1..{self.rounds_arrived[op.id]})")

    def _maybe_update_dynamic_stream(self, op_id: int) -> None:
        """Grow or seal a dynamic stream when an arriving round allows it."""
        if op_id in self._dynamic_streams:
            self.grow_stream(op_id)
            stream_state = self._dynamic_streams[op_id]
            if (not stream_state["sealed"]
                    and stream_state["source_round_limit"] is not None
                    and self.rounds_arrived[op_id] >= stream_state["source_round_limit"]):
                self.seal_stream(op_id, stream_state["source_round_limit"])

    def _check_windows_for_operation(self, op_id: int) -> None:
        """Re-check every window owned by one operation."""
        for window_index in range(self.window_count[op_id]):
            self._check_window((op_id, window_index))

    def _check_predecessor_windows(self, op: Operation) -> None:
        """Re-check predecessors that may need this op's overflow buffer rounds."""
        for predecessor_id in op.predecessors:
            self._check_windows_for_operation(predecessor_id)

    def prepend_idle_rounds(self, op_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation when the scheme asks for it."""
        if round_count <= 0 or not getattr(self.scheme, "batches_idle_rounds_into_next_op",
                                           False):
            return
        w = self.windows[(op_id, 0)]
        w.n_rounds += round_count
        self.engine.log("DecoderCluster",
                        f"{self.ops[op_id].name} W0 absorbs {round_count} idle rounds: "
                        f"its batch decode now covers {w.n_rounds} rounds (the "
                        f"feedback-to-feedback segment)")

    def on_memory_round(self, op_id: int) -> None:
        """Record an idle/memory round and re-check waiting windows."""
        self.memory_rounds[op_id] += 1
        self.engine.log("DecoderCluster",
                        f"memory round for {self.ops[op_id].name} "
                        f"(idle buffer rounds: {self.memory_rounds[op_id]})")
        for k in range(self.window_count[op_id]):
            self._check_window((op_id, k))

    @staticmethod
    def _xor_mask(previous_mask, incoming_mask) -> list:
        previous_bits = [int(bit) for bit in previous_mask] if previous_mask is not None else []
        incoming_bits = [int(bit) for bit in incoming_mask]
        if len(previous_bits) < len(incoming_bits):
            previous_bits += [0] * (len(incoming_bits) - len(previous_bits))
        for bit_index, bit in enumerate(incoming_bits):
            previous_bits[bit_index] ^= bit
        return previous_bits

    def _apply_boundary(self, w: Window, payload: SyndromePayload,
                        round_key: Optional[int] = None) -> SyndromePayload:
        """Return a payload copy with received artificial defects XORed in."""
        r = payload.round_index if round_key is None else round_key
        mask = w.boundary_in.get((r, payload.patch_id), w.boundary_in.get(r))
        if mask is None:
            return payload
        from dataclasses import replace
        bits = [int(m) for m in mask] if payload.bits is None \
            else self._xor_mask(payload.bits, mask)
        return replace(payload, bits=bits)

    def _assemble_payloads(self, w: Window) -> list:
        """Collect this window's payloads, including successor overflow rounds."""
        op_store = self.payload_store.get(w.op_id, {})
        operation_rounds = self._effective_round_count_for_window(w.op_id, w)
        end_round = min(w.buffer_hi, operation_rounds)
        payloads = []
        for round_index in range(w.start_round, end_round + 1):
            if round_index in op_store:
                payloads += [
                    self._apply_boundary(w, op_store[round_index][patch_id])
                    for patch_id in sorted(op_store[round_index])
                ]
        overflow = w.buffer_hi - operation_rounds
        if overflow > 0:
            for successor_id in self.successors.get(w.op_id, []):
                successor_store = self.payload_store.get(successor_id, {})
                for round_index in range(1, overflow + 1):
                    if round_index in successor_store:
                        payloads += [
                            self._apply_boundary(
                                w, successor_store[round_index][patch_id],
                                round_key=operation_rounds + round_index)
                            for patch_id in sorted(successor_store[round_index])
                        ]
        return payloads

    def _window_data_complete(self, w: Window) -> bool:
        op = self.ops[w.op_id]
        succ_rounds = self._successor_rounds_available(w)
        round_count = self._effective_round_count_for_window(w.op_id, w)
        has_successor = op.has_successor and not self._window_has_closed_boundary(w)
        return self.scheme.data_complete(
            w, rounds_arrived=self.rounds_arrived[w.op_id],
            successor_rounds=succ_rounds, memory_rounds=self.memory_rounds[w.op_id],
            round_count=round_count,
            has_successor=has_successor,
            op=op, layout=self.layout)

    def _effective_round_count_for_window(self, op_id, window: Optional[Window]) -> int:
        """Round count seen by a window after applying measurement-closed boundaries."""
        round_count = self._round_count_for_window(op_id, window)
        if window is None:
            return round_count

        closed_boundary = self._closed_boundary_round_for_window(window)
        if closed_boundary is None:
            return round_count
        return min(round_count, closed_boundary)

    def _window_has_closed_boundary(self, window: Window) -> bool:
        """Return True when a measurement closes this window's future buffer."""
        return self._closed_boundary_round_for_window(window) is not None

    def _closed_boundary_round_for_window(self, window: Window) -> Optional[int]:
        """Boundary round that closes the window's trailing context, if any."""
        stream_boundary = self._closed_stream_boundary_for_window(window)
        if stream_boundary is not None:
            return stream_boundary

        op = self.ops[window.op_id]
        if op.feedback_boundary_mode != "measurement_closed":
            return None
        if op.id not in self._blocking_ops:
            return None

        round_count = self._round_count_for_window(window.op_id, window)
        if window.commit_hi <= round_count < window.buffer_hi:
            return round_count
        return None

    def _closed_stream_boundary_for_window(self, window: Window) -> Optional[int]:
        """Return a closed live-stream boundary covered by this window."""
        boundaries = self._closed_stream_boundaries.get(window.op_id, ())
        covered = [
            boundary
            for boundary in boundaries
            if window.commit_hi <= boundary < window.buffer_hi
        ]
        return min(covered) if covered else None

    def _successor_rounds_available(self, window: Window) -> int:
        """Successor rounds available for a buffer that crosses an operation seam."""
        successor_ids = self.successors[window.op_id]
        successor_rounds = max((self.rounds_arrived[successor_id]
                                for successor_id in successor_ids), default=0)
        overflow = window.buffer_hi - self._round_count_for_window(window.op_id, window)
        if overflow <= 0 or not successor_ids or successor_rounds >= overflow:
            return successor_rounds
        successors_exhausted = all(
            self.rounds_arrived[successor_id] >= self._round_count_for_window(successor_id)
            for successor_id in successor_ids
        )
        return overflow if successors_exhausted else successor_rounds

    @property
    def _windowed(self) -> bool:
        return getattr(self.scheme, "windowed", True)

    def _job_desc(self, w: Window, op: Operation) -> str:
        if self._windowed:
            return f"{op.name} W{w.k} [commit {w.commit_lo}-{w.commit_hi}]"
        body_rounds = self._round_count_for_window(op.id, w)
        idle_rounds = max(0, w.n_rounds - body_rounds)
        if idle_rounds:
            return (f"{op.name} [whole op, {w.n_rounds} rounds: "
                    f"{idle_rounds} idle + {body_rounds} body]")
        return f"{op.name} [whole op, {w.n_rounds} rounds]"

    def _check_window(self, key: tuple) -> None:
        """If a window has its data and dependencies, submit its decode job."""
        window = self.windows[key]
        if window.queued or window.committed:
            return
        self._stamp_first_round_if_ready(window)
        if not self._window_data_complete(window):
            return
        self._stamp_data_complete_if_ready(window)
        op = self.ops[window.op_id]
        if self._waiting_on_dependencies(window, op):
            return
        self._submit_window_decode(key, window, op)

    def _stamp_first_round_if_ready(self, window: Window) -> None:
        """Record when this window first has any data in memory."""
        if (window.t_first_round is None
                and self.rounds_arrived[window.op_id] >= window.start_round):
            window.t_first_round = self.engine.now

    def _stamp_data_complete_if_ready(self, window: Window) -> None:
        """Record when this window first has all needed data."""
        if window.t_data_complete is None:
            window.t_data_complete = self.engine.now

    def _waiting_on_dependencies(self, window: Window, op: Operation) -> bool:
        """Return True when data is ready but boundary dependencies are not."""
        if window.deps_remaining <= 0:
            return False
        if not window.blocked_logged:
            window.blocked_logged = True
            self.engine.log(
                "DecoderCluster",
                f"{self._job_desc(window, op)} has all its data, but is WAITING for "
                f"the boundary from {window.deps_remaining} predecessor window(s)",
            )
        return True

    def _submit_window_decode(self, key: tuple, window: Window, op: Operation) -> None:
        """Build and submit the decode job for a ready window."""
        window.t_queued = self.engine.now
        deadline = self.deadline_policy.deadline(
            op, window, self.engine.now, on_reaction_path=(op.id in self._blocking_ops))
        job = DecodeJob(op_id=window.op_id, window_id=window.k, n_rounds=window.n_rounds,
                        ready_time=self.engine.now, deadline=deadline,
                        spatial_nodes=self._spatial_nodes(op),
                        payloads=self._assemble_payloads(window),
                        dem=self.window_models.get(key),
                        code=self.layout.code_for_op(op).name,
                        window=window,
                        label=self._job_desc(window, op))
        job.strong_label = f"strong({op.name} W{window.k})"
        window.queued = True
        self.decoder_manager.submit_window(job)

    def make_strong_decode_job(self, weak_job: DecodeJob, round_count: int,
                               label: str) -> DecodeJob:
        """Build a paper-style two-sided strong re-decode job for an escalated window."""
        key = (weak_job.op_id, weak_job.window_id)
        weak_window = self.windows[key]
        op = self.ops[weak_job.op_id]
        strong_window = self._strong_context_window(weak_window)
        return DecodeJob(
            op_id=weak_job.op_id,
            window_id=weak_job.window_id,
            n_rounds=round_count,
            ready_time=self.engine.now,
            deadline=self.engine.now,
            label=label,
            hint="strong",
            spatial_nodes=weak_job.spatial_nodes,
            code=weak_job.code,
            dem=self._strong_window_model(op, strong_window),
            payloads=self._assemble_payloads(strong_window),
            attempt=1,
            window=strong_window,
            strong_decode_for=key)

    def _strong_context_window(self, weak_window: Window) -> Window:
        """Strong context uses one buffer before and one buffer after the weak commit."""
        buffer_lo, commit_lo, commit_hi, buffer_hi = self._strong_context_bounds(weak_window)
        strong_window = Window(
            op_id=weak_window.op_id,
            k=weak_window.k,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=buffer_hi,
            buffer_lo=buffer_lo,
            n_rounds=buffer_hi - buffer_lo + 1)
        strong_window.boundary_in = dict(weak_window.boundary_in)
        return strong_window

    @staticmethod
    def _strong_context_bounds(window: Window) -> tuple[int, int, int, int]:
        """Return buffer_lo, commit_lo, commit_hi, buffer_hi for a strong re-decode."""
        buffer_rounds = max(0, window.buffer_hi - window.commit_hi)
        buffer_lo = max(1, window.commit_lo - buffer_rounds)
        buffer_hi = window.commit_hi + buffer_rounds
        return buffer_lo, window.commit_lo, window.commit_hi, buffer_hi

    def _strong_window_model(self, op: Operation, window: Window):
        """Ask the syndrome source for a strong re-decode model when one exists."""
        if self.syndrome_source is None:
            return None
        return self.syndrome_source.strong_window_model_for_operation(
            op, window, self._round_count_for_window(op.id, window),
            belief_matching=self._wants_belief_matching())

    def on_decode_done(self, job: DecodeJob, res: DecodeResult) -> None:
        """Commit a finished operation-window decode."""
        key = (job.op_id, job.window_id)
        window = self.windows[key]
        op = self.ops[job.op_id]
        self._commit_window(job, res, key, window, op)
        self._update_committed_stream_round_count(op.id)
        defects = res.boundary_defects if res is not None else None
        self._send_boundary_defects(window, op, defects)
        self._release_window_reads(key)
        self._finish_operation_if_ready(op)
        self._finish_workload_if_ready()

    def _commit_window(self, job: DecodeJob, res: DecodeResult, key: tuple,
                       window: Window, op: Operation) -> None:
        """Mark a decoded window committed and fold in its logical value."""
        window.committed = True
        window.t_done = self.engine.now
        self.committed_windows.add(key)
        self._committed_per_op[op.id] = self._committed_per_op.get(op.id, 0) + 1
        self.engine.log("DecoderCluster",
                        f"DECODE DONE {self._job_desc(window, op)}. "
                        f"{'committed' if self._windowed else 'decoded'} "
                        f"({self.decoder_manager.pool_tag(job.pool)}units free now "
                        f"{self.decoder_manager.pool_free[job.pool]})")
        if res is not None and res.logical_value is not None:
            self._replace_window_logical_value(key, op.id, int(res.logical_value))
        if job.awaiting_strong_result:
            self._mark_window_waiting_for_strong(key, op.id)
        self._release_strong_context_reads(key)

    def _replace_window_logical_value(self, key: tuple, op_id: int, value: int) -> None:
        """Replace one window's logical contribution in the operation result."""
        previous = self._window_logical_values.get(key)
        if previous is not None:
            self.op_results[op_id] = self.op_results.get(op_id, 0) ^ previous
        self._window_logical_values[key] = value
        self.op_results[op_id] = self.op_results.get(op_id, 0) ^ value

    def _mark_window_waiting_for_strong(self, key: tuple, op_id: int) -> None:
        """Delay operation-level output until this strong result arrives."""
        if key in self._pending_strong_windows:
            return
        self._pending_strong_windows.add(key)
        self._pending_strong_per_op[op_id] = self._pending_strong_per_op.get(op_id, 0) + 1

    def on_strong_decode_done(self, key: tuple, result: DecodeResult) -> None:
        """Finalize a weak-committed window with the strong decoder's logical result."""
        window = self.windows[key]
        op = self.ops[window.op_id]
        if result is not None and result.logical_value is not None:
            self._replace_window_logical_value(key, op.id, int(result.logical_value))
        # Stamp the op-level strong-commit tick (closed_loop's fence reads it).
        self.op_strong_commit_time[op.id] = max(
            self.op_strong_commit_time.get(op.id, 0), self.engine.now)
        self._resolve_strong_wait(key, op.id)
        self._release_stream_segments_at_commit(
            op.id,
            self._committed_stream_round_counts.get(op.id, 0))
        self._finish_operation_if_ready(op)
        self._finish_workload_if_ready()

    def _resolve_strong_wait(self, key: tuple, op_id: int) -> None:
        """Clear one pending strong finalization."""
        if key not in self._pending_strong_windows:
            return
        self._pending_strong_windows.remove(key)
        remaining = self._pending_strong_per_op.get(op_id, 0) - 1
        if remaining > 0:
            self._pending_strong_per_op[op_id] = remaining
        else:
            self._pending_strong_per_op.pop(op_id, None)

    def _release_strong_context_reads(self, key: tuple) -> None:
        """Free rounds retained only for a possible strong re-decode."""
        for round_key in self._strong_read_sets.pop(key, ()):
            self._drop_round_ref(round_key)

    def _update_committed_stream_round_count(self, stream_id) -> None:
        """Update how many initial rounds of a live stream are committed."""
        committed_round_count = self._committed_prefix_round_count(stream_id)
        if committed_round_count <= self._committed_stream_round_counts.get(stream_id, 0):
            return

        self._committed_stream_round_counts[stream_id] = committed_round_count
        self._release_stream_segments_at_commit(stream_id, committed_round_count)

    def _committed_prefix_round_count(self, stream_id) -> int:
        """Return how many initial rounds are covered by committed windows."""
        committed_ranges = sorted(
            (self.windows[key].commit_lo, self.windows[key].commit_hi)
            for key in self.committed_windows
            if key[0] == stream_id
        )
        committed_round_count = 0
        for start_round, end_round in committed_ranges:
            if start_round > committed_round_count + 1:
                break
            committed_round_count = max(committed_round_count, end_round)
        return committed_round_count

    def _release_stream_segments_at_commit(self, stream_id,
                                           committed_round_count: int) -> None:
        """Deliver segment results whose full round range has committed."""
        for operation in list(self.ops.values()):
            if operation.stream_id != stream_id:
                continue
            if operation.id not in self._blocking_ops:
                continue
            if operation.id in self._stream_segment_results_sent:
                continue

            segment_end = self._stream_segment_end(operation)
            if segment_end is None or segment_end > committed_round_count:
                continue
            if self._stream_segment_waits_for_strong(stream_id, segment_end):
                continue

            self._stream_segment_results_sent.add(operation.id)
            self.engine.schedule(
                self.links.do.cost(),
                lambda op=operation: self._deliver_stream_segment_to_orchestrator(op),
                label=f"result->orch({operation.name})")

    def _stream_segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        """Return true when a pending strong result can still change this segment."""
        for key in self._pending_strong_windows:
            if key[0] != stream_id:
                continue
            if self.windows[key].commit_lo <= segment_end:
                return True
        return False

    def _stream_segment_end(self, operation: Operation) -> Optional[int]:
        """Last global stream round covered by one scheduled segment."""
        if operation.stream_offset is None:
            return None
        return operation.stream_offset + self.rounds_for(operation)

    def _deliver_stream_segment_to_orchestrator(self, operation: Operation) -> None:
        """Deliver a frontier-backed result for a stream segment."""
        result = DecodeResult(operation.id, -1, logical_value=None)
        self.orchestrator.integrate(operation, result)

    def _send_boundary_defects(self, window: Window, op: Operation,
                               defects: Optional[dict]) -> None:
        """Send this window's artificial defects to dependent windows."""
        self._committed_boundary_defects[(window.op_id, window.k)] = defects
        for dep_key in window.dependents:
            dst = self.ops[dep_key[0]]
            self.engine.log("DecoderCluster",
                            f"decoder->decoder SEND: {op.name} W{window.k} -> {dst.name} "
                            f"W{dep_key[1]}  (boundary/artificial defects, arrives in t_dd)")
            self.engine.schedule(self.links.dd.cost(),
                                 lambda dk=dep_key, sn=op.name, sk=window.k,
                                        so=op.id, bd=defects:
                                     self._receive_boundary(dk, sn, sk, so, bd),
                                 label=f"defects {op.name}W{window.k}->{dst.name}W{dep_key[1]}")

    def _release_window_reads(self, key: tuple) -> None:
        """Free payload rounds that no unfinished window still reads."""
        for round_key in self._read_sets.get(key, ()):
            self._drop_round_ref(round_key)

    def _finish_operation_if_ready(self, op: Operation) -> None:
        """Deliver an operation result once all of its windows committed."""
        if op.id in self._finished_ops:
            return
        if self._pending_strong_per_op.get(op.id, 0) > 0:
            return
        if self._committed_per_op[op.id] == self.window_count[op.id] and self._stream_sealed(op.id):
            self._finished_ops.add(op.id)
            self.engine.schedule(self.links.do.cost(),
                                 lambda: self._deliver_to_orchestrator(op),
                                 label=f"result->orch({op.name})")
            freed = self.payload_store.pop(op.id, None)
            if freed:
                self.payloads_held -= sum(len(frags) for frags in freed.values())

    def _finish_workload_if_ready(self) -> None:
        """Run the workload completion callback once the full window set drains."""
        if self._workload_complete_sent:
            return
        if (len(self.committed_windows) == self.total_windows
                and not self._pending_strong_windows
                and not self._unsealed_streams
                and self.on_workload_complete is not None):
            self._workload_complete_sent = True
            self.on_workload_complete()

    def _store_boundary(self, w: Window, src_op_id: int, defects: Optional[dict]) -> None:
        """Store artificial defects on a dependent window in that window's round numbering."""
        if not defects:
            return
        shift = 0 if src_op_id == w.op_id else -self.rounds_for(self.ops[src_op_id])
        for key, mask in defects.items():
            r, patch = key if isinstance(key, tuple) else (key, None)
            r += shift
            if r < 1:
                continue
            dst_key = (r, patch) if patch is not None else r
            w.boundary_in[dst_key] = self._xor_mask(w.boundary_in.get(dst_key), mask)

    def _receive_boundary(self, key: tuple, src_name: str, src_k: int,
                          src_op_id: int, defects: Optional[dict] = None) -> None:
        """Receive a predecessor window's artificial-defect boundary."""
        w = self.windows[key]
        op = self.ops[w.op_id]
        self._store_boundary(w, src_op_id, defects)
        w.deps_remaining -= 1
        still = f"; still waiting on {w.deps_remaining}" if w.deps_remaining > 0 else ""
        self.engine.log("DecoderCluster",
                        f"decoder->decoder RECV: {op.name} W{w.k} <- {src_name} W{src_k}  "
                        f"(boundary arrived after t_dd){still}")
        self._check_window(key)

    def _deliver_to_orchestrator(self, op: Operation) -> None:
        """Deliver an operation-level decoded result to the orchestrator."""
        result = DecodeResult(op.id, self.window_count[op.id] - 1,
                              logical_value=self.op_results.get(op.id))
        self.orchestrator.integrate(op, result)
