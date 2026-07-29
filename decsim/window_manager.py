"""The windowing runtime: tracks each decode window from ready to committed.

When a window's rounds are all present it builds the decode job (submission
itself is DecodingStrategy.on_window_ready), hands the result to the
orchestrator, ships boundary data to dependent windows at commit (gated by
BoundaryPolicy.on_commit; Eager, the default, always ships), and declares
the op finished once every window is committed. The windows are laid out
either statically by the composition root or at runtime by DynamicWindows
for streams of unknown length; this hub runs whatever both produce. Two
helpers own adjacent state: PayloadStore (how
long raw syndrome rounds stay retained) and DynamicWindows (dynamic-stream
growth and sealing).

Event ordering here is load-bearing: same-tick events run FIFO, so the
frozen timing goldens pin the engine.schedule call order inside every
handler. Invariants to know before editing:
  - on_decode_done runs after the decode layer has set
    job.awaiting_strong_result, so the escalation check sees it.
  - a strong (redo) result revises the op's logical accumulator; under Eager,
    a changed boundary also rolls back and replays its static dependent cone.
  - op delivery waits on the per-op pending-strong count and the stream
    seal; requires_strong_commit marks the op but never gates delivery.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import Callable, Optional

from .message import (BoundaryDelivery, BoundaryUpdate, DecodeJob, DecodeResult,
                      Operation, SeamFaultOwner, SyndromeRoundPacket,
                      StrongRegionPlan, Window, WindowInfo, WindowPlan,
                      is_stable_identity, stable_identity_order_key)
from .links import LinkPath, TrafficAttribution
from .payload_store import PayloadStore
from .dynamic_windows import DynamicWindows
from .protocols import MultiFaultExclusionSyndromeDevice
from .speculative_recovery import SpeculativeRecovery


@dataclass(frozen=True)
class LogicalContribution:
    """One decoder prediction owner over an exact inclusive round extent."""

    owner_key: tuple
    commit_lo: int
    commit_hi: int
    ownership_kind: str
    logical_observables: Optional[tuple[int, ...]]


@dataclass(frozen=True)
class _ResolvedStrongRegion:
    """One policy-selected region resolved against the live window graph."""

    plan: StrongRegionPlan
    absorbed_window_keys: tuple
    restart_window_key: Optional[tuple]
    restart_read_keys: tuple
    strong_fault_exclusion_ranges: tuple
    restart_fault_exclusion_ranges: Optional[tuple]


class _EscalationPhase(Enum):
    """The one readiness condition that can transfer a pending strong job."""

    WAITING_FAR_BOUNDARY = auto()
    WAITING_TERMINAL_DATA = auto()


@dataclass(frozen=True)
class _PendingEscalation:
    """All immutable state retained until one strong-job transfer."""

    key: tuple
    weak_job: DecodeJob
    label: str
    resolved_region: _ResolvedStrongRegion
    strong_window: Window
    strong_model: object
    wsd_arrival_ticks: int
    phase: _EscalationPhase


class _EscalationRegistry:
    """Own pending escalations and their one exact readiness index."""

    def __init__(self) -> None:
        self._by_key: dict[tuple, _PendingEscalation] = {}
        self._by_far_boundary: dict[tuple, tuple] = {}
        self._by_terminal_operation: dict[object, tuple] = {}

    def register_far(
        self,
        pending: _PendingEscalation,
        far_boundary_key: tuple,
    ) -> None:
        self._register(
            pending,
            expected_phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
            readiness_index=self._by_far_boundary,
            readiness_key=far_boundary_key,
        )

    def register_terminal(
        self,
        pending: _PendingEscalation,
        operation_id,
    ) -> None:
        self._register(
            pending,
            expected_phase=_EscalationPhase.WAITING_TERMINAL_DATA,
            readiness_index=self._by_terminal_operation,
            readiness_key=operation_id,
        )

    def _register(
        self,
        pending: _PendingEscalation,
        *,
        expected_phase: _EscalationPhase,
        readiness_index: dict,
        readiness_key,
    ) -> None:
        if type(pending) is not _PendingEscalation:
            raise TypeError("pending escalation must use the exact record type")
        if not is_stable_identity(pending.key):
            raise TypeError("pending escalation key must be a stable identity")
        if not is_stable_identity(readiness_key):
            raise TypeError("readiness key must be a stable identity")
        if pending.phase is not expected_phase:
            raise RuntimeError(
                f"pending escalation {pending.key} has phase "
                f"{pending.phase.name}, expected {expected_phase.name}")
        if pending.key in self._by_key:
            raise RuntimeError(
                f"duplicate strong escalation for window {pending.key}: one "
                "switching event creates exactly one strong job")
        if readiness_key in readiness_index:
            raise RuntimeError(
                f"readiness index collision for {readiness_key}")
        self._by_key[pending.key] = pending
        readiness_index[readiness_key] = pending.key

    def peek_key(self, key: tuple) -> Optional[_PendingEscalation]:
        return self._by_key.get(key)

    def peek_far(self, far_boundary_key: tuple) -> Optional[_PendingEscalation]:
        key = self._by_far_boundary.get(far_boundary_key)
        return None if key is None else self._by_key[key]

    def peek_terminal(self, operation_id) -> Optional[_PendingEscalation]:
        key = self._by_terminal_operation.get(operation_id)
        return None if key is None else self._by_key[key]

    def take_far(
        self,
        far_boundary_key: tuple,
        expected: _PendingEscalation,
    ) -> _PendingEscalation:
        return self._take(
            expected,
            expected_phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
            readiness_index=self._by_far_boundary,
            readiness_key=far_boundary_key,
        )

    def take_terminal(
        self,
        operation_id,
        expected: _PendingEscalation,
    ) -> _PendingEscalation:
        return self._take(
            expected,
            expected_phase=_EscalationPhase.WAITING_TERMINAL_DATA,
            readiness_index=self._by_terminal_operation,
            readiness_key=operation_id,
        )

    def _take(
        self,
        expected: _PendingEscalation,
        *,
        expected_phase: _EscalationPhase,
        readiness_index: dict,
        readiness_key,
    ) -> _PendingEscalation:
        if type(expected) is not _PendingEscalation:
            raise TypeError("expected escalation must use the exact record type")
        if expected.phase is not expected_phase:
            raise RuntimeError(
                f"wrong-phase take for escalation {expected.key}")
        primary = self._by_key.get(expected.key)
        indexed_key = readiness_index.get(readiness_key)
        if primary is not expected or indexed_key != expected.key:
            raise RuntimeError(
                f"stale escalation take for readiness key {readiness_key}")
        del readiness_index[readiness_key]
        del self._by_key[expected.key]
        return expected

    def snapshot_phases(self):
        return MappingProxyType({
            key: pending.phase for key, pending in self._by_key.items()
        })

    def snapshot_work(self) -> tuple:
        """Return pending strong assignments without exposing live windows."""
        phase_names = {
            _EscalationPhase.WAITING_FAR_BOUNDARY: "waiting_far_boundary",
            _EscalationPhase.WAITING_TERMINAL_DATA: "waiting_terminal_data",
        }
        records = (
            (key, phase_names[pending.phase], pending.strong_window.n_rounds)
            for key, pending in self._by_key.items()
        )
        return tuple(sorted(
            records,
            key=lambda record: stable_identity_order_key(record[0]),
        ))


class WindowManager:
    """Own window state, readiness, commits, and boundary handoff."""

    def __init__(self, engine, *, scheme, code_geometry,
                 resolved_operations, resolved_patches,
                 deadline_policy, links, orchestrator, boundary_policy,
                 window_interaction,
                 planning_view_by_operation_id,
                 fault_model_requirement_for,
                 feedback_boundary_mode: str = "trailing_buffer",
                 syndrome_source=None, switching_active: bool = False,
                 store: Optional[PayloadStore] = None):
        self.engine = engine
        self.scheme = scheme
        self._code_geometry = code_geometry
        self._resolved_operations = MappingProxyType({
            operation.operation_id: operation
            for operation in resolved_operations
        })
        self._resolved_patches = MappingProxyType({
            patch.patch_identity: patch
            for patch in resolved_patches
        })
        self.deadline_policy = deadline_policy
        self.links = links
        self.orchestrator = orchestrator
        self.boundary_policy = boundary_policy
        self.window_interaction = window_interaction
        self._planning_view_by_operation_id = MappingProxyType(
            dict(planning_view_by_operation_id)
        )
        if not callable(fault_model_requirement_for):
            raise TypeError("fault_model_requirement_for must be callable")
        self._fault_model_requirement_for_code = fault_model_requirement_for
        self.feedback_boundary_mode = feedback_boundary_mode
        self.syndrome_source = syndrome_source
        #: retains rounds for possible strong re-decodes (today: switching set).
        self.switching_active = switching_active

        # Wired post-construction by the composition root:
        self.strategy = None
        self.services = None
        self.submit_fn: Optional[Callable] = None    # (job, delay_ticks) -> None
        self.on_workload_complete: Optional[Callable[[], None]] = None

        self.store = store if store is not None else PayloadStore()
        self.lifecycle = DynamicWindows(self)

        self._ops: dict[int, Operation] = {}
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
        self.op_results: dict[int, tuple[int, ...]] = {}
        self.logical_contributions: dict[tuple, LogicalContribution] = {}
        self._observable_arity_by_stream: dict[object, int] = {}
        self._committed_boundaries: dict[tuple, object] = {}
        self._boundary_versions: dict[tuple, int] = {}
        self._boundary_delivery_versions: dict[tuple, int] = {}
        self._released_boundary_dependencies: set[tuple] = set()
        self._held_boundary: dict[tuple, tuple] = {}      # Held policy deferrals
        self._pending_strong_windows: set[tuple] = set()
        self._pending_strong_per_op: dict[int, int] = {}
        self._escalations = _EscalationRegistry()
        self.absorbed_windows: set[tuple] = set()        # skipped by the weak chain
        self.op_strong_commit_time: dict[int, int] = {}
        self._finalize_gates: dict[int, Callable] = {}    # gate_finalize seam
        self._finished_ops: set[int] = set()
        self._workload_complete_sent = False
        self.speculative_recovery = SpeculativeRecovery(self)
        self.window_models: dict = {}
        self.total_windows = 0
        self._windows_built = False
        self._windowed_by_operation = {}
        self._batch_preceding_idle_rounds_by_operation = {}

    # ---------------------------------------------------------- registration

    def _fault_model_requirement(self, operation: Operation):
        """Resolve the decoder views for this operation's frozen code."""
        resolved = self._resolved_operations[operation.id]
        return self._fault_model_requirement_for_code(
            resolved.code_geometry.code_name
        )

    def register_op(self, op: Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        if op.id not in self._ops:
            self.rounds_arrived[op.id] = 0
            self.memory_rounds[op.id] = 0
            self.store.register_op(op.id)
        self._ops[op.id] = op
        if op.blocked_by is not None:
            self.blocking_ops.add(op.blocked_by)

    def _register_dynamic_stream(
        self,
        stream_op: Operation,
        resolved_operation,
    ) -> None:
        """Register a stream whose windows are created at runtime."""
        if stream_op.id != resolved_operation.operation_id:
            raise ValueError(
                "dynamic stream and resolved operation identities differ"
            )
        if (
            self._resolved_operations.get(stream_op.id)
            is not resolved_operation
        ):
            raise ValueError(
                "dynamic stream must use the root-resolved operation record"
            )
        resolved_feedback_mode = (
            stream_op.feedback_boundary_mode
            if stream_op.feedback_boundary_mode is not None
            else self.feedback_boundary_mode
        )
        stream_op = replace(
            stream_op,
            feedback_boundary_mode=resolved_feedback_mode,
        )
        stream_id = stream_op.id
        if stream_id not in self._planning_view_by_operation_id:
            raise ValueError(
                "dynamic stream must have a root-resolved planning view"
            )
        self._ops[stream_id] = stream_op
        self.rounds_arrived.setdefault(stream_id, 0)
        self.memory_rounds.setdefault(stream_id, 0)
        self.store.register_op(stream_id)
        self.window_count[stream_id] = 0
        self.op_windows[stream_id] = []
        self.successors.setdefault(stream_id, [])
        self._windowed_by_operation[stream_id] = True
        self._batch_preceding_idle_rounds_by_operation[stream_id] = False
        source_round_limit = None
        if self.syndrome_source is not None:
            source_round_limit = self.syndrome_source.register_dynamic_stream(
                stream_op, self.rounds_for(stream_op),
                fault_model_requirement=self._fault_model_requirement(stream_op))
        self.lifecycle.register(
            stream_op,
            commit_round_count=(
                resolved_operation.code_geometry.commit_round_count
            ),
            buffer_round_count=(
                resolved_operation.code_geometry.buffer_round_count
            ),
            source_round_limit=source_round_limit,
        )

    def rounds_for(self, op: Operation) -> int:
        """Return the root-resolved operation duration."""
        try:
            return self._resolved_operations[op.id].round_count
        except KeyError as error:
            raise ValueError(
                f"operation {op.id} has no resolved planning record"
            ) from error

    def _spatial_nodes(self, op: Operation) -> int:
        return self._resolved_operations[op.id].spatial_node_count

    def _planning_view(self, op: Operation):
        try:
            return self._planning_view_by_operation_id[op.id]
        except KeyError as error:
            raise ValueError(
                f"operation {op.id} has no frozen planning view"
            ) from error

    # ----------------------------------------------------------------- plan

    def load_execution_plan(self, plan: WindowPlan) -> None:
        """Install the pre-computed compile-time window plan."""
        if self._windows_built:
            return
        self._windows_built = True
        self.windows = plan.windows
        for window in self.windows.values():
            window.boundary_in = \
                self.window_interaction.initial_boundary_state(
                    WindowInfo.from_window(window))
        self.window_count = plan.window_count
        self.op_windows = plan.op_windows
        self.successors = plan.successors
        self._windowed_by_operation = plan.windowed_by_operation
        self._batch_preceding_idle_rounds_by_operation = (
            plan.batch_preceding_idle_rounds_by_operation
        )
        self.total_windows = plan.total_windows
        self._build_window_error_models()
        self._build_round_leases()

    def _build_window_error_models(self) -> None:
        """Ask the syndrome source for per-window detector error models."""
        if self.syndrome_source is None:
            return
        for op_id, op in self._ops.items():
            keys = [(op_id, k) for k in self.op_windows.get(op_id, [])]
            wins = [self.windows[key] for key in keys]
            if not wins:
                continue
            models = self.syndrome_source.window_models_for_operation(
                op, wins, self.rounds_for(op),
                fault_model_requirement=self._fault_model_requirement(op))
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

    def _require_retained_payloads(
        self, round_keys: list, purpose: str,
    ) -> None:
        """Reject a new consumer if any already-arrived input was released."""
        missing = [
            round_key for round_key in round_keys
            if (not self.store.has_op(round_key[0])
                or (round_key[1] <= self.rounds_arrived.get(round_key[0], 0)
                    and self.store.fragments(*round_key) is None))
        ]
        if missing:
            raise RuntimeError(
                f"{purpose} requires retained payload rounds that are no "
                f"longer available: {missing}")

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
        window.boundary_in = self.window_interaction.initial_boundary_state(
            WindowInfo.from_window(window))
        if window_index > 0:
            previous_key = (stream_id, window_index - 1)
            if (previous_key in self.committed_windows
                    and previous_key not in self._held_boundary):
                # boundary already shipped (a held one is NOT available yet)
                self._merge_available_boundary(
                    previous_key,
                    window,
                    self._committed_boundaries.get(previous_key),
                )
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
            self._ops[stream_id], stream_round_count)

    # -------------------------------------------------------------- arrivals

    def on_syndrome_arrival(self, packet: SyndromeRoundPacket) -> None:
        """Retain one complete syndrome round and re-check affected windows."""
        if type(packet) is not SyndromeRoundPacket:
            raise TypeError("window manager requires a SyndromeRoundPacket")
        try:
            op = self._ops[packet.operation_id]
        except KeyError as error:
            raise ValueError(
                f"unknown syndrome operation {packet.operation_id!r}"
            ) from error
        self._store_payload(packet, op)
        self.lifecycle.maybe_update(op.id)
        self._check_deferred_strong_after_arrival(op.id)
        self.check_windows_for_operation(op.id)
        for predecessor_id in op.predecessors:
            self.check_windows_for_operation(predecessor_id)

    def _store_payload(
        self,
        packet: SyndromeRoundPacket,
        op: Operation,
    ) -> None:
        if not self.store.has_op(op.id):
            raise RuntimeError(
                f"round {packet.round_index} of {op.name} arrived after the op's "
                f"last window committed and its syndrome RAM was freed. The device "
                f"emitted more rounds than the execution plan expects.")
        round_limit = self.lifecycle.arrival_round_limit(
            op.id,
            fallback_rounds=self.rounds_for(op),
        )
        if round_limit is not None and packet.round_index > round_limit:
            raise ValueError(
                f"round {packet.round_index} of {op.name} exceeds the device "
                f"round limit {round_limit}"
            )
        self.store.store_round(packet, completion_tick=self.engine.now)
        self.rounds_arrived[op.id] = max(
            self.rounds_arrived[op.id],
            packet.round_index,
        )
        self.engine.log("DecoderCluster",
                        f"round {packet.round_index} of {op.name} arrived "
                        f"(op now has rounds 1..{self.rounds_arrived[op.id]})")

    def on_memory_round(self, op_id: int) -> None:
        """Record an idle/memory round and re-check waiting windows."""
        self.memory_rounds[op_id] += 1
        self.memory_rounds_total += 1
        self.engine.log("DecoderCluster",
                        f"memory round for {self._ops[op_id].name} "
                        f"(idle buffer rounds: {self.memory_rounds[op_id]})")
        for k in range(self.window_count[op_id]):
            self.check_window((op_id, k))

    def prepend_idle_rounds(self, op_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style op when the scheme asks."""
        if (
            round_count <= 0
            or not self._batch_preceding_idle_rounds_by_operation.get(
                op_id,
                False,
            )
        ):
            return
        w = self.windows[(op_id, 0)]
        w.batched_preceding_idle_round_count += round_count

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
            window.t_first_round = self.store.round_complete_tick(
                window.op_id,
                window.start_round,
            )
        if not self._window_data_complete(window):
            return
        if window.t_data_complete is None:
            window.t_data_complete = self.engine.now
        op = self._ops[window.op_id]
        if window.deps_remaining > 0:
            if not window.blocked_logged:
                window.blocked_logged = True
            return
        self._submit_window_decode(key, window, op)

    def _window_data_complete(self, w: Window) -> bool:
        op = self._ops[w.op_id]
        succ_rounds = self._successor_rounds_available(w)
        round_count = self._effective_round_count_for_window(w.op_id, w)
        has_successor = op.has_successor and not self._window_has_closed_boundary(w)
        return self.scheme.data_complete(
            w, rounds_arrived=self.rounds_arrived[w.op_id],
            successor_rounds=succ_rounds, memory_rounds=self.memory_rounds[w.op_id],
            round_count=round_count, has_successor=has_successor,
            operation=self._planning_view(op))

    def _job_desc(self, w: Window, op: Operation) -> str:
        """Human decode-job label."""
        if self._windowed_by_operation[w.op_id]:
            return f"{op.name} W{w.k} [commit {w.commit_lo}-{w.commit_hi}]"
        body_rounds = self._round_count_for_window(op.id, w)
        idle_rounds = w.batched_preceding_idle_round_count
        if idle_rounds:
            effective_rounds = w.n_rounds + idle_rounds
            return (f"{op.name} [whole op, {effective_rounds} rounds: "
                    f"{idle_rounds} idle + {body_rounds} body]")
        return f"{op.name} [whole op, {w.n_rounds} rounds]"

    def _round_count_for_window(self, op_id, window: Optional[Window] = None) -> int:
        return self.lifecycle.round_count_for_window(
            op_id, window,
            fallback_rounds=self.rounds_for(self._ops[op_id])
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
        op = self._ops[window.op_id]
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

    def _deadline_for_window(self, op: Operation, window: Window) -> int:
        """Stamp one window, copying retained start-round provenance when present."""
        if window.t_first_round is None:
            first_round_tick = self.store.round_complete_tick(
                window.op_id,
                window.start_round,
            )
            if first_round_tick is not None:
                window.t_first_round = first_round_tick
        return self.deadline_policy.deadline(
            op,
            window,
            self.engine.now,
            on_reaction_path=(op.id in self.blocking_ops),
        )

    def _submit_window_decode(self, key: tuple, window: Window,
                              op: Operation) -> None:
        """Build the weak job, ask the strategy, and enqueue its submissions."""
        deadline = self._deadline_for_window(op, window)
        window.t_queued = self.engine.now
        job = DecodeJob(
                        op_id=window.op_id, window_id=window.k,
                        n_rounds=(
                            window.n_rounds
                            + window.batched_preceding_idle_round_count
                        ),
                        ready_time=self.engine.now, deadline=deadline,
                        spatial_nodes=self._spatial_nodes(op),
                        payloads=self._assemble_payloads(window),
                        dem=self.window_models.get(key),
                        code=(
                            self._resolved_operations[
                                op.id
                            ].code_geometry.code_name
                        ),
                        window=window, label=self._job_desc(window, op),
                        strong_label=f"strong({op.name} W{window.k})")
        window.queued = True
        for submission in self.strategy.on_window_ready(window, job,
                                                        self.services):
            if submission.job.strong_decode_for is None:
                self.submit_fn(submission.job, submission.delay_ticks)
            else:
                if submission.delay_ticks != 0:
                    raise ValueError(
                        "strong transport delay is owned by the link fabric"
                    )
                self._submit_strong_with_csd(submission.job)

    # ------------------------------------------------------- payload assembly

    def _assemble_payloads(self, w: Window) -> list:
        """Collect this window's payloads, including successor overflow rounds."""
        operation_rounds = self._effective_round_count_for_window(w.op_id, w)
        end_round = min(w.buffer_hi, operation_rounds)
        payloads = []
        window_info = WindowInfo.from_window(w)
        for round_index in range(w.start_round, end_round + 1):
            frags = self.store.fragments(w.op_id, round_index)
            if frags is not None:
                payloads += [
                    self.window_interaction.apply_boundary(
                        w.boundary_in, window_info, fragment,
                        round_index)
                    for fragment in sorted(
                        frags,
                        key=lambda item: stable_identity_order_key(
                            item.patch_id
                        ),
                    )
                ]
        overflow = w.buffer_hi - operation_rounds
        if overflow > 0:
            for successor_id in self.successors.get(w.op_id, []):
                for round_index in range(1, overflow + 1):
                    frags = self.store.fragments(successor_id, round_index)
                    if frags is not None:
                        payloads += [
                            self.window_interaction.apply_boundary(
                                w.boundary_in, window_info, fragment,
                                operation_rounds + round_index)
                            for fragment in sorted(
                                frags,
                                key=lambda item: stable_identity_order_key(
                                    item.patch_id
                                ),
                            )]
        return payloads

    @staticmethod
    def _job_payload_bits(job: DecodeJob) -> Optional[int]:
        sizes = [payload.size_bits for payload in (job.payloads or ())]
        if not all(size is not None for size in sizes):
            return None
        return sum(sizes)

    @staticmethod
    def _job_attribution(job: DecodeJob) -> TrafficAttribution:
        patches = {}
        for payload in job.payloads or ():
            patch_id = payload.patch_id
            patches[stable_identity_order_key(patch_id)] = patch_id
        patch_ids = tuple(patches[key] for key in sorted(patches))
        window = job.window
        if window is None:
            raise RuntimeError("window-scoped transport requires a DecodeJob window")
        return TrafficAttribution(
            operation_id=job.op_id,
            patch_ids=patch_ids,
            window_id=job.window_id,
            round_lo=(
                window.commit_lo
                if window.buffer_lo is None
                else window.buffer_lo
            ),
            round_hi=(
                window.commit_hi
                if window.buffer_hi is None
                else window.buffer_hi
            ),
        )

    @staticmethod
    def _window_attribution(
        window: Window,
        op: Operation,
    ) -> TrafficAttribution:
        return TrafficAttribution(
            operation_id=op.id,
            patch_ids=tuple(sorted(
                op.patches,
                key=stable_identity_order_key,
            )),
            window_id=window.k,
            round_lo=(
                window.commit_lo
                if window.buffer_lo is None
                else window.buffer_lo
            ),
            round_hi=(
                window.commit_hi
                if window.buffer_hi is None
                else window.buffer_hi
            ),
        )

    def _window_link_arrival(
        self,
        path: LinkPath,
        window: Window,
        op: Operation,
    ) -> int:
        reservation = self.links.reserve(
            path,
            payload_bits=None,
            now_ticks=self.engine.now,
            attribution=self._window_attribution(window, op),
        )
        return self.engine.now + reservation.total_delay_ticks

    def _link_arrival(
        self,
        path: LinkPath,
        job: DecodeJob,
        *,
        payload_bits: Optional[int],
    ) -> int:
        reservation = self.links.reserve(
            path,
            payload_bits=payload_bits,
            now_ticks=self.engine.now,
            attribution=self._job_attribution(job),
        )
        return self.engine.now + reservation.total_delay_ticks

    def _submit_strong_with_csd(
        self,
        strong_job: DecodeJob,
        *,
        wsd_arrival_ticks: Optional[int] = None,
    ) -> int:
        csd_arrival_ticks = self._link_arrival(
            LinkPath.CSD,
            strong_job,
            payload_bits=self._job_payload_bits(strong_job),
        )
        ready_ticks = (
            csd_arrival_ticks
            if wsd_arrival_ticks is None
            else max(csd_arrival_ticks, wsd_arrival_ticks)
        )
        self.submit_fn(strong_job, ready_ticks - self.engine.now)
        return csd_arrival_ticks

    def prepare_strong_selection(
        self,
        weak_job: DecodeJob,
        serial_submission,
    ) -> int:
        """Reserve real input legs and return WSD selection-delivery delay."""
        key = (weak_job.op_id, weak_job.window_id)
        pending = self._escalations.peek_key(key)
        if serial_submission is not None:
            wsd_arrival_ticks = self._link_arrival(
                LinkPath.WSD,
                weak_job,
                payload_bits=None,
            )
            self._submit_strong_with_csd(
                serial_submission.job,
                wsd_arrival_ticks=wsd_arrival_ticks,
            )
            return wsd_arrival_ticks - self.engine.now
        if pending is not None:
            return max(0, pending.wsd_arrival_ticks - self.engine.now)
        wsd_arrival_ticks = self._link_arrival(
            LinkPath.WSD,
            weak_job,
            payload_bits=None,
        )
        return wsd_arrival_ticks - self.engine.now

    # ------------------------------------------------------------ strong jobs

    def make_strong_decode_job(self, weak_job: DecodeJob, round_count: int,
                               label: str) -> DecodeJob:
        """Build the two-sided strong re-decode job for an escalated window."""
        key = (weak_job.op_id, weak_job.window_id)
        weak_window = self.windows[key]
        op = self._ops[weak_job.op_id]
        strong_window = self._strong_context_window(weak_window)
        deadline = self._deadline_for_window(op, strong_window)
        dem = None
        if self.syndrome_source is not None:
            dem = self.syndrome_source.strong_window_model_for_operation(
                op, strong_window,
                self._round_count_for_window(op.id, strong_window),
                fault_model_requirement=self._fault_model_requirement(op))
        return DecodeJob(
            op_id=weak_job.op_id, window_id=weak_job.window_id,
            n_rounds=round_count, ready_time=self.engine.now,
            deadline=deadline, label=label, hint="strong",
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
        strong_window.boundary_in = weak_window.boundary_in
        return strong_window

    @staticmethod
    def _strong_context_bounds(window: Window) -> tuple:
        buffer_rounds = max(0, window.buffer_hi - window.commit_hi)
        buffer_lo = max(1, window.commit_lo - buffer_rounds)
        buffer_hi = window.commit_hi + buffer_rounds
        return buffer_lo, window.commit_lo, window.commit_hi, buffer_hi

    # ------------------------------------------- faithful double window (III C)
    # arXiv:2510.25222 Fig. 12: forward slab, weak-chain skip, strong owns
    # the slab, start gated on both weak-determined boundaries. Protocol and
    # seam formalism are documented on Switching (switching.py).

    def defer_strong_escalation(self, weak_job: DecodeJob) -> None:
        """Lay out the forward slab, absorb the windows it covers, and hold
        the strong job until the restart window's weak commit
        (waiting_far_boundary) or, terminally, until every clamped slab
        round is stored (waiting_terminal_data). One strong job per
        escalation; duplicates raise."""
        key = (weak_job.op_id, weak_job.window_id)
        existing_contribution = self.logical_contributions.get(key)
        if (
            self._escalations.peek_key(key) is not None
            or (
                existing_contribution is not None
                and existing_contribution.ownership_kind == "strong_slab"
            )
        ):
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one switching "
                f"event creates exactly one strong job")
        if weak_job.strong_label is None:
            raise RuntimeError(
                f"double-window escalation {key} needs a declared strong label")
        weak_window = self.windows[key]
        op_id, escalated_index = key
        round_count = self._round_count_for_window(op_id, weak_window)
        later_windows = [
            self.windows[(op_id, window_index)]
            for window_index in self.op_windows[op_id]
            if window_index > escalated_index
        ]
        plan = self.window_interaction.plan_strong_region(
            WindowInfo.from_window(weak_window),
            [WindowInfo.from_window(window) for window in later_windows],
            round_count,
        )
        resolved_region = self._resolve_strong_region_plan(
            key, weak_window, later_windows, round_count, plan)
        restart_key = resolved_region.restart_window_key
        if restart_key is None:
            readiness_collision = self._escalations.peek_terminal(op_id)
            readiness_key = op_id
        else:
            readiness_collision = self._escalations.peek_far(restart_key)
            readiness_key = restart_key
        if readiness_collision is not None:
            raise RuntimeError(
                f"readiness index collision for {readiness_key}")

        restart_model = None
        if restart_key is not None:
            proposed_restart = deepcopy(self.windows[restart_key])
            proposed_restart.buffer_lo = plan.restart_buffer_lo
            restart_model = self._build_strong_window_model(
                self._ops[op_id],
                proposed_restart,
                round_count,
                resolved_region.restart_fault_exclusion_ranges,
            )
        slab = Window(
            op_id=key[0], k=key[1],
            commit_lo=plan.commit_lo,
            commit_hi=plan.commit_hi,
            buffer_hi=plan.context_hi,
            buffer_lo=plan.context_lo,
            n_rounds=plan.context_hi - plan.context_lo + 1,
        )
        strong_model = self._build_strong_window_model(
            self._ops[op_id], slab, round_count,
            resolved_region.strong_fault_exclusion_ranges)
        guard_lease = None
        if restart_key is not None:
            guard_lease = (key, "restart-plan-guard")
            self.store.lease(
                guard_lease, resolved_region.restart_read_keys)
        try:
            wsd_arrival_ticks = self._link_arrival(
                LinkPath.WSD,
                weak_job,
                payload_bits=None,
            )
            self._install_strong_slab_ownership(key, resolved_region)
            phase = (
                _EscalationPhase.WAITING_TERMINAL_DATA
                if restart_key is None
                else _EscalationPhase.WAITING_FAR_BOUNDARY
            )
            pending = _PendingEscalation(
                key=key,
                weak_job=weak_job,
                label=weak_job.strong_label,
                resolved_region=resolved_region,
                strong_window=slab,
                strong_model=strong_model,
                wsd_arrival_ticks=wsd_arrival_ticks,
                phase=phase,
            )
            if restart_key is None:
                self._escalations.register_terminal(pending, op_id)
            else:
                self._escalations.register_far(pending, restart_key)
            # The deferred slab is assembled after later weak commits release
            # their leases, so retain every context round until submission.
            self.store.replace((key, "strong"),
                               [(op_id, r) for r in
                                range(plan.context_lo, plan.context_hi + 1)])
            for absorbed_key in resolved_region.absorbed_window_keys:
                self._absorb_window(absorbed_key, restart_key)
            readiness_description = (
                "terminal data"
                if restart_key is None
                else "the far-side weak boundary"
            )
            self.engine.log(
                "DecoderCluster",
                f"{pending.label}: slab rounds {plan.commit_lo}-"
                f"{plan.commit_hi} assigned; weak chain skips "
                f"{len(resolved_region.absorbed_window_keys)} window(s); "
                f"strong start deferred until {readiness_description}",
            )
            if restart_key is None:
                # terminal slab: clamped tail rounds may not be generated yet
                if self.rounds_arrived[op_id] >= plan.context_hi:
                    self._submit_terminal_strong(op_id, pending)
            else:
                self._reslice_restart_window(
                    restart_key,
                    plan.restart_buffer_lo,
                    restart_model,
                    plan.commit_hi,
                    plan.restart_seam_fault_owner,
                )
                self.check_window(restart_key)  # its absorbed dep is gone
        finally:
            if guard_lease is not None:
                self.store.release(guard_lease)

    def _install_strong_slab_ownership(
        self,
        key: tuple,
        resolved_region: _ResolvedStrongRegion,
    ) -> None:
        """Replace every absorbed result owner with one durable slab owner."""
        plan = resolved_region.plan
        replaced_owner_keys = {
            key, *resolved_region.absorbed_window_keys,
        }
        for other_key, contribution in self.logical_contributions.items():
            if other_key[0] != key[0]:
                continue
            overlaps_slab = (
                contribution.commit_lo <= plan.commit_hi
                and plan.commit_lo <= contribution.commit_hi
            )
            if overlaps_slab and other_key not in replaced_owner_keys:
                raise RuntimeError(
                    f"strong slab {key} extent {plan.commit_lo}-"
                    f"{plan.commit_hi} overlaps unabsorbed logical "
                    f"contribution {other_key} extent "
                    f"{contribution.commit_lo}-{contribution.commit_hi}")

        candidate = dict(self.logical_contributions)
        for owner_key in replaced_owner_keys:
            candidate.pop(owner_key, None)
        previous = self.logical_contributions
        self.logical_contributions = candidate
        try:
            self._install_logical_contribution(
                LogicalContribution(
                    owner_key=key,
                    commit_lo=plan.commit_lo,
                    commit_hi=plan.commit_hi,
                    ownership_kind="strong_slab",
                    logical_observables=None,
                )
            )
        except Exception:
            self.logical_contributions = previous
            raise

    def _resolve_strong_region_plan(
        self, key: tuple, weak_window: Window, later_windows: list,
        round_count: int, plan,
    ) -> _ResolvedStrongRegion:
        if not isinstance(plan, StrongRegionPlan):
            raise TypeError(
                f"window interaction must return StrongRegionPlan for "
                f"double-window escalation {key}, got "
                f"{type(plan).__name__}")
        if not (
            1 <= plan.context_lo <= plan.commit_lo
            <= weak_window.commit_lo
            <= weak_window.commit_hi
            <= plan.commit_hi <= plan.context_hi <= round_count
        ):
            raise RuntimeError(
                f"invalid strong-region bounds for {key}: context "
                f"{plan.context_lo}-{plan.context_hi}, commit "
                f"{plan.commit_lo}-{plan.commit_hi}, operation 1-{round_count}")
        if plan.commit_lo < weak_window.commit_lo:
            raise RuntimeError(
                f"strong-region commit for {key} cannot precede the "
                f"escalated window's commit start {weak_window.commit_lo}")
        if plan.commit_lo != weak_window.commit_lo:
            raise RuntimeError(
                f"strong-region commit for {key} must start at the "
                f"escalated window's commit start {weak_window.commit_lo}")
        absorbed = tuple(
            window.key for window in later_windows
            if window.commit_hi <= plan.commit_hi
        )
        crossing = [
            window for window in later_windows
            if window.commit_lo <= plan.commit_hi < window.commit_hi
        ]
        if crossing:
            window = crossing[0]
            raise RuntimeError(
                f"window {window.key} commits {window.commit_lo}-"
                f"{window.commit_hi} across the strong-region edge "
                f"{plan.commit_hi}")
        for absorbed_key in absorbed:
            absorbed_window = self.windows[absorbed_key]
            if absorbed_window.queued or absorbed_window.committed:
                raise RuntimeError(
                    f"cannot absorb window {absorbed_key}: already "
                    f"{'queued' if absorbed_window.queued else 'committed'}")
        expected_restart = next(
            (window.key for window in later_windows
             if window.commit_lo > plan.commit_hi),
            None,
        )
        if expected_restart is None:
            restart_reads = []
            if (plan.restart_buffer_lo is not None
                    or plan.restart_seam_fault_owner is not None):
                raise RuntimeError(
                    f"terminal strong-region plan for {key} cannot define "
                    f"restart seam data")
        else:
            restart = self.windows[expected_restart]
            if restart.commit_lo != plan.commit_hi + 1:
                raise RuntimeError(
                    f"strong-region plan for {key} ends at "
                    f"{plan.commit_hi}, but restart {expected_restart} "
                    f"starts at {restart.commit_lo}; committed regions must "
                    "tile without a gap")
            if (plan.restart_buffer_lo is None
                    or not 1 <= plan.restart_buffer_lo <= restart.commit_lo):
                raise RuntimeError(
                    f"strong-region restart {expected_restart} needs a "
                    f"buffer start in 1-{restart.commit_lo}")
            restart_reads = self._read_keys_for_bounds(
                restart.op_id, plan.restart_buffer_lo, restart.buffer_hi,
                restart)
            if not isinstance(
                    plan.restart_seam_fault_owner, SeamFaultOwner):
                raise RuntimeError(
                    f"strong-region plan for {key} must select a valid "
                    f"restart seam fault owner")

        left_exclusions = (
            ((1, plan.commit_lo - 1),) if plan.commit_lo > 1 else ()
        )
        strong_exclusions = left_exclusions
        restart_exclusions = None
        if expected_restart is not None:
            if (plan.restart_seam_fault_owner
                    is SeamFaultOwner.STRONG_REGION):
                restart_exclusions = ((1, plan.commit_hi),)
            else:
                strong_exclusions = left_exclusions + (
                    (plan.commit_hi + 1, round_count),
                )
                restart_exclusions = left_exclusions

        required_reads = [
            (weak_window.op_id, round_index)
            for round_index in range(plan.context_lo, plan.context_hi + 1)
        ]
        required_reads.extend(restart_reads)
        self._require_retained_payloads(
            required_reads, f"strong-region plan for {key}")
        return _ResolvedStrongRegion(
            plan=plan,
            absorbed_window_keys=absorbed,
            restart_window_key=expected_restart,
            restart_read_keys=tuple(restart_reads),
            strong_fault_exclusion_ranges=strong_exclusions,
            restart_fault_exclusion_ranges=restart_exclusions,
        )

    def _build_strong_window_model(
        self, operation: Operation, window: Window, round_count: int,
        fault_exclusions: tuple,
    ):
        """Build through the historical or explicit multi-range device port."""
        if self.syndrome_source is None:
            return None
        if len(fault_exclusions) <= 1:
            exclusion = fault_exclusions[0] if fault_exclusions else None
            return self.syndrome_source.strong_window_model_for_operation(
                operation, window, round_count,
                fault_model_requirement=self._fault_model_requirement(operation),
                exclude_faults_touching=exclusion,
            )
        if not isinstance(
            self.syndrome_source, MultiFaultExclusionSyndromeDevice,
        ):
            raise TypeError(
                f"device {type(self.syndrome_source).__name__} cannot build "
                "a strong window with multiple fault-exclusion ranges; "
                "implement "
                "strong_window_model_for_operation_with_exclusions"
            )
        builder = (
            self.syndrome_source
            .strong_window_model_for_operation_with_exclusions
        )
        return builder(
            operation, window, round_count,
            fault_model_requirement=self._fault_model_requirement(operation),
            fault_exclusion_ranges=fault_exclusions,
        )

    def _reslice_restart_window(
        self, restart_key: tuple, buffer_lo: int, model,
        slab_hi: int, seam_owner: SeamFaultOwner,
    ) -> None:
        """Install a restart model prepared before plan mutation."""
        restart = self.windows[restart_key]
        restart.buffer_lo = buffer_lo
        restart.n_rounds = restart.buffer_hi - restart.buffer_lo + 1
        self._replace_window_read_refs(restart_key, restart)
        if model is not None:
            self.window_models[restart_key] = model
        self.engine.log("DecoderCluster",
                        f"restart window {restart_key} re-sliced across slab "
                        f"edge {slab_hi} (reads rounds {restart.buffer_lo}-"
                        f"{restart.buffer_hi}; crossing faults owned by "
                        f"{seam_owner.name.lower()})")

    def _absorb_window(self, key: tuple, restart_key: Optional[tuple]) -> None:
        """A slab-covered window is never weak-decoded: count it committed
        with no logical contribution and unhook the restart window."""
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

    def _build_pending_strong_job(
        self,
        pending: _PendingEscalation,
    ) -> DecodeJob:
        """Build a slab job after both boundary conditions are satisfied.

        The slab commits all r_strong rounds and reads one buffer of raw
        context per face, owning nothing that touches pre-slab rounds (see the
        seam formalism on Switching).
        """
        key = pending.key
        weak_job = pending.weak_job
        slab = pending.strong_window
        op = self._ops[key[0]]
        deadline = self._deadline_for_window(op, slab)
        dem = pending.strong_model
        payloads = self._assemble_payloads(slab)
        covered = {payload.round_index for payload in payloads}
        plan = pending.resolved_region.plan
        needed = set(range(plan.context_lo, plan.context_hi + 1))
        if covered != needed:
            raise RuntimeError(
                f"{pending.label}: slab submitted with rounds "
                f"{sorted(covered)} but it needs "
                f"{plan.context_lo}-{plan.context_hi}; a slab may "
                f"only start once every stored block exists (Fig. 12)")
        return DecodeJob(
            op_id=key[0], window_id=key[1],
            n_rounds=slab.n_rounds,
            ready_time=self.engine.now, deadline=deadline,
            label=pending.label, hint="strong",
            spatial_nodes=weak_job.spatial_nodes, code=weak_job.code,
            dem=dem, payloads=payloads,
            attempt=1, window=slab, strong_decode_for=key)

    def _submit_far_strong(
        self,
        far_boundary_key: tuple,
        pending: _PendingEscalation,
    ) -> None:
        strong_job = self._build_pending_strong_job(pending)
        self.services.check_strong_route(pending.weak_job, strong_job)
        self._escalations.take_far(far_boundary_key, pending)
        self._submit_strong_with_csd(
            strong_job,
            wsd_arrival_ticks=pending.wsd_arrival_ticks,
        )
        self.store.release((pending.key, "strong"))
        self.engine.log(
            "DecoderCluster",
            f"{pending.label}: far-side weak boundary determined -> strong "
            "slab submitted",
        )

    def _submit_terminal_strong(
        self,
        operation_id,
        pending: _PendingEscalation,
    ) -> None:
        strong_job = self._build_pending_strong_job(pending)
        self.services.check_strong_route(pending.weak_job, strong_job)
        self._escalations.take_terminal(operation_id, pending)
        self._submit_strong_with_csd(
            strong_job,
            wsd_arrival_ticks=pending.wsd_arrival_ticks,
        )
        self.store.release((pending.key, "strong"))
        self.engine.log(
            "DecoderCluster",
            f"{pending.label}: terminal data complete -> strong slab submitted",
        )

    def _check_deferred_strong_after_commit(self, committed_key: tuple) -> None:
        """The restart window's weak commit is the far boundary of a slab."""
        pending = self._escalations.peek_far(committed_key)
        if pending is not None:
            self._submit_far_strong(committed_key, pending)

    def _check_deferred_strong_after_arrival(self, op_id) -> None:
        """A terminal slab waits for its clamped tail rounds to be stored."""
        pending = self._escalations.peek_terminal(op_id)
        if pending is None:
            return
        if (
            self.rounds_arrived[op_id]
            >= pending.resolved_region.plan.context_hi
        ):
            self._submit_terminal_strong(op_id, pending)

    @property
    def pending_escalations(self) -> dict:
        """Typed deferral phase per escalated window, for tests and metrics."""
        return {
            key: phase.name.lower()
            for key, phase in self._escalations.snapshot_phases().items()
        }

    # ---------------------------------------------------------------- commit

    def on_decode_done(self, job: DecodeJob, res: DecodeResult) -> None:
        """Publish an accepted weak result only after its WDO transfer."""
        window = self.windows[(job.op_id, job.window_id)]
        window.t_done = self.engine.now
        if job.awaiting_strong_result:
            self._commit_decode_done(job, res)
            return
        op = self._ops[job.op_id]
        delivery_ticks = self._window_link_arrival(
            LinkPath.WDO,
            window,
            op,
        )
        self.engine.schedule(
            delivery_ticks - self.engine.now,
            lambda: self._commit_decode_done(job, res),
            label=f"weak result {op.name}W{window.k}->orchestrator",
        )

    def _commit_decode_done(self, job: DecodeJob, res: DecodeResult) -> None:
        """Commit after weak result transport, or provisionally on escalation."""
        key = (job.op_id, job.window_id)
        window = self.windows[key]
        op = self._ops[job.op_id]
        self._commit_window(job, res, key, window, op)
        self.lifecycle.update_committed_round_count(op.id)
        boundary = self.window_interaction.boundary_from_result(res, None)
        if job.awaiting_strong_result:
            self.speculative_recovery.begin(job, boundary)
        final = not job.awaiting_strong_result
        if self.boundary_policy.on_commit(window, final=final):
            self._send_boundary(window, op, boundary)
        else:
            self._held_boundary[key] = (op.id, boundary)    # Held: ship at final
        self.store.release(key)
        self._check_deferred_strong_after_commit(key)
        self.speculative_recovery.after_commit()
        self._finish_operation_if_ready(op)
        self.finish_workload_if_ready()

    def _commit_window(self, job: DecodeJob, res: DecodeResult, key: tuple,
                       window: Window, op: Operation) -> None:
        window.committed = True
        if window.t_done is None:
            window.t_done = self.engine.now
        self.committed_windows.add(key)
        self._committed_per_op[op.id] = self._committed_per_op.get(op.id, 0) + 1
        self.engine.log("DecoderCluster",
                        f"DECODE DONE {op.name} W{window.k} "
                        f"[commit {window.commit_lo}-{window.commit_hi}]")
        existing_contribution = self.logical_contributions.get(key)
        if (
            existing_contribution is None
            or existing_contribution.ownership_kind != "strong_slab"
        ):
            self._install_logical_contribution(
                LogicalContribution(
                    owner_key=key,
                    commit_lo=window.commit_lo,
                    commit_hi=window.commit_hi,
                    ownership_kind="ordinary_window",
                    logical_observables=res.logical_observables,
                )
            )
        if job.awaiting_strong_result:
            self._mark_window_waiting_for_strong(key, op.id)
        if self._escalations.peek_key(key) is None:
            # a deferred slab still needs these rounds; released at submission
            self.store.release((key, "strong"))

    def _install_logical_contribution(
        self,
        contribution: LogicalContribution,
    ) -> None:
        if type(contribution.owner_key) is not tuple \
                or len(contribution.owner_key) != 2:
            raise TypeError(
                "logical contribution owner_key must be a two-item tuple")
        if contribution.ownership_kind not in (
            "ordinary_window",
            "strong_slab",
        ):
            raise ValueError(
                "logical contribution ownership_kind must be "
                "'ordinary_window' or 'strong_slab'")
        if (
            type(contribution.commit_lo) is not int
            or type(contribution.commit_hi) is not int
        ):
            raise TypeError(
                "logical contribution bounds must be exact ints")
        if (
            contribution.commit_lo < 1
            or contribution.commit_hi < contribution.commit_lo
        ):
            raise ValueError(
                f"logical contribution {contribution.owner_key} has invalid "
                f"extent {contribution.commit_lo}-{contribution.commit_hi}")

        logical_observables = contribution.logical_observables
        if logical_observables is not None:
            if type(logical_observables) is not tuple:
                raise TypeError(
                    f"logical contribution {contribution.owner_key} "
                    "logical_observables must be an exact tuple")

        stream_id = contribution.owner_key[0]
        previous = self.logical_contributions.get(contribution.owner_key)
        if previous is not None and (
            previous.commit_lo != contribution.commit_lo
            or previous.commit_hi != contribution.commit_hi
            or previous.ownership_kind != contribution.ownership_kind
        ):
            raise RuntimeError(
                f"logical contribution {contribution.owner_key} cannot "
                f"change ownership from {previous.ownership_kind} "
                f"{previous.commit_lo}-{previous.commit_hi} to "
                f"{contribution.ownership_kind} "
                f"{contribution.commit_lo}-{contribution.commit_hi}")

        for other_key, other in self.logical_contributions.items():
            if other_key == contribution.owner_key \
                    or other_key[0] != stream_id:
                continue
            if (
                contribution.commit_lo <= other.commit_hi
                and other.commit_lo <= contribution.commit_hi
            ):
                raise RuntimeError(
                    f"logical contribution {contribution.owner_key} extent "
                    f"{contribution.commit_lo}-{contribution.commit_hi} "
                    f"overlaps {other_key} extent "
                    f"{other.commit_lo}-{other.commit_hi}")

        if logical_observables is not None:
            observed_arity = len(logical_observables)
            expected_arity = self._observable_arity_by_stream.get(stream_id)
            if (
                expected_arity is not None
                and observed_arity != expected_arity
            ):
                raise ValueError(
                    f"logical contribution {contribution.owner_key} has "
                    f"observable length {observed_arity}; expected "
                    f"{expected_arity} for stream {stream_id!r}")
            if expected_arity is None:
                self._observable_arity_by_stream[stream_id] = observed_arity

        self.logical_contributions[contribution.owner_key] = contribution

    def _logical_observables_for_interval(
        self,
        stream_id,
        commit_lo: int,
        commit_hi: int,
        *,
        boundary_policy: str,
    ) -> Optional[tuple[int, ...]]:
        if boundary_policy not in ("strict", "stream_segment"):
            raise ValueError(
                f"unknown logical contribution boundary policy "
                f"{boundary_policy!r}")
        if commit_lo < 1 or commit_hi < commit_lo:
            raise ValueError(
                f"invalid logical prediction interval "
                f"{commit_lo}-{commit_hi}")

        contributions = sorted(
            (
                contribution
                for key, contribution in self.logical_contributions.items()
                if key[0] == stream_id
                and contribution.commit_lo <= commit_hi
                and contribution.commit_hi >= commit_lo
            ),
            key=lambda contribution: (
                contribution.commit_lo,
                contribution.commit_hi,
                repr(contribution.owner_key),
            ),
        )
        if not contributions:
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} "
                f"{commit_lo}-{commit_hi} has no contribution coverage")

        cursor = commit_lo
        for contribution in contributions:
            covered_lo = max(contribution.commit_lo, commit_lo)
            covered_hi = min(contribution.commit_hi, commit_hi)
            if covered_lo != cursor:
                relation = "overlap" \
                    if covered_lo < cursor else "gap"
                raise RuntimeError(
                    f"logical prediction interval {stream_id!r} "
                    f"{commit_lo}-{commit_hi} has a contribution "
                    f"{relation} at round {cursor}")
            cursor = covered_hi + 1
        if cursor != commit_hi + 1:
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} "
                f"{commit_lo}-{commit_hi} has a contribution gap at "
                f"round {cursor}")

        for contribution in contributions:
            crosses_boundary = (
                contribution.commit_lo < commit_lo
                or contribution.commit_hi > commit_hi
            )
            if not crosses_boundary:
                continue
            if (
                boundary_policy == "stream_segment"
                and contribution.logical_observables is None
            ):
                continue
            if boundary_policy == "stream_segment":
                raise RuntimeError(
                    f"functional logical contribution "
                    f"{contribution.owner_key} crosses stream-segment "
                    f"boundary {commit_lo}-{commit_hi}")
            raise RuntimeError(
                f"logical contribution {contribution.owner_key} crosses "
                f"strict interval boundary {commit_lo}-{commit_hi}")

        if any(
            contribution.logical_observables is None
            for contribution in contributions
        ):
            return None

        arity = len(contributions[0].logical_observables)
        aggregate = [0] * arity
        for contribution in contributions:
            logical_observables = contribution.logical_observables
            if len(logical_observables) != arity:
                raise RuntimeError(
                    f"logical prediction interval {stream_id!r} changed "
                    "observable arity during aggregation")
            for observable_index, bit in enumerate(logical_observables):
                aggregate[observable_index] ^= bit
        return tuple(aggregate)

    def _replace_contribution_prediction(
        self,
        owner_key: tuple,
        logical_observables: tuple[int, ...],
    ) -> None:
        contribution = self.logical_contributions.get(owner_key)
        if contribution is None:
            raise RuntimeError(
                f"result for {owner_key} has no logical contribution owner")
        self._install_logical_contribution(
            LogicalContribution(
                owner_key=contribution.owner_key,
                commit_lo=contribution.commit_lo,
                commit_hi=contribution.commit_hi,
                ownership_kind=contribution.ownership_kind,
                logical_observables=logical_observables,
            )
        )

    def _mark_window_waiting_for_strong(self, key: tuple, op_id: int) -> None:
        if key in self._pending_strong_windows:
            return
        self._pending_strong_windows.add(key)
        self._pending_strong_per_op[op_id] = \
            self._pending_strong_per_op.get(op_id, 0) + 1

    def on_strong_decode_done(self, key: tuple, result: DecodeResult) -> None:
        """Publish an accepted strong result only after its DO transfer."""
        window = self.windows[key]
        op = self._ops[window.op_id]
        delivery_ticks = self._window_link_arrival(
            LinkPath.DO,
            window,
            op,
        )
        self.engine.schedule(
            delivery_ticks - self.engine.now,
            lambda: self._commit_strong_decode_done(key, result),
            label=f"strong result {op.name}W{window.k}->orchestrator",
        )

    def _commit_strong_decode_done(
        self,
        key: tuple,
        result: DecodeResult,
    ) -> None:
        """Finalize a weak-committed window with the delivered strong result.

        Held ships the strong boundary now. Eager delegates a boundary change
        to SpeculativeRecovery, which replays the affected static descendants.
        """
        window = self.windows[key]
        op = self._ops[window.op_id]
        self.op_strong_commit_time[op.id] = max(
            self.op_strong_commit_time.get(op.id, 0), self.engine.now)
        if self.speculative_recovery.complete(key, result):
            return
        if result.logical_observables is not None:
            self._replace_contribution_prediction(
                key,
                result.logical_observables,
            )
        self._resolve_strong_wait(key, op.id)
        if key in self._held_boundary:                        # Held: ship now
            src_op_id, boundary = self._held_boundary.pop(key)
            self._send_boundary(
                window, self._ops[src_op_id],
                self.window_interaction.boundary_from_result(result, boundary))
        self.speculative_recovery.after_commit()
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

    @property
    def speculative_replays(self) -> int:
        return self.speculative_recovery.replay_count

    # -------------------------------------------------------------- handoff

    def _send_boundary(self, window: Window, op: Operation, boundary) -> None:
        """Schedule policy-selected boundary deliveries over the dd link."""
        source_key = (window.op_id, window.k)
        selected_targets = tuple(self.window_interaction.boundary_targets(
            WindowInfo.from_window(window), self._window_infos()))
        if len(set(selected_targets)) != len(selected_targets):
            raise RuntimeError(
                f"window interaction selected duplicate boundary targets "
                f"for source {source_key}: {selected_targets}")
        targets = tuple(
            key for key in selected_targets
            if key not in self.absorbed_windows
        )
        for dep_key in targets:
            if dep_key not in self.windows:
                raise RuntimeError(
                    f"window interaction selected unknown boundary target "
                    f"{dep_key} for source {source_key}")
            target = self.windows[dep_key]
            if source_key not in target.deps:
                raise RuntimeError(
                    f"window interaction selected boundary target {dep_key} "
                    f"for source {source_key}, but it is not a live "
                    f"dependency declared by the window scheme")
            if target.queued or target.committed:
                raise RuntimeError(
                    f"window interaction selected boundary target {dep_key} "
                    f"for source {source_key} after its decode lifecycle "
                    f"started")

        version = self._boundary_versions.get(source_key, 0) + 1
        deliveries = []
        for dep_key in targets:
            delivery_key = (source_key, dep_key)
            delivery_version = \
                self._boundary_delivery_versions.get(delivery_key, 0) + 1
            deliveries.append((dep_key, delivery_key, delivery_version))

        self._boundary_versions[source_key] = version
        self._committed_boundaries[source_key] = boundary
        for dep_key, delivery_key, delivery_version in deliveries:
            self._boundary_delivery_versions[delivery_key] = delivery_version
            reservation = self.links.reserve(
                LinkPath.DD,
                payload_bits=None,
                now_ticks=self.engine.now,
                attribution=self._window_attribution(window, op),
            )
            self.engine.schedule(
                reservation.total_delay_ticks,
                lambda dk=dep_key, so=op.id, bd=boundary,
                       sk=source_key, v=version, dv=delivery_version:
                    self._receive_boundary(dk, so, bd, sk, v, dv),
                label=f"boundary {op.name}W{window.k}->W{dep_key}")

    def _merge_available_boundary(
        self, source_key: tuple, destination: Window, boundary,
    ) -> None:
        """Merge an already-delivered predecessor into a newly built window."""
        delivery_key = (source_key, destination.key)
        delivery = BoundaryDelivery(
            source_key=source_key,
            destination_key=destination.key,
            source_revision=self._boundary_versions.get(source_key, 0),
            delivery_revision=self._boundary_delivery_versions.get(
                delivery_key, 0),
            latest_source_revision=self._boundary_versions.get(source_key, 0),
            latest_delivery_revision=self._boundary_delivery_versions.get(
                delivery_key, 0),
            source_operation_round_count=self.rounds_for(
                self._ops[source_key[0]]),
            dependency_released=True,
            payload=boundary,
        )
        update = self._propose_boundary_update(delivery, destination)
        if update.release_dependency:
            raise RuntimeError(
                f"window interaction released boundary dependency "
                f"{delivery_key} more than once")
        if update.accepted:
            destination.boundary_in = update.state

    def _receive_boundary(self, key: tuple, src_op_id: int,
                          defects: Optional[dict] = None,
                          source_key: Optional[tuple] = None,
                          version: Optional[int] = None,
                          delivery_version: Optional[int] = None) -> None:
        if source_key is None or version is None or delivery_version is None:
            raise RuntimeError("boundary delivery is missing source provenance")
        delivery_key = (source_key, key)
        w = self.windows[key]
        dependency_released = \
            delivery_key in self._released_boundary_dependencies
        delivery = BoundaryDelivery(
            source_key=source_key,
            destination_key=key,
            source_revision=version,
            delivery_revision=delivery_version,
            latest_source_revision=self._boundary_versions.get(source_key, 0),
            latest_delivery_revision=self._boundary_delivery_versions.get(
                delivery_key, 0),
            source_operation_round_count=self.rounds_for(self._ops[src_op_id]),
            dependency_released=dependency_released,
            payload=defects,
        )
        update = self._propose_boundary_update(delivery, w)
        if update.accepted and (w.queued or w.committed):
            raise RuntimeError(
                f"accepted boundary delivery {delivery_key} reached window "
                f"{key} after its decode lifecycle started")
        if update.release_dependency:
            if dependency_released:
                raise RuntimeError(
                    f"window interaction released boundary dependency "
                    f"{delivery_key} more than once")
            if source_key not in w.deps or w.deps_remaining <= 0:
                raise RuntimeError(
                    f"window interaction released unresolved edge "
                    f"{delivery_key}, but it is not a live dependency")
        if update.accepted:
            w.boundary_in = update.state
            if update.release_dependency:
                self._released_boundary_dependencies.add(delivery_key)
                w.deps_remaining -= 1
        self.check_window(key)

    def _propose_boundary_update(
        self, delivery: BoundaryDelivery, destination: Window,
    ) -> BoundaryUpdate:
        """Let the interaction modify an isolated candidate boundary state."""
        try:
            candidate_state = deepcopy(destination.boundary_in)
        except Exception as error:
            raise TypeError(
                f"boundary state for {delivery.destination_key} must support "
                "deep copying before merge_boundary"
            ) from error
        update = self.window_interaction.merge_boundary(
            delivery, WindowInfo.from_window(destination), candidate_state)
        self._validate_boundary_update(delivery, update)
        return update

    @staticmethod
    def _validate_boundary_update(
        delivery: BoundaryDelivery, update,
    ) -> None:
        if not isinstance(update, BoundaryUpdate):
            raise TypeError(
                f"window interaction merge_boundary for "
                f"{delivery.source_key}->{delivery.destination_key} must "
                f"return BoundaryUpdate, got {type(update).__name__}")
        if not update.accepted and update.release_dependency:
            raise RuntimeError(
                f"rejected boundary {delivery.source_key}->"
                f"{delivery.destination_key} cannot release a dependency")

    def _window_infos(self):
        return MappingProxyType({
            key: WindowInfo.from_window(window)
            for key, window in self.windows.items()
        })

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
        op = self._ops.get(op_id)
        if op is not None:
            self._finish_operation_if_ready(op)

    def _finish_operation_if_ready(self, op: Operation) -> None:
        """Deliver an op result once every window is committed, no strong
        redo or speculative ancestor is pending, and the stream is sealed."""
        if op.id in self._finished_ops:
            return
        if self._pending_strong_per_op.get(op.id, 0) > 0:
            return
        if self.speculative_recovery.blocks_finality(op.id):
            return
        predicate = self._finalize_gates.get(op.id)
        if predicate is not None and not predicate(op):
            return
        if (self._committed_per_op.get(op.id, 0) == self.window_count[op.id]
                and self.lifecycle.sealed(op.id)):
            self._finished_ops.add(op.id)
            self._deliver_result(op)
            self.store.free_op(op.id)

    def finish_workload_if_ready(self) -> None:
        if self._workload_complete_sent:
            return
        if (len(self.committed_windows) == self.total_windows
                and not self._pending_strong_windows
                and not self.speculative_recovery.has_finality_blockers
                and not self.lifecycle.unsealed
                and self.on_workload_complete is not None):
            self._workload_complete_sent = True
            self.on_workload_complete()

    def _deliver_result(self, op: Operation) -> None:
        window_keys = [
            (op.id, window_index)
            for window_index in self.op_windows[op.id]
        ]
        commit_lo = min(
            self.windows[key].commit_lo
            for key in window_keys
        )
        commit_hi = max(
            self.windows[key].commit_hi
            for key in window_keys
        )
        logical_observables = self._logical_observables_for_interval(
            op.id,
            commit_lo,
            commit_hi,
            boundary_policy="strict",
        )
        if logical_observables is None:
            self.op_results.pop(op.id, None)
        else:
            self.op_results[op.id] = logical_observables
        result = DecodeResult(
            op.id,
            self.window_count[op.id] - 1,
            logical_observables=logical_observables,
        )
        self.orchestrator.integrate(op, result)

    # -------------------------------------------------------- stream segments

    def release_stream_segments_at_commit(self, stream_id,
                                          committed_round_count: int) -> None:
        """Deliver segment results whose full round range has committed
        (gated the same way as ops: no pending strong may still change it)."""
        for operation in list(self._ops.values()):
            if operation.stream_id != stream_id:
                continue
            if operation.id not in self.blocking_ops:
                continue
            if operation.id in self.lifecycle.segment_results_sent:
                continue
            segment_end = self._stream_segment_end(operation)
            if segment_end is None or segment_end > committed_round_count:
                continue
            if (self.speculative_recovery.blocks_finality(operation.id)
                    or self.speculative_recovery.blocks_stream_segment(
                        stream_id, segment_end)):
                continue
            if self._segment_waits_for_strong(stream_id, segment_end):
                continue
            segment_start = operation.stream_offset + 1
            logical_observables = self._logical_observables_for_interval(
                stream_id,
                segment_start,
                segment_end,
                boundary_policy="stream_segment",
            )
            if logical_observables is None:
                self.op_results.pop(operation.id, None)
            else:
                self.op_results[operation.id] = logical_observables
            self.lifecycle.segment_results_sent.add(operation.id)
            self.orchestrator.integrate(
                operation,
                DecodeResult(
                    operation.id,
                    -1,
                    logical_observables=logical_observables,
                ),
            )

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

    def pending_strong_work_snapshot(self) -> tuple:
        """Snapshot strong slabs assigned but not yet admitted for service."""
        return self._escalations.snapshot_work()

    @property
    def peak_payloads(self) -> int:
        return self.store.peak_payloads

    @property
    def payloads_held(self) -> int:
        return self.store.payloads_held
