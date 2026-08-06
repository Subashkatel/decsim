"""Replay Eager boundaries corrected by a later strong decode.

Retain raw syndrome until the weak cone can restart from that boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .message import EndpointRole, Replay


class _RecoveryState(Enum):
    WAITING_STRONG = auto()
    READY_TO_REPAIR = auto()
    REPLAYING = auto()


@dataclass
class _RecoveryRecord:
    descendants: tuple
    replay_owner: Replay
    blocked_ops: frozenset
    blocked_stream_starts: dict
    weak_boundary: object
    strong_completion: object = None
    strong_result: object = None
    strong_boundary: object = None
    state: _RecoveryState = _RecoveryState.WAITING_STRONG


class SpeculativeRecovery:
    """Own retained data and deterministic descendant replay for Eager mode."""
    def __init__(self, runtime, double_window: bool) -> None:
        self.runtime = runtime
        self._double_window = double_window
        self._records: dict[tuple, _RecoveryRecord] = {}
        self._next_generation: dict[tuple, int] = {}
        self._finality_blockers: dict[int, int] = {}
        self._newly_unblocked_ops: set[int] = set()
        self.replay_count = 0

    def begin(self, job, weak_boundary) -> None:
        """Retain a static descendant cone before its normal leases release."""
        if not getattr(self.runtime.boundary_policy, "speculative", False):
            return
        if self._double_window:
            return
        key = (job.op_id, job.window_id)
        selected = self.runtime.window_interaction.invalidated_windows(
            key, self.runtime._window_infos())
        try:
            descendants = tuple(selected)
        except TypeError as error:
            raise TypeError(
                f"window interaction invalidation for {key} must be "
                f"iterable") from error
        descendants = self._causal_closure(key, descendants)
        self._validate_descendants(key, descendants)
        if not descendants:
            return

        if key in self._records:
            raise RuntimeError(f"recovery for {key} is already live")
        retained_rounds = self._replay_packet_identities(descendants)
        generation = self._next_generation.get(key, 0)
        replay_owner = Replay(key, generation)
        self.runtime.store.register_owner(
            EndpointRole.SB1, replay_owner, retained_rounds)
        blocked_ops = frozenset(item[0] for item in descendants)
        for op_id in blocked_ops:
            self._finality_blockers[op_id] = (
                self._finality_blockers.get(op_id, 0) + 1)
        self._next_generation[key] = generation + 1
        self._records[key] = _RecoveryRecord(
            descendants=descendants,
            replay_owner=replay_owner,
            blocked_ops=blocked_ops,
            blocked_stream_starts=self._blocked_stream_starts(descendants),
            weak_boundary=weak_boundary,
        )

    def complete(self, completion) -> bool:
        """Return True when strong completion is deferred or handled here."""
        key = (completion.request_key.operation_id,
               completion.request_key.window_id)
        result = completion.result
        record = self._records.get(key)
        if record is None:
            return False
        strong_boundary = self.runtime.window_interaction.boundary_from_result(
            result, record.weak_boundary)
        if self.runtime.window_interaction.boundaries_equal(
                record.weak_boundary, strong_boundary):
            self._release_record(key)
            return False

        record.strong_completion = completion
        record.strong_result = result
        record.strong_boundary = strong_boundary
        record.state = _RecoveryState.READY_TO_REPAIR
        self._try_repair()
        return True

    def after_commit(self) -> None:
        """A slow strong result may have waited for the weak cone to quiesce."""
        self._try_repair()
        self._release_completed_replays()
        self._recheck_unblocked_operations()

    def blocks_finality(self, op_id: int) -> bool:
        """Whether a provisional ancestor can still invalidate this op."""
        return self._finality_blockers.get(op_id, 0) > 0

    @property
    def has_finality_blockers(self) -> bool:
        return bool(self._finality_blockers)

    def blocks_stream_segment(self, stream_id, segment_end: int) -> bool:
        """Whether a segment overlaps a stream suffix awaiting recovery."""
        for record in self._records.values():
            invalidated_start = record.blocked_stream_starts.get(stream_id)
            if invalidated_start is not None and segment_end >= invalidated_start:
                return True
        return False

    def _try_repair(self) -> None:
        while True:
            completed = [
                key for key, record in self._records.items()
                if record.state is _RecoveryState.READY_TO_REPAIR
            ]
            for key in sorted(completed):
                record = self._records[key]
                cone = set(record.descendants)
                if any(
                    self.runtime.windows[item].queued
                    and not self.runtime.windows[item].committed
                    for item in cone
                ):
                    continue
                unresolved = self.runtime._pending_strong_windows & cone
                if any(
                    item not in self._records
                    or self._records[item].strong_result is None
                    for item in unresolved
                ):
                    continue
                self._repair(key)
                break
            else:
                return

    def _repair(self, key: tuple) -> None:
        runtime = self.runtime
        record = self._records[key]
        result = record.strong_result
        completion = record.strong_completion
        descendants = record.descendants
        descendant_set = set(descendants)
        source_window = runtime.windows[key]
        source_op = runtime._ops[key[0]]

        if result is not None and result.logical_observables is not None:
            runtime._replace_contribution_prediction(
                key, result.logical_observables)
        corrected_boundary = record.strong_boundary
        runtime._committed_boundaries[key] = corrected_boundary

        superseded = [item for item in descendants if item in self._records]
        replacement_ids = self._replay_packet_identities(descendants)
        covered_owners = [record.replay_owner]
        covered_owners.extend(
            self._records[item].replay_owner for item in superseded)
        replacement_set = set(replacement_ids)
        for owner in covered_owners:
            if not set(runtime.store.owner_packet_identities(
                    EndpointRole.SB1, owner)) <= replacement_set:
                raise RuntimeError(
                    f"replacement Replay for {key} does not cover {owner!r}")
        generation = self._next_generation[key]
        replacement_owner = Replay(key, generation)
        runtime.store.register_owner(
            EndpointRole.SB1, replacement_owner, replacement_ids)
        self._next_generation[key] = generation + 1
        runtime.store.release_owner(EndpointRole.SB1, record.replay_owner)
        record.replay_owner = replacement_owner
        for item in superseded:
            self._release_record(item, source_will_replay=True)
            runtime._resolve_strong_wait(item, item[0])

        for item in descendants:
            self._reset_window(item)
        for stream_id in {item[0] for item in descendants}:
            runtime.lifecycle.recompute_committed_round_count(stream_id)
        for item in descendants:
            self._restore_dependencies(item, key, descendant_set)

        record.state = _RecoveryState.REPLAYING
        runtime._resolve_strong_wait(key, key[0])
        self.replay_count += 1
        runtime._send_boundary(
            source_window, source_op, corrected_boundary,
            source_request_key=completion.request_key)
        runtime.release_stream_segments_at_commit(
            source_op.id,
            runtime.lifecycle.committed_round_counts.get(source_op.id, 0),
        )
        runtime._finish_operation_if_ready(source_op)
        runtime.finish_workload_if_ready()

    def _release_completed_replays(self) -> None:
        """Keep finality blocked until every reset descendant is final again."""
        completed = []
        for key, record in self._records.items():
            if record.state is not _RecoveryState.REPLAYING:
                continue
            descendants = set(record.descendants)
            if not all(self.runtime.windows[item].committed
                       for item in descendants):
                continue
            if self.runtime._pending_strong_windows & descendants:
                continue
            completed.append(key)
        for key in completed:
            self._release_record(key)

    def _release_record(
        self, key: tuple, *, source_will_replay=None,
    ) -> _RecoveryRecord:
        """Release retained data and every finality claim owned by a record."""
        if source_will_replay is None:
            source_will_replay = any(
                key in candidate.descendants
                for source, candidate in self._records.items()
                if source != key
            )
        record = self._records.pop(key)
        self.runtime.store.release_owner(EndpointRole.SB1, record.replay_owner)
        if not source_will_replay:
            self._next_generation.pop(key)
        self._drop_finality_blockers(record.blocked_ops)
        return record

    def _drop_finality_blockers(self, operation_ids) -> None:
        for op_id in operation_ids:
            remaining = self._finality_blockers[op_id] - 1
            if remaining:
                self._finality_blockers[op_id] = remaining
            else:
                del self._finality_blockers[op_id]
                self._newly_unblocked_ops.add(op_id)

    def _recheck_unblocked_operations(self) -> None:
        """Publish operations only after all overlapping ancestors resolve."""
        ready = sorted(op_id for op_id in self._newly_unblocked_ops
                       if not self.blocks_finality(op_id))
        self._newly_unblocked_ops.difference_update(ready)
        for op_id in ready:
            op = self.runtime._ops.get(op_id)
            if op is None:
                continue
            self.runtime.release_stream_segments_at_commit(
                op_id,
                self.runtime.lifecycle.committed_round_counts.get(op_id, 0),
            )
            self.runtime._finish_operation_if_ready(op)
        self.runtime.finish_workload_if_ready()

    def _reset_window(self, key: tuple) -> None:
        runtime = self.runtime
        window = runtime.windows[key]
        # A boundary already in transit belongs to the invalidated decode.
        # Advancing the generation makes its scheduled callback a no-op.
        runtime._boundary_versions[key] = \
            runtime._boundary_versions.get(key, 0) + 1
        for dependency in window.deps:
            delivery_key = (dependency, key)
            runtime._boundary_delivery_versions[delivery_key] = \
                runtime._boundary_delivery_versions.get(delivery_key, 0) + 1
            runtime._released_boundary_dependencies.discard(delivery_key)
        runtime.logical_contributions.pop(key, None)
        runtime.op_results.pop(window.op_id, None)
        if window.committed:
            runtime.committed_windows.discard(key)
            remaining = runtime._committed_per_op.get(window.op_id, 0) - 1
            if remaining > 0:
                runtime._committed_per_op[window.op_id] = remaining
            else:
                runtime._committed_per_op.pop(window.op_id, None)

        window.committed = False
        window.queued = False
        window.blocked_logged = False
        window.t_queued = None
        window.t_dispatch = None
        window.t_done = None
        window.boundary_in = \
            runtime.window_interaction.initial_boundary_state(
                runtime._window_infos()[key])
        runtime._committed_boundaries.pop(key, None)
        runtime._held_boundary.pop(key, None)
        if runtime.store.has_owner(EndpointRole.SB0, key):
            runtime._replace_window_read_refs(key, window)

    def _restore_dependencies(self, key: tuple, root: tuple,
                              replayed: set[tuple]) -> None:
        runtime = self.runtime
        window = runtime.windows[key]
        remaining = 0
        for dependency in window.deps:
            if dependency == root or dependency in replayed:
                remaining += 1
                continue
            if dependency in runtime.committed_windows:
                runtime._merge_available_boundary(
                    dependency,
                    window,
                    runtime._committed_boundaries.get(dependency),
                )
                runtime._released_boundary_dependencies.add((dependency, key))
            else:
                remaining += 1
        window.deps_remaining = remaining

    def _causal_closure(self, root: tuple, selected_roots: tuple) -> tuple:
        """Expand policy-selected roots into a dependency-ordered replay cone."""
        seen = set()
        for key in selected_roots:
            if key in seen:
                raise RuntimeError(
                    f"window interaction invalidation for {root} selected "
                    f"window {key} more than once")
            seen.add(key)
        self._validate_descendants(root, selected_roots)

        closure = set(seen)
        stack = list(seen)
        while stack:
            key = stack.pop()
            for dependent in self.runtime.windows[key].dependents:
                if dependent not in closure:
                    closure.add(dependent)
                    stack.append(dependent)

        ordered = []
        remaining = set(closure)
        while remaining:
            ready = sorted(
                key for key in remaining
                if not (set(self.runtime.windows[key].deps) & remaining)
            )
            if not ready:
                raise RuntimeError(
                    f"window interaction invalidation for {root} contains "
                    f"a dependency cycle")
            ordered.extend(ready)
            remaining.difference_update(ready)
        return tuple(ordered)

    def _validate_descendants(self, root: tuple, descendants) -> None:
        reachable = set()
        stack = list(self.runtime.windows[root].dependents)
        while stack:
            candidate = stack.pop()
            if candidate in reachable:
                continue
            reachable.add(candidate)
            stack.extend(self.runtime.windows[candidate].dependents)

        for key in descendants:
            if key == root:
                raise RuntimeError(
                    f"window interaction invalidation for {root} includes "
                    f"the source window")
            if key not in self.runtime.windows:
                raise RuntimeError(
                    f"window interaction invalidation for {root} selected "
                    f"unknown window {key}")
            if key[0] in self.runtime._finished_ops:
                raise RuntimeError(
                    f"window interaction invalidation for {root} selected "
                    f"window {key} from a finished operation")
            if key not in reachable:
                raise RuntimeError(
                    f"window interaction invalidation for {root} selected "
                    f"{key}, which is not a downstream dependency")
            window = self.runtime.windows[key]
            if window.queued or window.committed:
                raise RuntimeError(
                    f"window interaction invalidation for {root} selected "
                    f"window {key} after its decode lifecycle started")
            reads = self.runtime._read_keys_for_bounds(
                window.op_id, window.start_round, window.buffer_hi, window)
            reads.extend(
                self.runtime._strong_context_read_keys(window, reads))
            self.runtime._require_retained_payloads(
                reads,
                f"window interaction invalidation for {root}, window {key}",
            )

    def _blocked_stream_starts(self, descendants: list[tuple]) -> dict:
        """Earliest invalidated commit round for each descendant decode stream."""
        starts = {}
        for key in descendants:
            window = self.runtime.windows[key]
            starts[window.op_id] = min(
                starts.get(window.op_id, window.commit_lo), window.commit_lo)
        return starts

    def _replay_packet_identities(self, descendants) -> tuple:
        retained = set()
        for key in descendants:
            window = self.runtime.windows[key]
            weak = self.runtime._read_keys_for_bounds(
                window.op_id, window.start_round, window.buffer_hi, window)
            retained.update(weak)
            retained.update(
                self.runtime._strong_context_read_keys(window, weak))
        return tuple(sorted(retained))
