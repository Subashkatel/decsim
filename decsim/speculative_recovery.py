"""Replay support for Eager boundaries corrected by a later strong decode.

This component is for speculative policy: it retains the same raw
syndrome data, waits until the already-running weak cone is quiescent, then
rolls that cone back and submits it again from the corrected boundary.
"""

from __future__ import annotations


class SpeculativeRecovery:
    """Own retained data and deterministic descendant replay for Eager mode."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._pending: dict[tuple, dict] = {}
        self._finality_blockers: dict[int, int] = {}
        self._newly_unblocked_ops: set[int] = set()
        self.replay_count = 0

    def begin(self, job, weak_boundary) -> None:
        """Retain a static descendant cone before its normal leases release."""
        if not getattr(self.runtime.boundary_policy, "speculative", False):
            return
        if getattr(self.runtime.strategy, "double_window", False):
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

        lease_id = ("speculative-recovery", key)
        retained_rounds = set()
        for descendant in descendants:
            window = self.runtime.windows[descendant]
            weak_reads = self.runtime._read_keys_for_bounds(
                window.op_id, window.start_round, window.buffer_hi, window)
            retained_rounds.update(weak_reads)
            retained_rounds.update(
                self.runtime._strong_context_read_keys(window, weak_reads))
        self.runtime.store.lease(lease_id, sorted(retained_rounds))
        blocked_ops = {descendant[0] for descendant in descendants}
        for op_id in blocked_ops:
            self._finality_blockers[op_id] = \
                self._finality_blockers.get(op_id, 0) + 1
        self._pending[key] = {
            "descendants": descendants,
            "lease_id": lease_id,
            "blocked_ops": blocked_ops,
            "blocked_stream_starts": self._blocked_stream_starts(descendants),
            "weak_boundary": weak_boundary,
            "strong_result": None,
            "state": "waiting_strong",
        }

    def complete(self, completion) -> bool:
        """Return True when strong completion is deferred or handled here."""
        key = (completion.request_key.operation_id,
               completion.request_key.window_id)
        result = completion.result
        record = self._pending.get(key)
        if record is None:
            return False
        strong_boundary = self.runtime.window_interaction.boundary_from_result(
            result, record["weak_boundary"])
        if self.runtime.window_interaction.boundaries_equal(
                record["weak_boundary"], strong_boundary):
            self._release_record(key)
            return False

        record["strong_completion"] = completion
        record["strong_result"] = result
        record["strong_boundary"] = strong_boundary
        record["state"] = "ready_to_repair"
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
        for record in self._pending.values():
            invalidated_start = record["blocked_stream_starts"].get(stream_id)
            if (invalidated_start is not None
                    and segment_end >= invalidated_start):
                return True
        return False

    def _try_repair(self) -> None:
        while True:
            repaired = False
            completed = [
                key for key, record in self._pending.items()
                if record["state"] == "ready_to_repair"
            ]
            for key in sorted(completed):
                record = self._pending[key]
                cone = set(record["descendants"])
                if any(
                    self.runtime.windows[item].queued
                    and not self.runtime.windows[item].committed
                    for item in cone
                ):
                    continue
                unresolved = self.runtime._pending_strong_windows & cone
                if any(
                    item not in self._pending
                    or self._pending[item]["strong_result"] is None
                    for item in unresolved
                ):
                    continue
                self._repair(key)
                repaired = True
                break
            if not repaired:
                return

    def _repair(self, key: tuple) -> None:
        runtime = self.runtime
        record = self._pending[key]
        result = record["strong_result"]
        completion = record["strong_completion"]
        descendants = record["descendants"]
        descendant_set = set(descendants)
        source_window = runtime.windows[key]
        source_op = runtime._ops[key[0]]

        if result is not None and result.logical_observables is not None:
            runtime._replace_contribution_prediction(
                key,
                result.logical_observables,
            )
        corrected_boundary = record["strong_boundary"]
        runtime._committed_boundaries[key] = corrected_boundary

        superseded = [
            item for item in descendants
            if item in self._pending
        ]
        for item in superseded:
            self._release_record(item)
            runtime._resolve_strong_wait(item, item[0])

        for item in descendants:
            self._reset_window(item)
        for stream_id in {item[0] for item in descendants}:
            runtime.lifecycle.recompute_committed_round_count(stream_id)
        for item in descendants:
            self._restore_dependencies(item, key, descendant_set)

        self._release_retained_rounds(record)
        record["state"] = "replaying"
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
        for key, record in self._pending.items():
            if record["state"] != "replaying":
                continue
            descendants = set(record["descendants"])
            if not all(self.runtime.windows[item].committed
                       for item in descendants):
                continue
            if self.runtime._pending_strong_windows & descendants:
                continue
            completed.append(key)
        for key in completed:
            self._release_record(key)

    def _release_record(self, key: tuple) -> dict:
        """Release retained data and every finality claim owned by a record."""
        record = self._pending.pop(key)
        self._release_retained_rounds(record)
        for op_id in record["blocked_ops"]:
            remaining = self._finality_blockers[op_id] - 1
            if remaining:
                self._finality_blockers[op_id] = remaining
            else:
                del self._finality_blockers[op_id]
                self._newly_unblocked_ops.add(op_id)
        return record

    def _release_retained_rounds(self, record: dict) -> None:
        lease_id = record.get("lease_id")
        if lease_id is None:
            return
        self.runtime.store.release(lease_id)
        record["lease_id"] = None

    def _recheck_unblocked_operations(self) -> None:
        """Publish operations only after all overlapping ancestors resolve."""
        ready = sorted(
            op_id for op_id in self._newly_unblocked_ops
            if not self.blocks_finality(op_id)
        )
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

    def _causal_closure(
        self, root: tuple, selected_roots: tuple,
    ) -> tuple:
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
            previous = starts.get(window.op_id)
            if previous is None or window.commit_lo < previous:
                starts[window.op_id] = window.commit_lo
        return starts
