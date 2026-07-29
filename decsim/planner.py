"""Rounds policies and deterministic static-window materialization.

Part module: planner.py port with the §5.21 generalizations —
  - every policy result is validated >= 1 (configured values < 1 raise);
  - GateRounds consults Operation.kind first (MEASURE/INJECT -> 1 round,
    MERGE -> merge_steps*d, IDLE/MEMORY -> d) and falls back to the
    len(op.qubits) >= 2 rule for GENERIC ops — the frozen
    behavior since all existing frontends emit GENERIC;
  - TemporalRounds decouples the temporal distance d_m from the spatial d
    (Chamberland–Campbell 2109.02746); constant-round architectures are
    op-kind -> 1 (Zhou/AFT 2406.17653);
  - PerOpRounds is the QLX pass-through adapter (schedule durations ->
    per-op counts).
RunSpec owns resolution and calls the private materializer below.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from .message import (
    Operation,
    OperationPlanningView,
    OperationWindowPlan,
    OpKind,
    ResolvedOperationPlanning,
    Window,
    WindowInfo,
    WindowPlan,
)


@dataclass(frozen=True)
class _ResolvedExecutionPlanSpec:
    """Immutable static plan produced by the root-owned materializer."""

    planned_operation_ids: tuple[int, ...]
    windows: tuple[WindowInfo, ...]
    window_count: tuple[tuple[int, int], ...]
    op_windows: tuple[tuple[int, tuple[int, ...]], ...]
    successors: tuple[tuple[int, tuple[int, ...]], ...]
    spatial_nodes: tuple[tuple[int, int], ...]
    rounds_by_operation: tuple[tuple[int, int], ...]
    code_names: tuple[tuple[int, str], ...]
    windowed_by_operation: tuple[tuple[int, bool], ...]
    batch_preceding_idle_rounds_by_operation: tuple[
        tuple[int, bool], ...
    ]
    total_windows: int
    summary_json: bytes

    def materialize(self) -> WindowPlan:
        """Create fresh runtime-owned windows and containers."""
        windows = {
            info.key: Window(
                op_id=info.op_id,
                k=info.k,
                commit_lo=info.commit_lo,
                commit_hi=info.commit_hi,
                buffer_hi=info.buffer_hi,
                n_rounds=info.n_rounds,
                buffer_lo=info.buffer_lo,
                deps=list(info.deps),
                dependents=list(info.dependents),
                deps_remaining=len(info.deps),
            )
            for info in self.windows
        }
        return WindowPlan(
            windows=windows,
            window_count=dict(self.window_count),
            op_windows={
                operation_id: list(window_indices)
                for operation_id, window_indices in self.op_windows
            },
            successors={
                operation_id: list(successor_ids)
                for operation_id, successor_ids in self.successors
            },
            spatial_nodes=dict(self.spatial_nodes),
            rounds_by_operation=dict(self.rounds_by_operation),
            code_names=dict(self.code_names),
            windowed_by_operation=dict(self.windowed_by_operation),
            batch_preceding_idle_rounds_by_operation=dict(
                self.batch_preceding_idle_rounds_by_operation
            ),
            total_windows=self.total_windows,
            summary=json.loads(self.summary_json),
        )


def _validated(value: int, source: str) -> int:
    if value < 1:
        raise ValueError(f"{source} must give >= 1 round (got {value})")
    return int(value)


def _validate_operation_graph(ops: list[Operation], *,
                              validate_blockers: bool = False,
                              external_blocker_ids=()) -> None:
    """Reject dependency graphs that dictionaries or an empty event queue
    would otherwise hide. Operation IDs are the graph's stable keys."""
    by_id = {}
    for operation in ops:
        if operation.id in by_id:
            raise ValueError(
                f"duplicate operation id {operation.id}: "
                f"{by_id[operation.id].name!r} and {operation.name!r}")
        by_id[operation.id] = operation

    valid_blocker_ids = set(by_id) | set(external_blocker_ids)
    successors = {operation_id: [] for operation_id in by_id}
    indegree = {operation_id: 0 for operation_id in by_id}
    for operation in ops:
        seen_predecessors = set()
        for predecessor_id in operation.predecessors:
            if predecessor_id in seen_predecessors:
                raise ValueError(
                    f"operation {operation.id} lists predecessor "
                    f"{predecessor_id} more than once")
            seen_predecessors.add(predecessor_id)
            if predecessor_id == operation.id:
                raise ValueError(f"operation {operation.id} depends on itself")
            if predecessor_id not in by_id:
                raise ValueError(
                    f"operation {operation.id} has unknown predecessor "
                    f"{predecessor_id}")
            successors[predecessor_id].append(operation.id)
            indegree[operation.id] += 1

        blocker = operation.blocked_by
        if validate_blockers and blocker is not None:
            if blocker == operation.id:
                raise ValueError(
                    f"operation {operation.id} is blocked by itself")
            if blocker not in valid_blocker_ids:
                raise ValueError(
                    f"operation {operation.id} has unknown blocking operation "
                    f"{blocker}")

    ready = [operation_id for operation_id, degree in indegree.items()
             if degree == 0]
    visited = 0
    while ready:
        operation_id = ready.pop()
        visited += 1
        for successor_id in successors[operation_id]:
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)

    if visited != len(by_id):
        cycle_ids = sorted(operation_id for operation_id, degree in indegree.items()
                           if degree > 0)
        raise ValueError(f"operation dependency cycle involving IDs {cycle_ids}")


class FixedRounds:
    """Every operation runs the same number of rounds."""

    def __init__(self, round_count: int):
        self.round_count = _validated(int(round_count), "FixedRounds")

    def rounds_for(self, op, code) -> int:
        return self.round_count


class PerOpRounds:
    """Per-operation round counts with a fallback policy (the QLX adapter)."""

    def __init__(self, rounds_by_op: dict, fallback=None):
        self.rounds_by_op = {
            op_id: _validated(int(r), f"PerOpRounds[{op_id}]")
            for op_id, r in dict(rounds_by_op).items()}
        self.fallback = fallback if fallback is not None else CodeRounds()

    def rounds_for(self, op, code) -> int:
        if op.id in self.rounds_by_op:
            return self.rounds_by_op[op.id]
        return self.fallback.rounds_for(op, code)


class CodeRounds:
    """Use each code model's own round count, optionally scaled."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def rounds_for(self, op, code) -> int:
        base = code.rounds_per_logical_cycle()
        return max(1, int(round(self.scale * base)))


class GateRounds:
    """Lattice-surgery round counts: op-kind aware, GENERIC falls back to
    the qubit-count rule (Horsman 1111.4022 / Litinski 1808.02892)."""

    def __init__(self, merge_steps: int = 2):
        self.merge_steps = _validated(int(merge_steps), "GateRounds.merge_steps")

    def rounds_for(self, op, code) -> int:
        d = code.distance
        kind = getattr(op, "kind", OpKind.GENERIC)
        if kind is OpKind.MEASURE or kind is OpKind.INJECT:
            return 1
        if kind is OpKind.MERGE:
            return self.merge_steps * d
        if kind in (OpKind.IDLE, OpKind.MEMORY):
            return d
        return self.merge_steps * d if len(op.qubits) >= 2 else d


class TemporalRounds:
    """Temporal distance d_m decoupled from spatial d: surgery/merge ops take
    d_m rounds, everything else the base policy (default GateRounds)."""

    def __init__(self, d_m: int, base=None):
        self.d_m = _validated(int(d_m), "TemporalRounds.d_m")
        self.base = base if base is not None else GateRounds()

    def rounds_for(self, op, code) -> int:
        kind = getattr(op, "kind", OpKind.GENERIC)
        if kind is OpKind.MERGE or (kind is OpKind.GENERIC
                                    and len(op.qubits) >= 2):
            return self.d_m
        return self.base.rounds_for(op, code)


def _materialize_execution_plan(
    operations: tuple[OperationPlanningView, ...],
    resolved_operations: tuple[ResolvedOperationPlanning, ...],
    operation_window_plans: tuple[OperationWindowPlan, ...],
) -> _ResolvedExecutionPlanSpec:
    """Materialize exactly the typed scheme ledgers and direct DAG edges."""
    if not (
        len(operations)
        == len(resolved_operations)
        == len(operation_window_plans)
    ):
        raise ValueError("operation planning inputs must have equal lengths")

    windows = {}
    op_windows = {}
    window_count = {}
    successors = {operation.id: [] for operation in operations}
    spatial_nodes = {}
    rounds_by_operation = {}
    code_names = {}
    windowed_by_operation = {}
    batch_idle_by_operation = {}
    plan_by_operation_id = {}

    for operation, resolved, operation_plan in zip(
        operations,
        resolved_operations,
        operation_window_plans,
    ):
        if type(operation) is not OperationPlanningView:
            raise TypeError(
                "operations must contain exact OperationPlanningView values"
            )
        if type(resolved) is not ResolvedOperationPlanning:
            raise TypeError(
                "resolved_operations must contain exact "
                "ResolvedOperationPlanning values"
            )
        if type(operation_plan) is not OperationWindowPlan:
            raise TypeError(
                "operation_window_plans must contain exact "
                "OperationWindowPlan values"
            )
        if (
            operation.id != resolved.operation_id
            or operation.id != operation_plan.operation_id
        ):
            raise ValueError("operation planning inputs must match by position")
        operation_id = operation.id
        if operation_id in plan_by_operation_id:
            raise ValueError(f"duplicate operation id {operation_id}")
        plan_by_operation_id[operation_id] = operation_plan
        window_count[operation_id] = len(operation_plan.windows)
        op_windows[operation_id] = list(range(len(operation_plan.windows)))
        spatial_nodes[operation_id] = resolved.spatial_node_count
        rounds_by_operation[operation_id] = resolved.round_count
        code_names[operation_id] = resolved.code_geometry.code_name
        windowed_by_operation[operation_id] = operation_plan.windowed
        batch_idle_by_operation[operation_id] = (
            operation_plan.batch_preceding_idle_rounds
        )
        for window_index, geometry in enumerate(operation_plan.windows):
            windows[(operation_id, window_index)] = Window(
                op_id=operation_id,
                k=window_index,
                commit_lo=geometry.commit_lo,
                commit_hi=geometry.commit_hi,
                buffer_hi=geometry.buffer_hi,
                n_rounds=geometry.round_count,
                buffer_lo=geometry.buffer_lo,
            )
        for source_index, destination_index in (
            operation_plan.internal_dependencies
        ):
            windows[(operation_id, destination_index)].deps.append(
                (operation_id, source_index)
            )

    for operation in operations:
        destination_plan = plan_by_operation_id[operation.id]
        for predecessor_id in operation.predecessors:
            if predecessor_id not in plan_by_operation_id:
                raise ValueError(
                    f"operation {operation.id} has unknown predecessor "
                    f"{predecessor_id}"
                )
            successors[predecessor_id].append(operation.id)
            predecessor_plan = plan_by_operation_id[predecessor_id]
            for source_index in predecessor_plan.exit_window_indices:
                for destination_index in (
                    destination_plan.entry_window_indices
                ):
                    windows[(operation.id, destination_index)].deps.append(
                        (predecessor_id, source_index)
                    )

    for window_key, window in windows.items():
        window.deps_remaining = len(window.deps)
        for dependency in window.deps:
            windows[dependency].dependents.append(window_key)

    first_resolved = resolved_operations[0] if resolved_operations else None
    summary = {
        "distance": (
            first_resolved.code_geometry.distance
            if first_resolved is not None
            else 0
        ),
        "commit": (
            first_resolved.code_geometry.commit_round_count
            if first_resolved is not None
            else 0
        ),
        "buffer": (
            first_resolved.code_geometry.buffer_round_count
            if first_resolved is not None
            else 0
        ),
        "rounds_per_op": (
            first_resolved.round_count if first_resolved is not None else 0
        ),
        "windows_per_op": (
            window_count[first_resolved.operation_id]
            if first_resolved is not None
            else 0
        ),
    }
    return _ResolvedExecutionPlanSpec(
        planned_operation_ids=tuple(
            operation.id for operation in operations
        ),
        windows=tuple(
            WindowInfo.from_window(windows[key])
            for key in sorted(windows)
        ),
        window_count=tuple(sorted(window_count.items())),
        op_windows=tuple(
            (operation_id, tuple(window_indices))
            for operation_id, window_indices in sorted(op_windows.items())
        ),
        successors=tuple(
            (operation_id, tuple(successor_ids))
            for operation_id, successor_ids in sorted(successors.items())
        ),
        spatial_nodes=tuple(sorted(spatial_nodes.items())),
        rounds_by_operation=tuple(sorted(rounds_by_operation.items())),
        code_names=tuple(sorted(code_names.items())),
        windowed_by_operation=tuple(
            sorted(windowed_by_operation.items())
        ),
        batch_preceding_idle_rounds_by_operation=tuple(
            sorted(batch_idle_by_operation.items())
        ),
        total_windows=len(windows),
        summary_json=json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
