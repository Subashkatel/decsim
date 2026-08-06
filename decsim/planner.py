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
import math

from .config import us
from .message import (
    EndpointRole,
    Operation,
    OperationPlanningView,
    OperationWindowPlan,
    OpKind,
    PotentialStrong,
    ResolvedCodeGeometry,
    ResolvedOperationPlanning,
    ResolvedPatchPlanning,
    Window,
    WindowPlan,
    same_stable_identity,
    stable_identity_bytes,
)


@dataclass(frozen=True)
class _RunPlan:
    """Derived values consumed by one simulator run."""

    code_geometry: ResolvedCodeGeometry
    resolved_operations: tuple[ResolvedOperationPlanning, ...]
    resolved_patches: tuple[ResolvedPatchPlanning, ...]
    round_ticks: int
    execution: WindowPlan
    buffering: _SyndromeBufferingPlan


@dataclass(frozen=True)
class _SyndromeBufferingPlan:
    sb0_owners: tuple
    potential_owners: tuple
    capacity_rows: tuple


def _plan_syndrome_buffering(execution, *, retain_strong_context, double_window,
                             has_open_ended_dynamic_streams=False):
    def read_keys(operation_id, lower, upper):
        round_count = execution.rounds_by_operation[operation_id]
        keys = [(operation_id, index)
                for index in range(lower, min(upper, round_count) + 1)]
        for successor_id in execution.successors.get(operation_id, ()):
            overflow = min(upper - round_count,
                           execution.rounds_by_operation[successor_id])
            keys += [(successor_id, index) for index in range(1, overflow + 1)]
        return tuple(keys)

    weak_owners, potential_owners = [], []
    lower_witnesses = {role: () for role in EndpointRole}
    sufficient = {role: set() for role in EndpointRole}
    for operation_id, indices in execution.op_windows.items():
        windows = [execution.windows[(operation_id, index)] for index in indices]
        round_count = execution.rounds_by_operation[operation_id]
        for window in windows:
            key = (operation_id, window.k)
            weak = read_keys(operation_id, window.start_round, window.buffer_hi)
            weak_owners.append((key, weak))
            sufficient[EndpointRole.SB0].update(weak)
            if len(weak) > len(lower_witnesses[EndpointRole.SB0]):
                lower_witnesses[EndpointRole.SB0] = weak
            if not retain_strong_context:
                continue
            buffer_rounds = max(0, window.buffer_hi - window.commit_hi)
            commit_hi = window.commit_hi
            if double_window:
                commit_hi = min(
                    round_count, window.commit_hi + 2 * buffer_rounds)
            potential = read_keys(
                operation_id, max(1, window.commit_lo - buffer_rounds),
                commit_hi + buffer_rounds)
            owner = PotentialStrong(key)
            potential_owners.append((owner, potential))
            sufficient[EndpointRole.SB1].update(potential)
            arrived = tuple(identity for identity in potential
                            if identity[0] != operation_id
                            or identity[1] <= window.buffer_hi)
            if len(arrived) > len(lower_witnesses[EndpointRole.SB1]):
                lower_witnesses[EndpointRole.SB1] = arrived
    rows = tuple((role, lower_witnesses[role], None if
                  has_open_ended_dynamic_streams else tuple(sorted(
                      sufficient[role], key=stable_identity_bytes)))
                 for role in EndpointRole)
    return _SyndromeBufferingPlan(
        tuple(weak_owners), tuple(potential_owners), rows)


def _validated(value: int, source: str) -> int:
    if value < 1:
        raise ValueError(f"{source} must give >= 1 round (got {value})")
    return int(value)


def _validate_operation_graph(ops: list[Operation], *,
                              validate_blockers: bool = False,
                              external_blocker_ids=(),
                              dependency_field: str = "predecessors") -> None:
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
        for predecessor_id in getattr(operation, dependency_field):
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


def _validate_workload_identity(ops, decode_ops, dynamic_streams) -> None:
    """Protect runtime mappings from ambiguous operation and stream keys."""
    groups = (("ops", ops), ("decode_ops", decode_ops),
              ("dynamic_streams", dynamic_streams))
    operation_by_id = {}
    roles_by_object = {}
    for role, operations in groups:
        seen_ids = set()
        for operation in operations:
            if type(operation) is not Operation:
                raise TypeError(f"{role} entries must be exact Operation values")
            if operation.id in seen_ids:
                raise ValueError(
                    f"operation id {operation.id} appears more than once in {role}")
            seen_ids.add(operation.id)
            prior = operation_by_id.get(operation.id)
            if prior is not None and prior is not operation:
                raise ValueError(
                    f"operation id {operation.id} belongs to distinct objects "
                    "across workload roles")
            operation_by_id[operation.id] = operation
            roles = roles_by_object.setdefault(id(operation), set())
            roles.add(role)
            if "dynamic_streams" in roles and len(roles) > 1:
                other_role = "ops" if "ops" in roles else "decode_ops"
                raise ValueError(
                    f"operation id {operation.id} cannot appear in both "
                    f"{other_role} and dynamic_streams")
    static_owners = tuple(decode_ops)
    dynamic_owners = tuple(dynamic_streams)
    for operation in ops:
        if operation.stream_id is None:
            if not operation.emits_detector_data:
                continue
            if decode_ops and not any(
                owner is operation for owner in static_owners
            ):
                raise ValueError(
                    f"operation {operation.id} must share static decode membership")
            continue
        owners = dynamic_owners if dynamic_streams else static_owners
        owner = next((candidate for candidate in owners if same_stable_identity(
            candidate.id, operation.stream_id)), None) if owners else (
            operation if same_stable_identity(operation.stream_id, operation.id)
            else None)
        if owner is None:
            raise ValueError(
                f"operation {operation.id} stream_id {operation.stream_id} "
                "does not name a declared stream owner")


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
            op_id: self._validated(int(r), op_id)
            for op_id, r in dict(rounds_by_op).items()}
        self.fallback = fallback if fallback is not None else CodeRounds()

    def rounds_for(self, op, code) -> int:
        if op.id in self.rounds_by_op:
            return self.rounds_by_op[op.id]
        return self.fallback.rounds_for(op, code)

    @staticmethod
    def _validated(value: int, operation_id) -> int:
        if value < 0:
            raise ValueError(
                f"PerOpRounds[{operation_id}] must give >= 0 rounds "
                f"(got {value})")
        return value


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


def _plan_execution(
    *,
    operations: tuple[OperationPlanningView, ...],
    planned_operation_ids: tuple[int, ...],
    code,
    layout,
    scheme,
    rounds_policy,
    fallback_round_us: float,
    retain_strong_context: bool,
    double_window: bool,
    has_open_ended_dynamic_streams: bool = False,
) -> _RunPlan:
    """Resolve geometry, cadence, and windows once for one runtime code."""
    round_us = code.round_period_us()
    if round_us is None:
        round_us = fallback_round_us
    if type(round_us) not in (int, float) or not math.isfinite(round_us):
        raise ValueError("resolved round_us must be a finite built-in number")
    round_ticks = us(round_us)
    if round_ticks < 1:
        raise ValueError("resolved round cadence must be at least one tick")

    leading, trailing = code.buffering_floor()
    patch_count_by_id = {
        operation.id: max(
            1,
            len(operation.patches or operation.qubits),
        )
        for operation in operations
    }
    if len(patch_count_by_id) != len(operations):
        raise ValueError("operation planning ids must be unique")
    base_nodes = {
        count: code.spatial_nodes(count)
        for count in {1, *patch_count_by_id.values()}
    }
    geometry = ResolvedCodeGeometry(
        code_name=code.name,
        distance=code.distance,
        commit_round_count=code.commit_rounds(),
        buffer_round_count=code.buffer_rounds(),
        minimum_leading_buffer_round_count=leading,
        minimum_trailing_buffer_round_count=trailing,
        one_patch_spatial_node_count=base_nodes[1],
        buffer_floor_override_active=code.buffer_floor_override_active(),
    )
    scheme.validate_buffer(geometry)

    resolved = []
    patches_by_key = {}
    for operation in operations:
        if layout.code_for_op(operation) is not code:
            raise ValueError(
                f"layout operation {operation.id} selected a code different "
                "from the resolved run code"
            )
        resolved.append(ResolvedOperationPlanning(
            operation_id=operation.id,
            code_geometry=geometry,
            round_count=rounds_policy.rounds_for(operation, code),
            round_ticks=round_ticks,
            spatial_node_count=layout.spatial_nodes_for(
                operation,
                base_spatial_node_count=base_nodes[
                    patch_count_by_id[operation.id]
                ],
            ),
        ))
        patch_ids = operation.patches or operation.qubits or (0,)
        for patch_id in patch_ids:
            patches_by_key.setdefault(stable_identity_bytes(patch_id), patch_id)

    patches = []
    for patch_id in patches_by_key.values():
        if layout.code_for_patch(patch_id) is not code:
            raise ValueError(
                f"layout patch {patch_id!r} selected a code different from "
                "the resolved run code"
            )
        patches.append(ResolvedPatchPlanning(
            patch_identity=patch_id,
            code_geometry=geometry,
            round_ticks=round_ticks,
            spatial_node_count=layout.patch_spatial_nodes_for(
                patch_id,
                base_spatial_node_count=base_nodes[1],
            ),
        ))

    view_by_id = {operation.id: operation for operation in operations}
    resolved_by_id = {
        operation.operation_id: operation for operation in resolved
    }
    try:
        planned_views = tuple(view_by_id[op_id] for op_id in planned_operation_ids)
        planned_resolved = tuple(
            resolved_by_id[op_id] for op_id in planned_operation_ids
        )
    except KeyError as error:
        raise ValueError(f"unknown planned operation id {error.args[0]}") from error
    if any(operation.round_count < 1 for operation in planned_resolved):
        raise ValueError("decode owners must have at least one round")
    _validate_operation_graph(
        list(planned_views),
        dependency_field="decoder_boundary_predecessors",
    )
    window_ledgers = tuple(
        scheme.plan_operation(
            operation.operation_id,
            operation.round_count,
            commit_round_count=geometry.commit_round_count,
            buffer_round_count=geometry.buffer_round_count,
        )
        for operation in planned_resolved
    )
    execution = _materialize_execution_plan(
        planned_views, planned_resolved, window_ledgers)
    return _RunPlan(
        code_geometry=geometry,
        resolved_operations=tuple(resolved),
        resolved_patches=tuple(patches),
        round_ticks=round_ticks,
        execution=execution,
        buffering=_plan_syndrome_buffering(
            execution,
            retain_strong_context=retain_strong_context,
            double_window=double_window,
            has_open_ended_dynamic_streams=has_open_ended_dynamic_streams),
    )


def _materialize_execution_plan(
    operations: tuple[OperationPlanningView, ...],
    resolved_operations: tuple[ResolvedOperationPlanning, ...],
    operation_window_plans: tuple[OperationWindowPlan, ...],
) -> WindowPlan:
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
        for predecessor_id in operation.decoder_boundary_predecessors:
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

    return WindowPlan(
        windows=windows,
        window_count=window_count,
        op_windows=op_windows,
        successors=successors,
        spatial_nodes=spatial_nodes,
        rounds_by_operation=rounds_by_operation,
        code_names=code_names,
        windowed_by_operation=windowed_by_operation,
        batch_preceding_idle_rounds_by_operation=batch_idle_by_operation,
        total_windows=len(windows),
    )
