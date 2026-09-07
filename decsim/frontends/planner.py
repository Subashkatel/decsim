"""The plan of one run: cadence, geometry, windows and buffer holds.

Built once before the engine starts, from the workload the frontends
lowered, the code card, the layout, the windowing scheme and the round
policy. Planning is build time, off the reaction path; the window manager
and the buffers read the plan and never change it.
"""

import dataclasses
import math
from typing import Callable, Optional

import decsim.config as config
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.program as program_records
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """Derived values consumed by one simulator run."""

    code_geometry: program_records.ResolvedCodeGeometry
    resolved_operations: tuple[program_records.ResolvedOperationPlanning, ...]
    resolved_patches: tuple[program_records.ResolvedPatchPlanning, ...]
    round_ticks: int
    execution: window_records.WindowPlan
    buffering: "SyndromeBufferingPlan"


@dataclasses.dataclass(frozen=True)
class SyndromeBufferingPlan:
    """The holds every window places on the stores, and the store floors.

    A hold names the rounds a consumer keeps alive. The minimum is the
    longest single hold; the sufficient set is the union of every hold,
    None when an open-ended dynamic stream makes it unbounded.
    """

    weak_holds: tuple
    potential_holds: tuple
    minimum_live_rounds: tuple
    sufficient_live_rounds: Optional[tuple]
    sb1_minimum_live_rounds: tuple
    sb1_sufficient_live_rounds: Optional[tuple]


def plan_execution(
    *,
    operations: tuple[program_records.OperationPlanningView, ...],
    planned_operation_ids: tuple[int, ...],
    code,
    layout,
    scheme,
    rounds_policy,
    fallback_round_microseconds: float,
    retain_strong_context: bool,
    double_window: bool,
    has_open_ended_dynamic_streams: bool = False,
) -> RunPlan:
    """Resolve cadence, geometry and windows once for one runtime code."""
    round_ticks = _resolve_round_ticks(code, fallback_round_microseconds)
    _check_geometry_counts(code)
    patch_count_by_id = _patch_count_by_operation_id(operations)
    base_nodes = _base_nodes_by_patch_count(code, patch_count_by_id)
    geometry = _resolve_geometry(code, base_nodes[1])
    scheme.validate_buffer(geometry)
    resolved = []
    patches_by_key = {}
    for operation in operations:
        patch_count = patch_count_by_id[operation.id]
        planning = _resolve_operation(
            operation,
            code,
            layout,
            rounds_policy,
            geometry,
            round_ticks,
            base_nodes[patch_count],
        )
        resolved.append(planning)
        _note_patches(operation, patches_by_key)
    patches = _resolve_patches(
        patches_by_key, code, layout, geometry, round_ticks, base_nodes[1]
    )
    planned_views, planned_resolved = _planned(
        operations, resolved, planned_operation_ids
    )
    execution = _plan_windows(planned_views, planned_resolved, scheme, geometry)
    buffering = _plan_syndrome_buffering(
        execution,
        retain_strong_context=retain_strong_context,
        double_window=double_window,
        has_open_ended_dynamic_streams=has_open_ended_dynamic_streams,
    )
    return RunPlan(
        code_geometry=geometry,
        resolved_operations=tuple(resolved),
        resolved_patches=tuple(patches),
        round_ticks=round_ticks,
        execution=execution,
        buffering=buffering,
    )


def check_operation_graph(
    operations: list[program_records.Operation],
    *,
    validate_blockers: bool = False,
    external_blocker_ids=(),
    dependency_field: str = "predecessors",
) -> None:
    """Refuse a dependency graph a dictionary or an empty queue would hide.

    Operation ids are the graph's stable keys: no id twice, no edge to an
    unknown or to itself, no cycle; a blocker, when checked, names a
    known operation.
    """
    by_id = _index_by_id(operations)
    external_ids = set(external_blocker_ids)
    valid_blocker_ids = set(by_id) | external_ids
    successors = {}
    indegree = {}
    for operation_id in by_id:
        successors[operation_id] = []
        indegree[operation_id] = 0
    for operation in operations:
        predecessor_ids = getattr(operation, dependency_field)
        _check_predecessors(operation, predecessor_ids, by_id, successors)
        indegree[operation.id] = len(predecessor_ids)
        if validate_blockers:
            _check_blocker(operation, valid_blocker_ids)
    _check_acyclic(by_id, successors, indegree)


def check_workload_identity(operations, decode_operations, dynamic_streams):
    """Refuse operation and stream keys the runtime maps could confuse.

    An id names one object across the three roles, an operation is never
    both a stream owner and a workload operation, and every stream_id
    names a declared owner.
    """
    groups = (
        ("ops", operations),
        ("decode_ops", decode_operations),
        ("dynamic_streams", dynamic_streams),
    )
    operation_by_id = {}
    roles_by_object = {}
    for role, members in groups:
        _note_role(role, members, operation_by_id, roles_by_object)
    static_owners = tuple(decode_operations)
    dynamic_owners = tuple(dynamic_streams)
    owners = static_owners
    if dynamic_streams:
        owners = dynamic_owners
    for operation in operations:
        if operation.stream_id is None:
            _check_static_membership(operation, static_owners)
            continue
        owner = _stream_owner(operation, owners)
        if owner is None:
            raise ValueError(
                f"operation {operation.id} stream_id {operation.stream_id} "
                "does not name a declared stream owner"
            )


def _resolve_round_ticks(code, fallback_round_microseconds: float) -> int:
    """The cadence in ticks: the code card's period, else the run's."""
    round_microseconds = code.round_period_us()
    if round_microseconds is None:
        round_microseconds = fallback_round_microseconds
    round_microseconds = float(round_microseconds)
    if not math.isfinite(round_microseconds):
        raise ValueError("resolved round_us must be a finite real number")
    round_ticks = config.microseconds_to_ticks(round_microseconds)
    if round_ticks < 1:
        raise ValueError("resolved round cadence must be at least one tick")
    return round_ticks


def _check_geometry_counts(code) -> None:
    """A zero or fractional geometry never terminates; refuse the card."""
    commit_round_count = code.commit_rounds()
    buffer_round_count = code.buffer_rounds()
    counts = (
        ("distance", code.distance, 1),
        ("commit_round_count", commit_round_count, 1),
        ("buffer_round_count", buffer_round_count, 0),
    )
    for label, value, minimum in counts:
        value_type = type(value)
        if value_type is not int or value < minimum:
            raise TypeError(
                f"{label} must be an int >= {minimum}; got {value!r}"
            )


def _patch_count_by_operation_id(operations) -> dict:
    counts = {}
    for operation in operations:
        patches = operation.patches or operation.qubits
        counts[operation.id] = max(1, len(patches))
    return counts


def _base_nodes_by_patch_count(code, patch_count_by_id: dict) -> dict:
    """The code's spatial node count for one patch and for every used count."""
    counts = {1, *patch_count_by_id.values()}
    base_nodes = {}
    for count in counts:
        base_nodes[count] = code.spatial_nodes(count)
    return base_nodes


def _resolve_geometry(code, one_patch_node_count: int):
    leading, trailing = code.buffering_floor()
    commit_round_count = code.commit_rounds()
    buffer_round_count = code.buffer_rounds()
    return program_records.ResolvedCodeGeometry(
        code_name=code.name,
        distance=code.distance,
        commit_round_count=commit_round_count,
        buffer_round_count=buffer_round_count,
        minimum_leading_buffer_round_count=leading,
        minimum_trailing_buffer_round_count=trailing,
        one_patch_spatial_node_count=one_patch_node_count,
        window_floor_justification=code.window_floor_justification,
    )


def _resolve_operation(
    operation, code, layout, rounds_policy, geometry, round_ticks, base_nodes
) -> program_records.ResolvedOperationPlanning:
    operation_code = layout.code_for_op(operation)
    if operation_code is not code:
        raise ValueError(
            f"layout operation {operation.id} selected a code different "
            "from the resolved run code"
        )
    round_count = rounds_policy.rounds_for(operation, code)
    spatial_node_count = layout.spatial_nodes_for(
        operation, base_spatial_node_count=base_nodes
    )
    return program_records.ResolvedOperationPlanning(
        operation_id=operation.id,
        code_geometry=geometry,
        round_count=round_count,
        round_ticks=round_ticks,
        spatial_node_count=spatial_node_count,
    )


def _note_patches(operation, patches_by_key: dict) -> None:
    """Record each patch once, under its stable identity."""
    patch_ids = operation.patches or operation.qubits
    if not patch_ids:
        patch_ids = (0,)
    for patch_id in patch_ids:
        key = identity_records.stable_identity_bytes(patch_id)
        patches_by_key.setdefault(key, patch_id)


def _resolve_patches(
    patches_by_key, code, layout, geometry, round_ticks, base_nodes
) -> list[program_records.ResolvedPatchPlanning]:
    patches = []
    for patch_id in patches_by_key.values():
        patch_code = layout.code_for_patch(patch_id)
        if patch_code is not code:
            raise ValueError(
                f"layout patch {patch_id!r} selected a code different from "
                "the resolved run code"
            )
        spatial_node_count = layout.patch_spatial_nodes_for(
            patch_id, base_spatial_node_count=base_nodes
        )
        planning = program_records.ResolvedPatchPlanning(
            patch_identity=patch_id,
            code_geometry=geometry,
            round_ticks=round_ticks,
            spatial_node_count=spatial_node_count,
        )
        patches.append(planning)
    return patches


def _planned(operations, resolved, planned_operation_ids) -> tuple:
    """The views and resolved plannings of the operations that get windows."""
    view_by_id = {}
    for operation in operations:
        view_by_id[operation.id] = operation
    resolved_by_id = {}
    for planning in resolved:
        resolved_by_id[planning.operation_id] = planning
    planned_views = []
    planned_resolved = []
    for operation_id in planned_operation_ids:
        planned_views.append(view_by_id[operation_id])
        planned_resolved.append(resolved_by_id[operation_id])
    for planning in planned_resolved:
        if planning.round_count < 1:
            raise ValueError("decode owners must have at least one round")
    return tuple(planned_views), tuple(planned_resolved)


def _plan_windows(
    planned_views, planned_resolved, scheme, geometry
) -> window_records.WindowPlan:
    check_operation_graph(
        list(planned_views), dependency_field="decoder_boundary_predecessors"
    )
    window_ledgers = []
    for planning in planned_resolved:
        ledger = scheme.plan_operation(
            planning.operation_id,
            planning.round_count,
            commit_round_count=geometry.commit_round_count,
            buffer_round_count=geometry.buffer_round_count,
        )
        window_ledgers.append(ledger)
    return _materialize_execution_plan(
        planned_views, planned_resolved, tuple(window_ledgers)
    )


def _plan_syndrome_buffering(
    execution,
    *,
    retain_strong_context: bool,
    double_window: bool,
    has_open_ended_dynamic_streams: bool = False,
) -> SyndromeBufferingPlan:
    """Plan logical holds over one upstream round allocation.

    Weak and possible-strong consumers may overlap, but overlapping holds
    do not create another physical packet allocation, so the sufficient
    witness is the union of round identities, not a sum of ledgers.
    """
    weak = _HoldSet()
    strong = _HoldSet()
    for operation_id, indices in execution.op_windows.items():
        for index in indices:
            window = execution.windows[(operation_id, index)]
            _hold_window(execution, operation_id, window, weak, double_window)
            _hold_strong_context(
                execution,
                operation_id,
                window,
                strong,
                retain_strong_context,
                double_window,
            )
    weak_sufficient = weak.sufficient_live_rounds(
        has_open_ended_dynamic_streams
    )
    strong_sufficient = strong.sufficient_live_rounds(
        has_open_ended_dynamic_streams
    )
    return SyndromeBufferingPlan(
        tuple(weak.holds),
        tuple(strong.holds),
        weak.minimum_live_rounds,
        weak_sufficient,
        strong.minimum_live_rounds,
        strong_sufficient,
    )


def _hold_window(
    execution, operation_id, window, weak: "_HoldSet", double_window: bool
) -> None:
    """The window's Buffer 0 holds: its weak read, its potential restart read.

    The weak decode reads the window from its start to its buffer.
    """
    key = (operation_id, window.k)
    round_keys = _read_keys(
        execution, operation_id, window.start_round, window.buffer_hi
    )
    weak.add(key, round_keys, round_keys)
    _hold_restart_reads(execution, operation_id, window, weak, double_window)


def _hold_restart_reads(
    execution, operation_id, window, weak: "_HoldSet", double_window: bool
) -> None:
    """Under the double window, a bounded window keeps one buffer before it.

    An escalation of an earlier window may re-slice this window as its
    restart window, whose weak decode re-reads one buffer into the
    strong region (Toshio 2510.25222 Sec. III C). The rounds stay in
    Buffer 0 past this window's own request and landing, until the
    window before it commits (round_retention.release_restart_reads).
    """
    if not double_window:
        return
    if not _has_same_operation_dependency(window, operation_id):
        return
    look_ahead = window.buffer_hi - window.commit_hi
    buffer_rounds = max(0, look_ahead)
    lower_start = window.commit_lo - buffer_rounds
    lower = max(1, lower_start)
    round_keys = _read_keys(execution, operation_id, lower, window.buffer_hi)
    owner = decoding_records.PotentialRestart((operation_id, window.k))
    weak.add(owner, round_keys, round_keys)


def _has_same_operation_dependency(window, operation_id) -> bool:
    """Whether an earlier window of the same operation bounds this one."""
    for dependency in window.deps:
        if dependency[0] == operation_id:
            return True
    return False


def _hold_strong_context(
    execution,
    operation_id,
    window,
    strong: "_HoldSet",
    retain_strong_context: bool,
    double_window: bool,
) -> None:
    """A possible strong redo reads one buffer of context on each side."""
    if not retain_strong_context:
        return
    round_count = execution.rounds_by_operation[operation_id]
    look_ahead = window.buffer_hi - window.commit_hi
    buffer_rounds = max(0, look_ahead)
    commit_hi = window.commit_hi
    if double_window:
        extended = window.commit_hi + 2 * buffer_rounds
        commit_hi = min(round_count, extended)
    context_start = window.commit_lo - buffer_rounds
    lower = max(1, context_start)
    upper = commit_hi + buffer_rounds
    potential = _read_keys(execution, operation_id, lower, upper)
    owner = decoding_records.PotentialStrong((operation_id, window.k))
    arrived = _arrived_by_buffer(potential, operation_id, window.buffer_hi)
    strong.add(owner, potential, arrived)


def _read_keys(execution, operation_id, lower: int, upper: int) -> tuple:
    """The round identities from lower to upper, spilling into successors."""
    round_count = execution.rounds_by_operation[operation_id]
    last = min(upper, round_count)
    keys = []
    stop = last + 1
    for index in range(lower, stop):
        keys.append((operation_id, index))
    spill = upper - round_count
    for successor_id in execution.successors.get(operation_id, ()):
        successor_rounds = execution.rounds_by_operation[successor_id]
        overflow = min(spill, successor_rounds)
        overflow_stop = overflow + 1
        for index in range(1, overflow_stop):
            keys.append((successor_id, index))
    return tuple(keys)


def _arrived_by_buffer(potential, operation_id, buffer_hi: int) -> tuple:
    """The rounds of a hold that have arrived once the window's buffer has."""
    arrived = []
    for identity in potential:
        is_own = identity[0] == operation_id
        if is_own and identity[1] > buffer_hi:
            continue
        arrived.append(identity)
    return tuple(arrived)


class _HoldSet:
    """The holds one consumer places, the longest one, and their union."""

    def __init__(self):
        self.holds = []
        self.minimum_live_rounds = ()
        self._live_rounds = set()

    def add(self, owner, round_keys: tuple, arrived_keys: tuple) -> None:
        """Record a hold; the longest arrived set is the store's floor."""
        self.holds.append((owner, round_keys))
        self._live_rounds.update(round_keys)
        if len(arrived_keys) > len(self.minimum_live_rounds):
            self.minimum_live_rounds = arrived_keys

    def sufficient_live_rounds(self, is_open_ended: bool) -> Optional[tuple]:
        """Every round any hold names; None when a stream never ends."""
        if is_open_ended:
            return None
        ordered = sorted(
            self._live_rounds, key=identity_records.stable_identity_bytes
        )
        return tuple(ordered)


def _materialize_execution_plan(
    operations: tuple[program_records.OperationPlanningView, ...],
    resolved_operations: tuple[program_records.ResolvedOperationPlanning, ...],
    operation_window_plans: tuple[window_records.OperationWindowPlan, ...],
) -> window_records.WindowPlan:
    """Materialize exactly the typed scheme ledgers and direct DAG edges."""
    rows = list(zip(operations, resolved_operations, operation_window_plans))
    windows = {}
    plan_by_operation_id = {}
    for operation, _resolved, operation_plan in rows:
        plan_by_operation_id[operation.id] = operation_plan
        _add_operation_windows(windows, operation.id, operation_plan)
    successors = _link_boundary_windows(
        windows, operations, plan_by_operation_id
    )
    _count_window_dependencies(windows)
    tables = _operation_tables(rows)
    return window_records.WindowPlan(
        windows=windows,
        successors=successors,
        total_windows=len(windows),
        **tables,
    )


def _add_operation_windows(windows: dict, operation_id, operation_plan) -> None:
    """One Window per geometry, with the plan's internal dependencies."""
    for window_index, geometry in enumerate(operation_plan.windows):
        windows[(operation_id, window_index)] = window_records.Window(
            op_id=operation_id,
            k=window_index,
            commit_lo=geometry.commit_lo,
            commit_hi=geometry.commit_hi,
            buffer_hi=geometry.buffer_hi,
            n_rounds=geometry.round_count,
            buffer_lo=geometry.buffer_lo,
            closed_temporal_boundaries=geometry.closed_temporal_boundaries,
        )
    for source_index, destination_index in operation_plan.internal_dependencies:
        destination = windows[(operation_id, destination_index)]
        destination.deps.append((operation_id, source_index))


def _link_boundary_windows(
    windows: dict, operations, plan_by_operation_id: dict
) -> dict:
    """Every exit window of a predecessor feeds every entry window."""
    successors = {}
    for operation in operations:
        successors[operation.id] = []
    for operation in operations:
        destination_plan = plan_by_operation_id[operation.id]
        for predecessor_id in operation.decoder_boundary_predecessors:
            successors[predecessor_id].append(operation.id)
            predecessor_plan = plan_by_operation_id[predecessor_id]
            _link_exits_to_entries(
                windows,
                predecessor_id,
                predecessor_plan,
                operation.id,
                destination_plan,
            )
    return successors


def _link_exits_to_entries(
    windows, predecessor_id, predecessor_plan, operation_id, destination_plan
) -> None:
    for source_index in predecessor_plan.exit_window_indices:
        for destination_index in destination_plan.entry_window_indices:
            destination = windows[(operation_id, destination_index)]
            destination.deps.append((predecessor_id, source_index))


def _count_window_dependencies(windows: dict) -> None:
    for window_key, window in windows.items():
        window.deps_remaining = len(window.deps)
        for dependency in window.deps:
            windows[dependency].dependents.append(window_key)


def _operation_tables(rows: list) -> dict:
    """The per-operation tables of a WindowPlan, keyed by operation id."""
    tables = {
        "window_count": _table(rows, _window_count_of),
        "op_windows": _table(rows, _window_indices_of),
        "spatial_nodes": _table(rows, _spatial_nodes_of),
        "rounds_by_operation": _table(rows, _round_count_of),
        "code_names": _table(rows, _code_name_of),
        "windowed_by_operation": _table(rows, _windowed_of),
        "batch_preceding_idle_rounds_by_operation": _table(
            rows, _batch_idle_of
        ),
        "protocol_by_operation": _table(rows, _protocol_of),
    }
    return tables


def _table(rows: list, value_of: Callable) -> dict:
    table = {}
    for operation, resolved, operation_plan in rows:
        table[operation.id] = value_of(resolved, operation_plan)
    return table


def _window_count_of(resolved, operation_plan) -> int:
    del resolved
    return len(operation_plan.windows)


def _window_indices_of(resolved, operation_plan) -> list:
    del resolved
    count = len(operation_plan.windows)
    return list(range(count))


def _spatial_nodes_of(resolved, operation_plan) -> int:
    del operation_plan
    return resolved.spatial_node_count


def _round_count_of(resolved, operation_plan) -> int:
    del operation_plan
    return resolved.round_count


def _code_name_of(resolved, operation_plan) -> str:
    del operation_plan
    return resolved.code_geometry.code_name


def _windowed_of(resolved, operation_plan) -> bool:
    del resolved
    return operation_plan.windowed


def _batch_idle_of(resolved, operation_plan) -> bool:
    del resolved
    return operation_plan.batch_preceding_idle_rounds


def _protocol_of(resolved, operation_plan):
    del resolved
    return operation_plan.protocol


def _index_by_id(operations) -> dict:
    by_id = {}
    for operation in operations:
        if operation.id in by_id:
            earlier = by_id[operation.id]
            raise ValueError(
                f"duplicate operation id {operation.id}: "
                f"{earlier.name!r} and {operation.name!r}"
            )
        by_id[operation.id] = operation
    return by_id


def _check_predecessors(
    operation, predecessor_ids, by_id: dict, successors: dict
) -> None:
    seen = set()
    for predecessor_id in predecessor_ids:
        if predecessor_id in seen:
            raise ValueError(
                f"operation {operation.id} lists predecessor "
                f"{predecessor_id} more than once"
            )
        seen.add(predecessor_id)
        if predecessor_id == operation.id:
            raise ValueError(f"operation {operation.id} depends on itself")
        if predecessor_id not in by_id:
            raise ValueError(
                f"operation {operation.id} has unknown predecessor "
                f"{predecessor_id}"
            )
        successors[predecessor_id].append(operation.id)


def _check_blocker(operation, valid_blocker_ids: set) -> None:
    blocker = operation.blocked_by
    if blocker is None:
        return
    if blocker == operation.id:
        raise ValueError(f"operation {operation.id} is blocked by itself")
    if blocker not in valid_blocker_ids:
        raise ValueError(
            f"operation {operation.id} has unknown blocking operation {blocker}"
        )


def _check_acyclic(by_id: dict, successors: dict, indegree: dict) -> None:
    """Kahn's topological order; what it never reaches is a cycle."""
    ready = []
    for operation_id, degree in indegree.items():
        if degree == 0:
            ready.append(operation_id)
    visited = 0
    while ready:
        operation_id = ready.pop()
        visited += 1
        _release_successors(operation_id, successors, indegree, ready)
    if visited != len(by_id):
        cycle_ids = _stuck_ids(indegree)
        raise ValueError(
            f"operation dependency cycle involving IDs {cycle_ids}"
        )


def _stuck_ids(indegree: dict) -> list:
    """The operations a topological order never reached, in id order."""
    stuck = []
    for operation_id, degree in indegree.items():
        if degree > 0:
            stuck.append(operation_id)
    return sorted(stuck)


def _release_successors(operation_id, successors, indegree, ready) -> None:
    for successor_id in successors[operation_id]:
        indegree[successor_id] -= 1
        if indegree[successor_id] == 0:
            ready.append(successor_id)


def _note_role(role: str, members, operation_by_id: dict, roles_by_object):
    """Index one role's members; refuse an id or object two roles share."""
    seen_ids = set()
    for operation in members:
        if operation.id in seen_ids:
            raise ValueError(
                f"operation id {operation.id} appears more than once in {role}"
            )
        seen_ids.add(operation.id)
        prior = operation_by_id.get(operation.id)
        if prior is not None and prior is not operation:
            raise ValueError(
                f"operation id {operation.id} belongs to distinct objects "
                "across workload roles"
            )
        operation_by_id[operation.id] = operation
        identity = id(operation)
        roles = roles_by_object.setdefault(identity, set())
        roles.add(role)
        _check_stream_role(operation, roles)


def _check_stream_role(operation, roles: set) -> None:
    if "dynamic_streams" not in roles:
        return
    if len(roles) == 1:
        return
    other_role = "decode_ops"
    if "ops" in roles:
        other_role = "ops"
    raise ValueError(
        f"operation id {operation.id} cannot appear in both "
        f"{other_role} and dynamic_streams"
    )


def _check_static_membership(operation, static_owners: tuple) -> None:
    """An operation with detector data joins the static decode plan."""
    if not operation.emits_detector_data:
        return
    if not static_owners:
        return
    for owner in static_owners:
        if owner is operation:
            return
    raise ValueError(
        f"operation {operation.id} must share static decode membership"
    )


def _stream_owner(operation, owners: tuple):
    """The declared owner of the operation's stream, or None."""
    if not owners:
        is_own = identity_records.same_stable_identity(
            operation.stream_id, operation.id
        )
        if is_own:
            return operation
        return None
    for candidate in owners:
        if identity_records.same_stable_identity(
            candidate.id, operation.stream_id
        ):
            return candidate
    return None
