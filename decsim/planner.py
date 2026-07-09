"""Rounds policies (port 5) and the compile-time execution planner (port 7).

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
WindowPlanner.plan(ops) is the single owner of windowing resolution.
"""

from __future__ import annotations

from .message import Operation, OpKind, Window, WindowPlan


def _validated(value: int, source: str) -> int:
    if value < 1:
        raise ValueError(f"{source} must give >= 1 round (got {value})")
    return int(value)


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
        base = code.rounds_per_op() if hasattr(code, "rounds_per_op") \
            else code.rounds_per_logical_cycle()
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


class WindowPlanner:
    """Build the compile-time decode window plan (the ExecutionPlanner port):
    plan(ops) is the single owner of windowing resolution for ops of known
    length. Its runtime twin is dynamic_windows.DynamicWindows, which lays
    out windows on the fly for streams whose length is not known up front."""

    def __init__(self, scheme, layout, rounds):
        self.scheme = scheme
        self.layout = layout
        if hasattr(rounds, "rounds_for"):
            self.rounds_policy = rounds
        else:
            self.rounds_policy = FixedRounds(int(rounds))

    def plan(self, ops: list[Operation]) -> WindowPlan:
        """Compute windows, dependencies, and job sizes for these operations."""
        operations = {operation.id: operation for operation in ops}
        rounds_by_operation = {
            op_id: _validated(
                self.rounds_policy.rounds_for(op, self.layout.code_for_op(op)),
                f"rounds_for({op.name})")
            for op_id, op in operations.items()}
        successors = {op_id: [] for op_id in operations}
        for op_id, operation in operations.items():
            for predecessor_id in operation.predecessors:
                successors[predecessor_id].append(op_id)
        window_specs = {
            op_id: self.scheme.plan_windows(
                op_id, rounds_by_operation[op_id],
                self.layout.code_for_op(operation))
            for op_id, operation in operations.items()}
        windows, operation_windows = self._materialize_windows(window_specs)
        window_count = {op_id: len(window_specs[op_id]) for op_id in operations}
        self._wire_dependencies(operations, windows, window_count)
        spatial_nodes = {
            op_id: self.layout.spatial_nodes_for(operation)
            for op_id, operation in operations.items()}
        summary = self._summary(operations, rounds_by_operation, window_count)
        return WindowPlan(windows=windows, window_count=window_count,
                          op_windows=operation_windows, successors=successors,
                          spatial_nodes=spatial_nodes,
                          total_windows=len(windows), summary=summary)

    @staticmethod
    def _materialize_windows(window_specs: dict) -> tuple:
        windows: dict = {}
        operation_windows: dict = {}
        for operation_id, specs in window_specs.items():
            for window_index, spec in enumerate(specs):
                if len(spec) == 3:
                    commit_lo, commit_hi, buffer_hi = spec
                    buffer_lo = commit_lo
                else:
                    buffer_lo, commit_lo, commit_hi, buffer_hi = spec
                windows[(operation_id, window_index)] = Window(
                    op_id=operation_id, k=window_index, commit_lo=commit_lo,
                    commit_hi=commit_hi, buffer_hi=buffer_hi,
                    n_rounds=buffer_hi - buffer_lo + 1, buffer_lo=buffer_lo)
                operation_windows.setdefault(operation_id, []).append(
                    window_index)
        return windows, operation_windows

    def _wire_dependencies(self, operations: dict, windows: dict,
                           window_count: dict) -> None:
        wire = getattr(self.scheme, "wire_deps", None)
        entry = getattr(self.scheme, "entry_windows", None)
        exits = getattr(self.scheme, "exit_windows", None)
        for operation_id, operation in operations.items():
            op_windows = [windows[(operation_id, k)]
                          for k in range(window_count[operation_id])]
            if wire is not None:
                wire(op_windows)
            else:
                for k in range(1, window_count[operation_id]):
                    op_windows[k].deps.append((operation_id, k - 1))
            entry_windows = entry(op_windows) if entry is not None \
                else [op_windows[0]]
            for entry_window in entry_windows:
                for predecessor_id in operation.predecessors:
                    predecessor_windows = [
                        windows[(predecessor_id, k)]
                        for k in range(window_count[predecessor_id])]
                    exit_windows = exits(predecessor_windows) \
                        if exits is not None else [predecessor_windows[-1]]
                    for exit_window in exit_windows:
                        entry_window.deps.append((predecessor_id, exit_window.k))
        for window_key, window in windows.items():
            window.deps_remaining = len(window.deps)
            for dependency in window.deps:
                windows[dependency].dependents.append(window_key)

    def _summary(self, operations: dict, rounds_by_operation: dict,
                 window_count: dict) -> dict:
        representative_code = self.layout.codes()[0]
        first_operation_id = next(iter(operations), None)
        return dict(
            distance=representative_code.distance,
            commit=representative_code.commit_rounds(),
            buffer=representative_code.buffer_rounds(),
            rounds_per_op=rounds_by_operation[first_operation_id]
            if first_operation_id is not None else 0,
            windows_per_op=window_count.get(first_operation_id, 0)
            if first_operation_id is not None else 0)
