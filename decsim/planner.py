"""Offline window planning.

The planner turns an operation DAG into a `WindowPlan` before syndrome data
arrives. Runtime code consumes that plan and pays no simulated time for planning.
See docs/PAPER_MODEL_MAP.md for the paper contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .message import Operation, Window, WindowPlan

if TYPE_CHECKING:
    from .protocols import DecodingScheme, LayoutModel


class FixedRounds:
    """Every operation runs the same number of rounds."""

    def __init__(self, round_count: int):
        self.round_count = int(round_count)

    def rounds_for(self, op, code) -> int:
        """Return the fixed round count (ignores op and code)."""
        return self.round_count


class PerOpRounds:
    """Per-operation round counts with a fallback policy."""

    def __init__(self, rounds_by_op: dict, fallback=None):
        self.rounds_by_op = dict(rounds_by_op)
        self.fallback = fallback if fallback is not None else CodeRounds()

    def rounds_for(self, op, code) -> int:
        """The mapped round count for this op, else the fallback policy's."""
        if op.id in self.rounds_by_op:
            return int(self.rounds_by_op[op.id])
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
    """Lattice-surgery-style round counts by operation size."""

    def __init__(self, merge_steps: int = 2):
        self.merge_steps = int(merge_steps)

    def rounds_for(self, op, code) -> int:
        """Multi-qubit ops take merge_steps * d rounds; single-qubit ops take d."""
        d = code.distance
        return self.merge_steps * d if len(op.qubits) >= 2 else d


class WindowPlanner:
    """Build the compile-time decode window plan."""

    def __init__(self, scheme: DecodingScheme, layout: LayoutModel, rounds):
        self.scheme = scheme
        self.layout = layout
        if hasattr(rounds, "rounds_for"):
            self.rounds_policy = rounds
        else:
            self.rounds_policy = FixedRounds(int(rounds))

    def plan(self, ops: list[Operation]) -> WindowPlan:
        """Compute windows, dependencies, and job sizes for these operations."""
        operations = {operation.id: operation for operation in ops}
        rounds_by_operation = self._rounds_by_operation(operations)
        successors = self._successors(operations)
        window_specs = self._window_specs(operations, rounds_by_operation)
        windows, operation_windows = self._materialize_windows(window_specs)
        window_count = {operation_id: len(window_specs[operation_id])
                        for operation_id in operations}
        self._wire_dependencies(operations, windows, window_count)
        spatial_nodes = {
            operation_id: self.layout.spatial_nodes_for(operation)
            for operation_id, operation in operations.items()
        }
        summary = self._summary(operations, rounds_by_operation, window_count)
        return WindowPlan(windows=windows, window_count=window_count,
                          op_windows=operation_windows,
                          successors=successors, spatial_nodes=spatial_nodes,
                          total_windows=len(windows), summary=summary)

    def _rounds_by_operation(self, operations: dict[int, Operation]) -> dict[int, int]:
        """Round count for each operation under its own code."""
        return {
            operation_id: self.rounds_policy.rounds_for(
                operation, self.layout.code_for_op(operation))
            for operation_id, operation in operations.items()
        }

    @staticmethod
    def _successors(operations: dict[int, Operation]) -> dict[int, list[int]]:
        """Forward operation edges from predecessor lists."""
        successors = {operation_id: [] for operation_id in operations}
        for operation_id, operation in operations.items():
            for predecessor_id in operation.predecessors:
                successors[predecessor_id].append(operation_id)
        return successors

    def _window_specs(self, operations: dict[int, Operation],
                      rounds_by_operation: dict[int, int]) -> dict[int, list]:
        """Ask the scheme for each operation's window layout."""
        return {
            operation_id: self.scheme.plan_windows(
                operation_id, rounds_by_operation[operation_id],
                self.layout.code_for_op(operation)
            )
            for operation_id, operation in operations.items()
        }

    @staticmethod
    def _materialize_windows(window_specs: dict[int, list]) -> tuple[dict, dict]:
        """Convert scheme tuples into Window objects."""
        windows: dict = {}
        operation_windows: dict = {}
        for operation_id, specs in window_specs.items():
            for window_index, spec in enumerate(specs):
                window = WindowPlanner._window_from_spec(
                    operation_id, window_index, spec)
                windows[(operation_id, window_index)] = window
                operation_windows.setdefault(operation_id, []).append(window_index)
        return windows, operation_windows

    @staticmethod
    def _window_from_spec(operation_id: int, window_index: int, spec: tuple) -> Window:
        """Build a Window from a three-field or four-field scheme tuple."""
        if len(spec) == 3:
            commit_lo, commit_hi, buffer_hi = spec
            buffer_lo = commit_lo
        else:
            buffer_lo, commit_lo, commit_hi, buffer_hi = spec
        window_rounds = buffer_hi - buffer_lo + 1
        return Window(
            op_id=operation_id,
            k=window_index,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=buffer_hi,
            n_rounds=window_rounds,
            buffer_lo=buffer_lo,
        )

    def _wire_dependencies(self, operations: dict[int, Operation], windows: dict,
                           window_count: dict[int, int]) -> None:
        """Wire intra-operation and cross-operation window dependencies."""
        wire = getattr(self.scheme, "wire_deps", None)
        entry = getattr(self.scheme, "entry_windows", None)
        exits = getattr(self.scheme, "exit_windows", None)
        for operation_id, operation in operations.items():
            operation_window_list = [
                windows[(operation_id, window_index)]
                for window_index in range(window_count[operation_id])
            ]
            if wire is not None:
                wire(operation_window_list)
            else:
                for window_index in range(1, window_count[operation_id]):
                    operation_window_list[window_index].deps.append(
                        (operation_id, window_index - 1))
            entry_windows = entry(operation_window_list) if entry is not None \
                else [operation_window_list[0]]
            for entry_window in entry_windows:
                for predecessor_id in operation.predecessors:
                    predecessor_windows = [
                        windows[(predecessor_id, window_index)]
                        for window_index in range(window_count[predecessor_id])
                    ]
                    exit_windows = exits(predecessor_windows) if exits is not None \
                        else [predecessor_windows[-1]]
                    for exit_window in exit_windows:
                        entry_window.deps.append((predecessor_id, exit_window.k))
        for window_key, window in windows.items():
            window.deps_remaining = len(window.deps)
            for dependency in window.deps:
                windows[dependency].dependents.append(window_key)

    def _summary(self, operations: dict[int, Operation],
                 rounds_by_operation: dict[int, int],
                 window_count: dict[int, int]) -> dict:
        """Representative values used only for logs and metrics."""
        representative_code = self.layout.codes()[0]
        first_operation_id = next(iter(operations), None)
        return dict(
            distance=representative_code.distance,
            commit=representative_code.commit_rounds(),
            buffer=representative_code.buffer_rounds(),
            rounds_per_op=rounds_by_operation[first_operation_id]
            if first_operation_id is not None else 0,
            windows_per_op=window_count.get(first_operation_id, 0)
            if first_operation_id is not None else 0,
        )
