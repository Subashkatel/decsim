"""The round policies: how many syndrome rounds an operation occupies.

Every policy fills the RoundsPolicy port (decsim/ports.py): given an
operation and its code card, return the round count, at least one. The
lattice-surgery unit of d rounds per step comes from Horsman et al.
(arXiv 1111.4022v3, Sec. 3.1, 3.2 and 6: d rounds of error correction per
merge, per split, and per operation) and Litinski (arXiv 1808.02892v3,
"Translation to surface codes": a two-patch or multi-patch measurement is
one time step of d code cycles).
"""

from typing import Optional

import decsim.ports as ports
import decsim.qpu.code_geometry as code_geometry
import decsim.records.program as program_records


class FixedRounds:
    """Every operation runs the same number of rounds."""

    def __init__(self, round_count: int):
        round_count = int(round_count)
        self.round_count = _at_least_one_round(round_count, "FixedRounds")

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: code_geometry.CodeModel,
    ) -> int:
        """The fixed count."""
        del operation, code
        return self.round_count


class PerOperationRounds:
    """A round count per operation id, with a fallback policy for the rest.

    The QLX frontend fills this from each task's duration; a zero count is
    allowed there, because a zero-duration task finalizes a stream round
    without occupying the QPU.
    """

    def __init__(
        self,
        rounds_by_operation: dict,
        fallback: Optional[ports.RoundsPolicy] = None,
    ):
        self.rounds_by_operation = {}
        rounds_by_operation = dict(rounds_by_operation)
        for operation_id, round_count in rounds_by_operation.items():
            round_count = int(round_count)
            self.rounds_by_operation[operation_id] = _at_least_zero_rounds(
                round_count, operation_id
            )
        self.fallback = fallback
        if fallback is None:
            self.fallback = CodeRounds()

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: code_geometry.CodeModel,
    ) -> int:
        """The operation's own count, or the fallback policy's."""
        if operation.id in self.rounds_by_operation:
            return self.rounds_by_operation[operation.id]
        return self.fallback.rounds_for(operation, code)


class CodeRounds:
    """Each code card's own rounds per logical cycle, optionally scaled."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: code_geometry.CodeModel,
    ) -> int:
        """The scaled logical cycle, rounded, never below one."""
        del operation
        base_rounds = code.rounds_per_logical_cycle()
        scaled_rounds = self.scale * base_rounds
        rounded_rounds = round(scaled_rounds)
        return max(1, int(rounded_rounds))


class GateRounds:
    """Lattice-surgery round counts by operation kind, proportional to d.

    A measurement or an injection costs one round; a merge costs
    merge_step_count steps of d rounds; idle and memory cost d rounds; a
    generic operation on two or more qubits counts as a merge, on one qubit
    as memory. The default merge_step_count of two, the qubit-count
    convention for GENERIC, and the one-round MEASURE and INJECT cost are
    project coefficients that the cited sections do not establish.
    """

    def __init__(self, merge_step_count: int = 2):
        merge_step_count = int(merge_step_count)
        self.merge_step_count = _at_least_one_round(
            merge_step_count, "GateRounds.merge_step_count"
        )

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: code_geometry.CodeModel,
    ) -> int:
        """The kind's cost in rounds of the code's distance."""
        distance = code.distance
        kind = operation.kind
        if (
            kind is program_records.OpKind.MEASURE
            or kind is program_records.OpKind.INJECT
        ):
            return 1
        if kind is program_records.OpKind.MERGE:
            return self.merge_step_count * distance
        if kind in (program_records.OpKind.IDLE, program_records.OpKind.MEMORY):
            return distance
        if len(operation.qubits) >= 2:
            return self.merge_step_count * distance
        return distance


class TemporalRounds:
    """A temporal distance for surgery, decoupled from the spatial distance.

    A merge, or a generic operation on two or more qubits, takes
    temporal_distance rounds; everything else takes what the base policy
    says (GateRounds by default).
    """

    def __init__(
        self,
        temporal_distance: int,
        base: Optional[ports.RoundsPolicy] = None,
    ):
        temporal_distance = int(temporal_distance)
        self.temporal_distance = _at_least_one_round(
            temporal_distance, "TemporalRounds.temporal_distance"
        )
        self.base = base
        if base is None:
            self.base = GateRounds()

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: code_geometry.CodeModel,
    ) -> int:
        """The temporal distance for surgery, else the base policy's count."""
        kind = operation.kind
        is_merge = kind is program_records.OpKind.MERGE
        is_generic = kind is program_records.OpKind.GENERIC
        is_multi_qubit = len(operation.qubits) >= 2
        is_generic_surgery = is_generic and is_multi_qubit
        if is_merge or is_generic_surgery:
            return self.temporal_distance
        return self.base.rounds_for(operation, code)


def _at_least_one_round(value: int, source: str) -> int:
    if value < 1:
        raise ValueError(f"{source} must give >= 1 round (got {value})")
    return value


def _at_least_zero_rounds(value: int, operation_id) -> int:
    if value < 0:
        raise ValueError(
            f"PerOperationRounds[{operation_id}] must give >= 0 rounds "
            f"(got {value})"
        )
    return value
