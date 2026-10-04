"""The round policies: how many syndrome rounds an operation occupies.

The lattice-surgery unit of d rounds per step is Horsman et al.'s
(1111.4022v3, Secs. 3.1, 3.2 and 6: d rounds per merge, per split, per
operation) and Litinski's (1808.02892v3: a multi-patch measurement is
one time step of d code cycles). A policy is a frozen record, so two
tasks whose policies differ in an argument have two strong ids.
"""

import dataclasses
import numbers

import decsim.ports as ports
import decsim.records.program as program_records


@dataclasses.dataclass(frozen=True)
class FixedRounds:
    """Every operation runs the same number of rounds."""

    round_count: int

    def __post_init__(self) -> None:
        _check_round_count("FixedRounds", self.round_count, minimum=1)

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: ports.CodeModel,
    ) -> int:
        """The fixed count."""
        del operation, code
        return self.round_count


@dataclasses.dataclass(frozen=True)
class CodeRounds:
    """Each code card's own rounds per logical cycle, optionally scaled."""

    scale: float = 1.0

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: ports.CodeModel,
    ) -> int:
        """The scaled logical cycle, rounded, never below one."""
        del operation
        base_rounds = code.rounds_per_logical_cycle()
        scaled_rounds = self.scale * base_rounds
        rounded_rounds = round(scaled_rounds)
        return max(1, int(rounded_rounds))


@dataclasses.dataclass(frozen=True)
class PerOperationRounds:
    """A round count per operation id, with a fallback policy for the rest.

    rounds_by_operation is a tuple of (id, count) pairs so the record is
    plain data. A zero count is allowed: an operation may finalize a stream
    round without occupying the QPU.
    """

    rounds_by_operation: tuple
    fallback: ports.RoundsPolicy = CodeRounds()

    def __post_init__(self) -> None:
        _check_pairs(self.rounds_by_operation)
        for operation_id, round_count in self.rounds_by_operation:
            name = f"PerOperationRounds[{operation_id}]"
            _check_round_count(name, round_count, minimum=0)

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: ports.CodeModel,
    ) -> int:
        """The operation's own count, or the fallback policy's."""
        for operation_id, round_count in self.rounds_by_operation:
            if operation_id == operation.id:
                return round_count
        return self.fallback.rounds_for(operation, code)


@dataclasses.dataclass(frozen=True)
class GateRounds:
    """Lattice-surgery round counts by operation kind, proportional to d.

    A merge costs merge_step_count steps of d rounds, two by default: a
    merge, "d rounds of error correction, treating the entire system as a
    single data surface" (Horsman et al. 1111.4022 Sec. 3.1), and the split,
    "a total of d rounds of error correction" (Sec. 3.2). Idle and memory
    cost d rounds; a generic operation on two or more qubits counts as a
    merge, on one as memory; a measurement or injection costs one round. The
    GENERIC convention and the one-round MEASURE and INJECT are project
    coefficients the cited sections do not establish.
    """

    merge_step_count: int = 2

    def __post_init__(self) -> None:
        _check_round_count(
            "GateRounds.merge_step_count", self.merge_step_count, minimum=1
        )

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: ports.CodeModel,
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


def _check_round_count(name: str, value, minimum: int) -> None:
    """A round count is a whole number, at least minimum; not a bool."""
    is_whole = isinstance(value, numbers.Integral)
    if not is_whole or isinstance(value, bool):
        raise ValueError(
            f"{name} must give a whole number of rounds (got {value!r})"
        )
    if value >= minimum:
        return
    unit = "rounds"
    if minimum == 1:
        unit = "round"
    raise ValueError(f"{name} must give >= {minimum} {unit} (got {value})")


def _check_pairs(rounds_by_operation) -> None:
    """The counts are a tuple of (operation id, round count) pairs."""
    refusal = ValueError(
        "PerOperationRounds.rounds_by_operation must be a tuple of "
        f"(operation id, round count) pairs, got {rounds_by_operation!r}; "
        "tuple(counts.items()) turns a dict into one"
    )
    if not isinstance(rounds_by_operation, tuple):
        raise refusal
    for pair in rounds_by_operation:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise refusal
