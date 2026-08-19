"""Round-count policies implementing the RoundsPolicy seam."""

from __future__ import annotations

from ..message import OpKind


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
    """Lattice-surgery round counts: op-kind aware, distance-proportional.

    The cited sections establish the unit only, one lattice-surgery step
    costing d rounds: Horsman arXiv:1111.4022v3 (Sec. 3.1, 3.2 and 6, d rounds
    of error correction per merge, per split, and per operation) and Litinski
    arXiv:1808.02892v3 ("Translation to surface codes", a two-patch or
    multi-patch measurement is one time step of d code cycles). The default
    merge_steps=2, the qubit-count convention for GENERIC, and the one-round
    MEASURE and INJECT cost are project coefficients that those sections do
    not establish.
    """

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
