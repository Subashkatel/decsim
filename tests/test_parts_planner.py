"""Rounds policies: op-kinds, temporal d_m, validation, QLX pass-through."""
import pytest

from decsim.message import Operation, OpKind
from decsim.planner import (CodeRounds, WindowPlanner, FixedRounds,
                           GateRounds, PerOpRounds, TemporalRounds)


class _Code:
    name = "fake"
    distance = 5
    def rounds_per_op(self): return 5
    def commit_rounds(self): return 5
    def buffer_rounds(self): return 5


def _op(kind=OpKind.GENERIC, qubits=(0,)):
    return Operation(0, "op", qubits, kind=kind)


def test_gate_rounds_kind_table():
    g, c = GateRounds(), _Code()
    assert g.rounds_for(_op(OpKind.MEASURE), c) == 1
    assert g.rounds_for(_op(OpKind.INJECT), c) == 1
    assert g.rounds_for(_op(OpKind.MERGE, (0, 1)), c) == 10       # merge_steps*d
    assert g.rounds_for(_op(OpKind.MEMORY), c) == 5
    # GENERIC falls back to today's qubit-count rule (parity)
    assert g.rounds_for(_op(OpKind.GENERIC, (0,)), c) == 5
    assert g.rounds_for(_op(OpKind.GENERIC, (0, 1)), c) == 10


def test_temporal_rounds_decouples_dm():
    t, c = TemporalRounds(d_m=3), _Code()
    assert t.rounds_for(_op(OpKind.MERGE, (0, 1)), c) == 3        # d_m, not d
    assert t.rounds_for(_op(OpKind.GENERIC, (0, 1)), c) == 3
    assert t.rounds_for(_op(OpKind.MEMORY), c) == 5               # base policy


def test_validation_rejects_below_one():
    with pytest.raises(ValueError):
        FixedRounds(0)
    with pytest.raises(ValueError):
        PerOpRounds({3: 0})
    with pytest.raises(ValueError):
        TemporalRounds(0)


def test_per_op_passthrough_and_fallback():
    p = PerOpRounds({7: 42}, fallback=FixedRounds(11))
    assert p.rounds_for(Operation(7, "x", (0,)), _Code()) == 42
    assert p.rounds_for(Operation(8, "y", (0,)), _Code()) == 11


def test_planner_plans_windows_and_deps():
    class _Scheme:
        def plan_windows(self, op_id, rounds, code):
            return [(1, 5, 10), (6, 10, 15)]
    class _Layout:
        def code_for_op(self, op): return _Code()
        def spatial_nodes_for(self, op): return 25
        def codes(self): return [_Code()]
    a = Operation(0, "a", (0,), has_successor=True)
    b = Operation(1, "b", (0,), predecessors=(0,))
    plan = WindowPlanner(_Scheme(), _Layout(), FixedRounds(10)).plan([a, b])
    assert plan.total_windows == 4
    assert plan.windows[(0, 1)].deps == [(0, 0)]                  # intra chain
    assert (0, 1) in plan.windows[(1, 0)].deps                    # cross-op entry<-exit
    assert plan.successors == {0: [1], 1: []}
