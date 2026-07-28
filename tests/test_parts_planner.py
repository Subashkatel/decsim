"""Rounds policies: op-kinds, temporal d_m, validation, QLX pass-through."""
import pytest

from decsim.message import (
    Operation,
    OperationPlanningView,
    OperationWindowPlan,
    OpKind,
    ResolvedCodeGeometry,
    ResolvedOperationPlanning,
    WindowGeometry,
)
from decsim.planner import (
    CodeRounds,
    FixedRounds,
    GateRounds,
    PerOpRounds,
    TemporalRounds,
    _materialize_execution_plan,
)


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


def test_materializer_uses_only_ledger_and_direct_operation_edges():
    a = Operation(0, "a", (0,), has_successor=True)
    b = Operation(1, "b", (0,), predecessors=(0,))
    geometry = ResolvedCodeGeometry(
        code_name="fake",
        distance=5,
        commit_round_count=5,
        buffer_round_count=5,
        minimum_leading_buffer_round_count=5,
        minimum_trailing_buffer_round_count=5,
        one_patch_spatial_node_count=25,
        buffer_floor_override_active=False,
    )
    resolved = tuple(
        ResolvedOperationPlanning(
            operation_id=operation.id,
            code_geometry=geometry,
            round_count=10,
            round_ticks=1,
            spatial_node_count=25,
        )
        for operation in (a, b)
    )
    ledgers = tuple(
        OperationWindowPlan(
            operation_id=operation.id,
            windows=(
                WindowGeometry(1, 1, 5, 10),
                WindowGeometry(6, 6, 10, 15),
            ),
            internal_dependencies=((0, 1),),
            entry_window_indices=(0,),
            exit_window_indices=(1,),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )
        for operation in (a, b)
    )
    frozen_plan = _materialize_execution_plan(
        tuple(OperationPlanningView.from_operation(op) for op in (a, b)),
        resolved,
        ledgers,
    )
    plan = frozen_plan.materialize()
    assert frozen_plan.total_windows == 4
    assert plan.windows[(0, 1)].deps == [(0, 0)]                  # intra chain
    assert (0, 1) in plan.windows[(1, 0)].deps                    # cross-op entry<-exit
    assert plan.successors == {0: [1], 1: []}


def test_materializer_rejects_live_operations():
    operation = Operation(0, "op", (0,))

    with pytest.raises(TypeError, match="OperationPlanningView"):
        _materialize_execution_plan((operation,), (object(),), (object(),))
