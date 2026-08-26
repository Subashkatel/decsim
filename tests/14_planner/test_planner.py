"""Behavior tests for planning round, graph, window, and retention contracts."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from decsim.config import TICKS_PER_US
from decsim.syndrome_buffer.syndrome_buffer import PotentialStrong
from decsim.message import (
    Operation,
    OperationPlanningView,
    OperationWindowPlan,
    OpKind,
    ResolvedCodeGeometry,
    ResolvedOperationPlanning,
    Window,
    WindowGeometry,
    WindowPlan,
    WindowProtocol,
)
from decsim.frontends.planner import (
    _RunPlan,
    _SyndromeBufferingPlan,
    _materialize_execution_plan,
    _plan_execution,
    _plan_syndrome_buffering,
    _validate_operation_graph,
    _validate_workload_identity,
)
from decsim.qpu.round_policies import (
    CodeRounds,
    FixedRounds,
    GateRounds,
    PerOpRounds,
    TemporalRounds,
)


def operation(
    operation_id,
    *,
    qubits=(0,),
    patches=(),
    predecessors=(),
    decoder_predecessors=(),
    stream_id=None,
    emits_detector_data=True,
    blocked_by=None,
    kind=OpKind.GENERIC,
):
    return Operation(
        id=operation_id,
        name=f"operation-{operation_id}",
        qubits=qubits,
        patches=patches,
        predecessors=predecessors,
        decoder_boundary_predecessors=decoder_predecessors,
        stream_id=stream_id,
        emits_detector_data=emits_detector_data,
        blocked_by=blocked_by,
        kind=kind,
    )


def planning_view(value):
    return OperationPlanningView.from_operation(value)


def geometry(name="surface"):
    return ResolvedCodeGeometry(
        code_name=name,
        distance=3,
        commit_round_count=2,
        buffer_round_count=1,
        minimum_leading_buffer_round_count=0,
        minimum_trailing_buffer_round_count=0,
        one_patch_spatial_node_count=10,
        window_floor_justification=None,
    )


def resolved(operation_id, *, rounds=4, nodes=10, name="surface"):
    return ResolvedOperationPlanning(
        operation_id=operation_id,
        code_geometry=geometry(name),
        round_count=rounds,
        round_ticks=TICKS_PER_US,
        spatial_node_count=nodes,
    )


def operation_plan(
    operation_id,
    windows,
    *,
    dependencies=(),
    entries=None,
    exits=None,
    windowed=True,
    batch_idle=False,
    protocol=WindowProtocol.GENERIC,
):
    if entries is None:
        incoming = {destination for _, destination in dependencies}
        entries = tuple(index for index in range(len(windows)) if index not in incoming)
    if exits is None:
        outgoing = {source for source, _ in dependencies}
        exits = tuple(index for index in range(len(windows)) if index not in outgoing)
    return OperationWindowPlan(
        operation_id=operation_id,
        windows=tuple(windows),
        internal_dependencies=tuple(dependencies),
        entry_window_indices=entries,
        exit_window_indices=exits,
        windowed=windowed,
        batch_preceding_idle_rounds=batch_idle,
        protocol=protocol,
    )


class RecordingCode:
    name = "surface"
    distance = 3

    def __init__(self, cadence=1.25):
        self.cadence = cadence
        self.spatial_node_calls = []

    def rounds_per_logical_cycle(self):
        return 5

    def round_period_us(self):
        return self.cadence

    def commit_rounds(self):
        return 2

    def buffer_rounds(self):
        return 1

    def buffering_floor(self):
        return (1, 1)

    window_floor_justification = None

    def spatial_nodes(self, patch_count):
        self.spatial_node_calls.append(patch_count)
        return patch_count * 10


class RecordingLayout:
    def __init__(self, code):
        self.code = code
        self.operation_code_override = None
        self.patch_code_override = None
        self.operation_calls = []
        self.patch_calls = []

    def code_for_op(self, value):
        return self.operation_code_override or self.code

    def code_for_patch(self, patch_identity):
        return self.patch_code_override or self.code

    def spatial_nodes_for(self, value, *, base_spatial_node_count):
        self.operation_calls.append((value.id, base_spatial_node_count))
        return base_spatial_node_count + 1

    def patch_spatial_nodes_for(self, patch_identity, *, base_spatial_node_count):
        self.patch_calls.append((patch_identity, base_spatial_node_count))
        return base_spatial_node_count + 2


class RecordingScheme:
    def __init__(self):
        self.validated_geometry = None
        self.plan_calls = []

    def validate_buffer(self, value):
        self.validated_geometry = value

    def plan_operation(
        self,
        operation_id,
        round_count,
        *,
        commit_round_count,
        buffer_round_count,
    ):
        self.plan_calls.append(
            (operation_id, round_count, commit_round_count, buffer_round_count)
        )
        return operation_plan(
            operation_id,
            (WindowGeometry(1, 1, round_count, round_count),),
            windowed=False,
        )


def compile_plan(
    operations,
    planned_ids,
    *,
    code=None,
    layout=None,
    scheme=None,
    rounds_policy=None,
    fallback_round_us=2.0,
    retain_strong_context=False,
    double_window=False,
    open_ended=False,
):
    selected_code = code or RecordingCode()
    selected_layout = layout or RecordingLayout(selected_code)
    selected_scheme = scheme or RecordingScheme()
    selected_policy = rounds_policy or FixedRounds(4)
    plan = _plan_execution(
        operations=tuple(planning_view(value) for value in operations),
        planned_operation_ids=tuple(planned_ids),
        code=selected_code,
        layout=selected_layout,
        scheme=selected_scheme,
        rounds_policy=selected_policy,
        fallback_round_us=fallback_round_us,
        retain_strong_context=retain_strong_context,
        double_window=double_window,
        has_open_ended_dynamic_streams=open_ended,
    )
    return plan, selected_code, selected_layout, selected_scheme


def test_plan_records_are_frozen_without_recursively_freezing_execution():
    """Top-level plan records are immutable while their runtime window graph stays mutable."""
    execution = WindowPlan({}, {}, {}, {}, {}, {}, {}, 0, {}, {}, {})
    buffering = _SyndromeBufferingPlan((), (), (), (), (), None)
    run_plan = _RunPlan(geometry(), (), (), 1, execution, buffering)

    with pytest.raises(FrozenInstanceError):
        run_plan.round_ticks = 2
    with pytest.raises(FrozenInstanceError):
        buffering.minimum_live_rounds = ((1, 1),)

    execution.total_windows = 3
    assert run_plan.execution.total_windows == 3
    assert _SyndromeBufferingPlan("unchecked", (), (), None, (), None).weak_holds == "unchecked"


def test_buffering_plan_accounts_for_overlap_successors_and_open_streams():
    """Retention ledgers include direct overflow once and suppress finite capacity for open streams."""
    first = Window(1, 0, 2, 4, 6, 5, buffer_lo=1)
    second = Window(2, 0, 1, 2, 3, 3)
    third = Window(3, 0, 1, 2, 3, 3)
    execution = WindowPlan(
        windows={(1, 0): first, (2, 0): second, (3, 0): third},
        window_count={1: 1, 2: 1, 3: 1},
        op_windows={1: [0]},
        successors={1: [2, 3], 2: [3], 3: []},
        spatial_nodes={1: 1, 2: 1, 3: 1},
        rounds_by_operation={1: 5, 2: 3, 3: 3},
        code_names={1: "surface", 2: "surface", 3: "surface"},
        total_windows=3,
        windowed_by_operation={1: True, 2: True, 3: True},
        batch_preceding_idle_rounds_by_operation={1: False, 2: False, 3: False},
    )

    weak_only = _plan_syndrome_buffering(
        execution,
        retain_strong_context=False,
        double_window=False,
    )
    assert weak_only.weak_holds == (
        ((1, 0), ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (3, 1))),
    )
    assert weak_only.potential_holds == ()
    assert weak_only.minimum_live_rounds == weak_only.weak_holds[0][1]
    assert set(weak_only.sufficient_live_rounds) == set(weak_only.weak_holds[0][1])
    assert (3, 2) not in weak_only.sufficient_live_rounds

    open_ended = _plan_syndrome_buffering(
        execution,
        retain_strong_context=False,
        double_window=False,
        has_open_ended_dynamic_streams=True,
    )
    assert open_ended.sufficient_live_rounds is None


def test_strong_buffering_extends_context_and_unions_shared_rounds():
    """Strong context extends around commits without double-counting shared physical rounds."""
    window = Window(1, 0, 3, 4, 6, 7, buffer_lo=2)
    execution = WindowPlan(
        windows={(1, 0): window},
        window_count={1: 1},
        op_windows={1: [0]},
        successors={1: [2], 2: []},
        spatial_nodes={1: 1, 2: 1},
        rounds_by_operation={1: 7, 2: 4},
        code_names={1: "surface", 2: "surface"},
        total_windows=1,
        windowed_by_operation={1: True},
        batch_preceding_idle_rounds_by_operation={1: False},
    )

    ordinary = _plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        double_window=False,
    )
    doubled = _plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        double_window=True,
    )

    owner, ordinary_rounds = ordinary.potential_holds[0]
    assert owner == PotentialStrong((1, 0))
    assert ordinary_rounds == tuple((1, index) for index in range(1, 7))
    assert doubled.potential_holds[0][1] == (
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 1),
        (2, 2),
    )
    assert set(doubled.sufficient_live_rounds) == set(doubled.weak_holds[0][1])
    assert set(doubled.sb1_sufficient_live_rounds) == set(
        doubled.potential_holds[0][1])
    assert doubled.minimum_live_rounds == doubled.weak_holds[0][1]
    assert doubled.sb1_minimum_live_rounds == (
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 1),
        (2, 2),
    )


@pytest.mark.parametrize(
    "bad_operations, expected_text",
    [
        ([operation(1), operation(1)], "duplicate operation id"),
        ([operation(1, predecessors=(1,))], "depends on itself"),
        ([operation(1, predecessors=(9,))], "unknown predecessor"),
        ([operation(1), operation(2, predecessors=(1, 1))], "more than once"),
        ([operation(1, predecessors=(2,)), operation(2, predecessors=(1,))], "cycle"),
    ],
)
def test_operation_graph_rejects_ambiguous_or_stranded_dependency_graphs(
    bad_operations, expected_text
):
    """Dependency validation rejects collisions, invalid edges, and cycles."""
    with pytest.raises(ValueError, match=expected_text):
        _validate_operation_graph(bad_operations)


def test_operation_graph_validates_selected_edges_and_optional_blockers():
    """Graph validation follows the selected dependency field and checks blockers only on request."""
    first = operation(1, blocked_by=2)
    second = operation(2, blocked_by=1, decoder_predecessors=(1,))
    _validate_operation_graph([first, second])
    _validate_operation_graph(
        [first, second], dependency_field="decoder_boundary_predecessors"
    )
    _validate_operation_graph(
        [operation(1, blocked_by=99)],
        validate_blockers=True,
        external_blocker_ids=(99,),
    )

    with pytest.raises(ValueError, match="unknown blocking operation"):
        _validate_operation_graph(
            [operation(1, blocked_by=99)], validate_blockers=True
        )
    with pytest.raises(ValueError, match="blocked by itself"):
        _validate_operation_graph(
            [operation(1, blocked_by=1)], validate_blockers=True
        )


def test_workload_identity_accepts_unambiguous_static_and_dynamic_owners():
    """Workload roles accept shared static objects and explicitly named dynamic stream owners."""
    static = operation(1)
    producer = operation(2, stream_id=("stream", 3))
    dynamic = operation(("stream", 3))

    _validate_workload_identity([static], [static], [])
    _validate_workload_identity([producer], [], [dynamic])
    _validate_workload_identity([operation(4, stream_id=4)], [], [])


def test_workload_identity_rejects_role_and_stream_ambiguity():
    """Workload validation rejects role collisions and missing owners."""
    cases = [
        ((operation(1), operation(1)), (), ()),
        ((operation(1),), (operation(1),), ()),
    ]
    for ops, decode_ops, dynamic_streams in cases:
        with pytest.raises((TypeError, ValueError)):
            _validate_workload_identity(ops, decode_ops, dynamic_streams)

    shared = operation(2)
    with pytest.raises(ValueError, match="dynamic_streams"):
        _validate_workload_identity((shared,), (), (shared,))
    with pytest.raises(ValueError, match="static decode membership"):
        _validate_workload_identity((operation(3),), (operation(4),), ())
    with pytest.raises(ValueError, match="does not name"):
        _validate_workload_identity(
            (operation(5, stream_id=True),), (), (operation(1),)
        )


def test_fixed_and_per_operation_policies_normalize_and_validate_counts():
    """Constant and override policies normalize constructor values and guard their count domains."""
    fixed = FixedRounds(3.9)
    assert fixed.round_count == 3
    assert fixed.rounds_for(object(), object()) == 3
    assert FixedRounds(True).round_count == 1
    with pytest.raises(ValueError):
        FixedRounds(0)

    original = {1: 0, 2: 4.8}
    fallback = SimpleNamespace(rounds_for=lambda value, code: 9)
    per_operation = PerOpRounds(original, fallback=fallback)
    original[1] = 7
    assert per_operation.rounds_for(SimpleNamespace(id=1), object()) == 0
    assert per_operation.rounds_for(SimpleNamespace(id=2), object()) == 4
    assert per_operation.rounds_for(SimpleNamespace(id=3), object()) == 9
    with pytest.raises(ValueError):
        PerOpRounds({1: -1})


def test_code_rounds_scale_with_python_rounding_and_clamp_to_one():
    """Code-derived rounds use Python rounding and never return less than one."""
    code = SimpleNamespace(rounds_per_logical_cycle=lambda: 5)
    assert CodeRounds(0.5).rounds_for(object(), code) == 2
    assert CodeRounds(-10).rounds_for(object(), code) == 1
    assert CodeRounds(True).rounds_for(object(), code) == 5


def test_gate_rounds_apply_operation_kind_and_qubit_arity_costs():
    """Gate rounds distinguish constant, distance, merge, and multi-qubit costs."""
    code = SimpleNamespace(distance=5)
    policy = GateRounds(merge_steps=3)
    expected = {
        OpKind.MEASURE: 1,
        OpKind.INJECT: 1,
        OpKind.MERGE: 15,
        OpKind.IDLE: 5,
        OpKind.MEMORY: 5,
    }
    for kind, round_count in expected.items():
        assert policy.rounds_for(operation(1, kind=kind), code) == round_count
    assert policy.rounds_for(operation(2, qubits=(0, 1)), code) == 15
    assert policy.rounds_for(operation(3, qubits=(0,)), code) == 5
    assert GateRounds(True).merge_steps == 1
    with pytest.raises(ValueError):
        GateRounds(0)


def test_temporal_rounds_override_merges_and_delegate_other_operations():
    """Temporal rounds replace merge distance while delegating all other operation kinds."""
    fallback = SimpleNamespace(rounds_for=lambda value, code: 13)
    policy = TemporalRounds(7, base=fallback)
    assert policy.rounds_for(operation(1, kind=OpKind.MERGE), object()) == 7
    assert policy.rounds_for(operation(2, qubits=(0, 1)), object()) == 7
    assert policy.rounds_for(operation(3, kind=OpKind.MEASURE), object()) == 13
    assert TemporalRounds(True).d_m == 1
    with pytest.raises(ValueError):
        TemporalRounds(0)


def test_execution_planning_resolves_geometry_patches_seams_and_graph():
    """Execution planning compiles custom code, layout, policy, and scheme seams into one canonical plan."""
    code = RecordingCode(cadence=None)
    layout = RecordingLayout(code)
    scheme = RecordingScheme()
    first = operation(10, qubits=("q0",), patches=(1, "1", 1))
    second = operation(20, qubits=("q1", "q2"), decoder_predecessors=(10,))

    plan, _, _, _ = compile_plan(
        (first, second),
        (10, 20),
        code=code,
        layout=layout,
        scheme=scheme,
        rounds_policy=FixedRounds(4),
        fallback_round_us=2.5,
    )

    assert plan.round_ticks == 2_500_000
    assert plan.code_geometry == ResolvedCodeGeometry(
        code_name="surface",
        distance=3,
        commit_round_count=2,
        buffer_round_count=1,
        minimum_leading_buffer_round_count=1,
        minimum_trailing_buffer_round_count=1,
        one_patch_spatial_node_count=10,
        window_floor_justification=None,
    )
    assert [value.operation_id for value in plan.resolved_operations] == [10, 20]
    assert [value.spatial_node_count for value in plan.resolved_operations] == [31, 21]
    assert [value.patch_identity for value in plan.resolved_patches] == [1, "1", "q1", "q2"]
    assert [value.spatial_node_count for value in plan.resolved_patches] == [12] * 4
    assert set(code.spatial_node_calls) == {1, 2, 3}
    assert scheme.validated_geometry == plan.code_geometry
    assert scheme.plan_calls == [(10, 4, 2, 1), (20, 4, 2, 1)]
    assert plan.execution.successors == {10: [20], 20: []}
    assert plan.execution.windows[(20, 0)].deps == [(10, 0)]
    assert plan.execution.windows[(10, 0)].dependents == [(20, 0)]


def test_execution_planning_accepts_real_numpy_scalars():
    """Cadence normalization accepts NumPy real scalars."""
    witnesses = [
        (np.float32(1.25), 1_250_000),
        (np.float64(1.5), 1_500_000),
        (np.int64(2), 2_000_000),
    ]
    for cadence, expected_ticks in witnesses:
        plan, _, _, _ = compile_plan((operation(1),), (1,), code=RecordingCode(cadence))
        assert plan.round_ticks == expected_ticks
        assert type(plan.round_ticks) is int


def test_execution_planning_rejects_invalid_cadence_and_code_selection():
    """Execution planning rejects nonfinite cadence, sub-tick cadence, and inconsistent selected code objects."""
    for cadence in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite real"):
            compile_plan((operation(1),), (1,), code=RecordingCode(cadence))
    with pytest.raises(ValueError, match="at least one tick"):
        compile_plan((operation(1),), (1,), code=RecordingCode(0.4e-6))

    code = RecordingCode()
    operation_layout = RecordingLayout(code)
    operation_layout.operation_code_override = RecordingCode()
    with pytest.raises(ValueError, match="operation 1"):
        compile_plan((operation(1),), (1,), code=code, layout=operation_layout)

    patch_layout = RecordingLayout(code)
    patch_layout.patch_code_override = RecordingCode()
    with pytest.raises(ValueError, match="patch"):
        compile_plan((operation(1, patches=("patch",)),), (1,), code=code, layout=patch_layout)


def test_execution_planning_rejects_unknown_zero_round_and_invalid_boundary_owners():
    """Execution planning rejects unknown owners, zero owner rounds, and invalid boundary graphs."""
    with pytest.raises(ValueError, match="unknown planned operation id"):
        compile_plan((operation(1),), (9,))
    with pytest.raises(ValueError, match="at least one round"):
        compile_plan(
            (operation(1),),
            (1,),
            rounds_policy=PerOpRounds({1: 0}),
        )
    with pytest.raises(ValueError, match="unknown predecessor"):
        compile_plan(
            (operation(1, decoder_predecessors=(9,)),),
            (1,),
        )

    plan, _, _, _ = compile_plan(
        (operation(1), operation(2)),
        (1,),
        rounds_policy=PerOpRounds({1: 2, 2: 0}),
    )
    assert [value.round_count for value in plan.resolved_operations] == [2, 0]


def test_execution_planning_rejects_duplicate_planned_ids_through_graph_validation():
    """Selected owner duplication is rejected when the boundary graph is validated."""
    with pytest.raises(ValueError, match="duplicate operation id"):
        compile_plan((operation(1),), (1, 1))


def test_materialization_copies_ledgers_and_builds_cartesian_boundary_edges():
    """Materialization copies scheme ledgers and links every exit window to every boundary entry."""
    source = planning_view(operation(1))
    destination = planning_view(operation(2, decoder_predecessors=(1,)))
    source_plan = operation_plan(
        1,
        (WindowGeometry(1, 1, 1, 1), WindowGeometry(2, 2, 2, 2)),
        protocol=WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
    )
    destination_plan = operation_plan(
        2,
        (WindowGeometry(1, 1, 1, 1), WindowGeometry(2, 2, 2, 2)),
        batch_idle=True,
    )

    result = _materialize_execution_plan(
        (source, destination),
        (resolved(1, nodes=11), resolved(2, nodes=12)),
        (source_plan, destination_plan),
    )

    expected_dependencies = [(1, 0), (1, 1)]
    assert result.windows[(2, 0)].deps == expected_dependencies
    assert result.windows[(2, 1)].deps == expected_dependencies
    assert result.windows[(1, 0)].dependents == [(2, 0), (2, 1)]
    assert result.windows[(1, 1)].dependents == [(2, 0), (2, 1)]
    assert result.windows[(2, 0)].deps_remaining == 2
    assert result.window_count == {1: 2, 2: 2}
    assert result.op_windows == {1: [0, 1], 2: [0, 1]}
    assert result.spatial_nodes == {1: 11, 2: 12}
    assert result.rounds_by_operation == {1: 4, 2: 4}
    assert result.windowed_by_operation == {1: True, 2: True}
    assert result.batch_preceding_idle_rounds_by_operation == {1: False, 2: True}
    assert result.protocol_by_operation[1] is WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE
    assert result.total_windows == 4


def test_materialization_preserves_internal_edges():
    """Materialization preserves internal dependencies."""
    view = planning_view(operation(1))
    plan = operation_plan(
        1,
        (WindowGeometry(1, 1, 1, 1), WindowGeometry(2, 2, 2, 2)),
        dependencies=((0, 1),),
    )
    result = _materialize_execution_plan((view,), (resolved(1),), (plan,))
    assert result.windows[(1, 1)].deps == [(1, 0)]
    assert result.windows[(1, 0)].dependents == [(1, 1)]


def test_materialization_truncates_excess_positional_inputs():
    """Materialization deliberately relies on zip and ignores excess cards and ledgers."""
    first = planning_view(operation(1))
    first_plan = operation_plan(1, (WindowGeometry(1, 1, 1, 1),))
    excess_plan = operation_plan(2, (WindowGeometry(1, 1, 1, 1),))
    result = _materialize_execution_plan(
        (first,),
        (resolved(1), resolved(2)),
        (first_plan, excess_plan),
    )
    assert result.total_windows == 1
    assert result.op_windows == {1: [0]}
