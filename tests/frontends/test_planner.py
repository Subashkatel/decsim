"""The plan's cadence: where a round period comes from, and its floor.

decsim/frontends/planner.py resolves the round period once, before the
engine starts: the code card's own period if it declares one, else the
run's round_period_microseconds. A card declares a period when its
hardware fixes one (a superconducting surface-code round of about a
microsecond, Google 2207.06431 Fig. 1); a card that declares none
(SurfaceCodeModel by default) leaves the period to the run, which is
what the p-versus-d sweeps vary. The resolved period is then a whole
number of ticks, and a period under one tick is refused, because a
zero-tick cadence never advances the clock.
"""

import numpy
import pytest

import decsim.config as config
import decsim.frontends.planner as planner
import decsim.machine as machine_module
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings


def test_the_run_period_is_used_when_the_card_declares_none():
    qpu = qpu_settings.QpuSettings(round_period_microseconds=0.75)
    settings = machine_settings.MachineSettings(qpu=qpu)
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.cycle_ticks == 750_000


def test_the_card_period_wins_over_the_run_period():
    card = code_geometry.SurfaceCodeModel(round_microseconds=2.0)
    qpu = qpu_settings.QpuSettings(round_period_microseconds=1.25, code=card)
    settings = machine_settings.MachineSettings(qpu=qpu)
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.cycle_ticks == 2_000_000


def test_a_period_shorter_than_one_tick_is_refused():
    qpu = qpu_settings.QpuSettings(round_period_microseconds=0.0)
    settings = machine_settings.MachineSettings(qpu=qpu)
    with pytest.raises(
        ValueError, match="resolved round cadence must be at least one tick"
    ):
        machine_module.Machine.build(settings)


def test_a_card_period_saves_a_run_period_shorter_than_one_tick():
    card = code_geometry.SurfaceCodeModel(round_microseconds=2.0)
    qpu = qpu_settings.QpuSettings(round_period_microseconds=0.0, code=card)
    settings = machine_settings.MachineSettings(qpu=qpu)
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.cycle_ticks == 2_000_000


def test_a_cadence_that_is_not_a_finite_number_is_refused():
    card = code_geometry.SurfaceCodeModel(round_microseconds=float("inf"))
    qpu = qpu_settings.QpuSettings(code=card)
    settings = machine_settings.MachineSettings(qpu=qpu)
    with pytest.raises(
        ValueError, match="resolved round_us must be a finite real number"
    ):
        machine_module.Machine.build(settings)


def test_a_distance_that_is_not_a_whole_number_is_refused_by_name():
    """The card holds what the yaml said; the plan is where it must be a count.

    A geometry that is zero or fractional never terminates: the round
    count, the window sizes and the node counts all derive from it.
    """
    card = code_geometry.SurfaceCodeModel(distance=3.5)
    qpu = qpu_settings.QpuSettings(code=card)
    settings = machine_settings.MachineSettings(qpu=qpu)
    with pytest.raises(TypeError, match="distance must be an int >= 1"):
        machine_module.Machine.build(settings)


def test_a_distance_of_zero_is_refused_by_name():
    card = code_geometry.SurfaceCodeModel(distance=0)
    qpu = qpu_settings.QpuSettings(code=card)
    settings = machine_settings.MachineSettings(qpu=qpu)
    with pytest.raises(TypeError, match="distance must be an int >= 1"):
        machine_module.Machine.build(settings)


def test_a_commit_width_of_zero_is_refused_by_name():
    card = code_geometry.SurfaceCodeModel(commit_rounds_override=0)
    qpu = qpu_settings.QpuSettings(code=card)
    settings = machine_settings.MachineSettings(qpu=qpu)
    with pytest.raises(
        TypeError, match="commit_round_count must be an int >= 1"
    ):
        machine_module.Machine.build(settings)


# The plan's other half: the graph it validates, the windows it
# materializes, and the buffer holds it places.


def operation_of(operation_id, **fields):
    name = f"operation-{operation_id}"
    fields.setdefault("qubits", ())
    return program_records.Operation(id=operation_id, name=name, **fields)


def planning_view(operation):
    return program_records.OperationPlanningView.from_operation(operation)


def resolved_geometry(name="surface"):
    return program_records.ResolvedCodeGeometry(
        code_name=name,
        distance=3,
        commit_round_count=2,
        buffer_round_count=1,
        minimum_leading_buffer_round_count=0,
        minimum_trailing_buffer_round_count=0,
        one_patch_spatial_node_count=10,
        window_floor_justification=None,
    )


def resolved_operation(operation_id, nodes=10):
    geometry = resolved_geometry()
    return program_records.ResolvedOperationPlanning(
        operation_id=operation_id,
        code_geometry=geometry,
        round_count=4,
        round_ticks=config.TICKS_PER_MICROSECOND,
        spatial_node_count=nodes,
    )


def operation_window_plan(operation_id, geometries, dependencies=()):
    """One operation's windows, with the internal edges given."""
    destinations = set()
    sources = set()
    for source, destination in dependencies:
        sources.add(source)
        destinations.add(destination)
    entries = []
    exits = []
    for index in range(len(geometries)):
        if index not in destinations:
            entries.append(index)
        if index not in sources:
            exits.append(index)
    return window_records.OperationWindowPlan(
        operation_id=operation_id,
        windows=tuple(geometries),
        internal_dependencies=tuple(dependencies),
        entry_window_indices=tuple(entries),
        exit_window_indices=tuple(exits),
        windowed=True,
        batch_preceding_idle_rounds=False,
    )


def window_geometry(commit_lo, commit_hi):
    return window_records.WindowGeometry(
        commit_lo, commit_lo, commit_hi, commit_hi
    )


class RecordingCode:
    """A code card that records which patch counts it was sized for."""

    name = "surface"
    distance = 3
    window_floor_justification = None

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

    def spatial_nodes(self, patch_count):
        self.spatial_node_calls.append(patch_count)
        return patch_count * 10


class RecordingLayout:
    """A layout that adds one node per operation and two per patch."""

    def __init__(self, code):
        self.code = code
        self.operation_calls = []
        self.patch_calls = []

    def code_for_op(self, operation):
        del operation
        return self.code

    def code_for_patch(self, patch_identity):
        del patch_identity
        return self.code

    def spatial_nodes_for(self, operation, *, base_spatial_node_count):
        self.operation_calls.append(operation.id)
        return base_spatial_node_count + 1

    def patch_spatial_nodes_for(
        self, patch_identity, *, base_spatial_node_count
    ):
        self.patch_calls.append(patch_identity)
        return base_spatial_node_count + 2


class RecordingScheme:
    """A scheme that plans one window per operation and notes its sizes."""

    def __init__(self):
        self.validated_geometry = None
        self.plan_calls = []

    def validate_buffer(self, geometry):
        self.validated_geometry = geometry

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
        geometry = window_records.WindowGeometry(1, 1, round_count, round_count)
        return operation_window_plan(operation_id, (geometry,))


def compiled_plan(operations, planned_ids, **overrides):
    """The plan of one workload, with the collaborators given or recorded."""
    code = overrides.pop("code", None)
    if code is None:
        code = RecordingCode()
    layout = overrides.pop("layout", None)
    if layout is None:
        layout = RecordingLayout(code)
    scheme = overrides.pop("scheme", None)
    if scheme is None:
        scheme = RecordingScheme()
    rounds_policy = overrides.pop("rounds_policy", None)
    if rounds_policy is None:
        rounds_policy = round_policies.FixedRounds(4)
    cadence = overrides.pop("fallback_round_microseconds", 2.0)
    retain_strong_context = overrides.pop("retain_strong_context", False)
    absorbs_weak_windows = overrides.pop("absorbs_weak_windows", False)
    reread_regions = overrides.pop("restart_reread_buffer_regions", 0)
    open_ended = overrides.pop("open_ended", False)
    views = []
    for operation in operations:
        view = planning_view(operation)
        views.append(view)
    return planner.plan_execution(
        operations=tuple(views),
        planned_operation_ids=tuple(planned_ids),
        code=code,
        layout=layout,
        scheme=scheme,
        rounds_policy=rounds_policy,
        fallback_round_microseconds=cadence,
        retain_strong_context=retain_strong_context,
        absorbs_weak_windows=absorbs_weak_windows,
        restart_reread_buffer_regions=reread_regions,
        has_open_ended_dynamic_streams=open_ended,
    )


def test_the_plan_sizes_every_operation_and_every_patch_through_the_layout():
    """One geometry for the run, one node count per operation and patch.

    The card sizes a patch count and the layout may adjust it, so a
    two-patch operation is priced from the card's two-patch figure and
    each patch identity is resolved once, in the order it appears.
    """
    first = operation_of(10, qubits=("q0",), patches=(1, "1"))
    second = operation_of(20, qubits=("q1", "q2"))
    code = RecordingCode(cadence=None)
    layout = RecordingLayout(code)
    scheme = RecordingScheme()

    plan = compiled_plan(
        (first, second),
        (10, 20),
        code=code,
        layout=layout,
        scheme=scheme,
        fallback_round_microseconds=2.5,
    )

    operation_nodes = []
    for resolved in plan.resolved_operations:
        operation_nodes.append(resolved.spatial_node_count)
    patch_identities = []
    for resolved in plan.resolved_patches:
        patch_identities.append(resolved.patch_identity)
    assert plan.round_ticks == 2_500_000
    assert plan.code_geometry.one_patch_spatial_node_count == 10
    assert operation_nodes == [21, 21]
    assert patch_identities == [1, "1", "q1", "q2"]
    assert scheme.validated_geometry == plan.code_geometry
    assert scheme.plan_calls == [(10, 4, 2, 1), (20, 4, 2, 1)]


def test_a_cadence_that_is_a_numpy_scalar_is_taken_as_its_value():
    """A sweep hands the plan numpy floats; a tick count is still an int."""
    swept_cadences = [
        (numpy.float32(1.25), 1_250_000),
        (numpy.float64(1.5), 1_500_000),
        (numpy.int64(2), 2_000_000),
    ]
    for cadence, expected_ticks in swept_cadences:
        code = RecordingCode(cadence)
        only = operation_of(1)
        plan = compiled_plan((only,), (1,), code=code)
        assert plan.round_ticks == expected_ticks
        assert type(plan.round_ticks) is int


def test_an_operation_planned_for_no_rounds_is_refused():
    rounds_policy = round_policies.PerOperationRounds({1: 0})
    only = operation_of(1)
    with pytest.raises(ValueError, match="at least one round"):
        compiled_plan((only,), (1,), rounds_policy=rounds_policy)


def test_an_operation_nobody_plans_may_run_for_no_rounds():
    """Only the planned operations decode, so only they need a round."""
    rounds_policy = round_policies.PerOperationRounds({1: 2, 2: 0})
    first = operation_of(1)
    second = operation_of(2)
    plan = compiled_plan((first, second), (1,), rounds_policy=rounds_policy)
    round_counts = []
    for resolved in plan.resolved_operations:
        round_counts.append(resolved.round_count)
    assert round_counts == [2, 0]


@pytest.mark.parametrize(
    "graph, sentence",
    [
        (((1, ()), (1, ())), "duplicate operation id"),
        (((1, (1,)),), "depends on itself"),
        (((1, (9,)),), "unknown predecessor"),
        (((1, ()), (2, (1, 1))), "more than once"),
        (((1, (2,)), (2, (1,))), "cycle"),
    ],
)
def test_a_workload_graph_that_cannot_run_is_refused_by_name(graph, sentence):
    """The graph is checked once, at build: a run cannot repair it."""
    operations = []
    for operation_id, predecessor_ids in graph:
        operation = operation_of(operation_id, predecessors=predecessor_ids)
        operations.append(operation)
    with pytest.raises(ValueError, match=sentence):
        planner.check_operation_graph(operations)


def test_a_blocking_operation_is_checked_only_when_the_run_asks():
    """A blocker may be external, so its check is the caller's to ask for."""
    blocked = operation_of(1, blocked_by=99)
    planner.check_operation_graph([blocked])
    planner.check_operation_graph(
        [blocked], validate_blockers=True, external_blocker_ids=(99,)
    )
    with pytest.raises(ValueError, match="unknown blocking operation"):
        planner.check_operation_graph([blocked], validate_blockers=True)


def test_an_operation_blocked_by_itself_is_refused():
    blocked = operation_of(1, blocked_by=1)
    with pytest.raises(ValueError, match="blocked by itself"):
        planner.check_operation_graph([blocked], validate_blockers=True)


def test_a_stream_producer_and_its_owner_are_two_operations():
    """A dynamic stream is owned by one operation and fed by others."""
    static = operation_of(1)
    producer = operation_of(2, stream_id=3)
    owner = operation_of(3)
    static_workload = [static]
    planner.check_workload_identity(static_workload, static_workload, [])
    planner.check_workload_identity([producer], [], [owner])


def test_two_objects_with_one_id_in_one_role_are_refused_naming_it():
    """Two entries with one id would make the runtime maps disagree.

    The runtime keeps its accounts in dictionaries keyed by operation
    id (decsim/frontends/planner.py, check_workload_identity), so two
    distinct objects under one id would leave whichever the map dropped
    running with another operation's accounts. The refusal names the
    role the duplicate sits in, because that is the workload entry the
    caller has to fix.
    """
    first = operation_of(1)
    second_with_the_same_id = operation_of(1)
    workload = (first, second_with_the_same_id)
    with pytest.raises(
        ValueError, match="operation id 1 appears more than once in ops"
    ):
        planner.check_workload_identity(workload, (), ())


def test_one_operation_in_two_roles_is_refused():
    shared = operation_of(2)
    with pytest.raises(ValueError, match="dynamic_streams"):
        planner.check_workload_identity((shared,), (), (shared,))


def test_a_decode_operation_outside_the_workload_is_refused():
    workload_operation = operation_of(3)
    decode_owner = operation_of(4)
    with pytest.raises(ValueError, match="static decode membership"):
        planner.check_workload_identity(
            (workload_operation,), (decode_owner,), ()
        )


def test_a_stream_whose_owner_is_not_declared_is_refused():
    producer = operation_of(5, stream_id=7)
    another_owner = operation_of(1)
    with pytest.raises(ValueError, match="does not name"):
        planner.check_workload_identity((producer,), (), (another_owner,))


def test_every_exit_window_feeds_every_entry_window_of_a_successor():
    """A boundary edge between operations joins their window graphs.

    The successor's entry windows read the predecessor's exit boundaries,
    and nothing says which exit feeds which entry, so the plan joins
    them all: a later window waits for every boundary it could read.
    """
    first = operation_of(1)
    second = operation_of(2, decoder_boundary_predecessors=(1,))
    source = planning_view(first)
    destination = planning_view(second)
    geometries = (window_geometry(1, 1), window_geometry(2, 2))
    source_plan = operation_window_plan(1, geometries)
    destination_plan = operation_window_plan(2, geometries)
    source_resolved = resolved_operation(1, nodes=11)
    destination_resolved = resolved_operation(2, nodes=12)

    plan = planner._materialize_execution_plan(
        (source, destination),
        (source_resolved, destination_resolved),
        (source_plan, destination_plan),
    )

    assert plan.windows[(2, 0)].deps == [(1, 0), (1, 1)]
    assert plan.windows[(2, 1)].deps == [(1, 0), (1, 1)]
    assert plan.windows[(1, 0)].dependents == [(2, 0), (2, 1)]
    assert plan.windows[(2, 0)].deps_remaining == 2
    assert plan.total_windows == 4
    assert plan.spatial_nodes == {1: 11, 2: 12}


def test_a_windows_own_predecessor_inside_one_operation_is_kept():
    only = operation_of(1)
    view = planning_view(only)
    geometries = (window_geometry(1, 1), window_geometry(2, 2))
    operation_plan = operation_window_plan(1, geometries, ((0, 1),))
    resolved = resolved_operation(1)

    plan = planner._materialize_execution_plan(
        (view,), (resolved,), (operation_plan,)
    )

    assert plan.windows[(1, 1)].deps == [(1, 0)]
    assert plan.windows[(1, 0)].dependents == [(1, 1)]


def overlapping_successor_plan():
    """One window whose buffer overflows into two successor operations."""
    first = window_records.Window(1, 0, 2, 4, 6, 5, buffer_lo=1)
    second = window_records.Window(2, 0, 1, 2, 3, 3)
    third = window_records.Window(3, 0, 1, 2, 3, 3)
    return window_records.WindowPlan(
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


def test_a_windows_hold_reaches_into_the_successors_it_overflows_into():
    """A buffer past the operation's end is filled by its successors.

    The window reads through buffer_hi, and rounds past the operation's
    last round arrive on the successor's stream (Skoric 2209.08552: the
    buffer region is the next window's context), so Buffer 0 must hold
    the successors' first rounds too, once each however many successors
    share them.
    """
    execution = overlapping_successor_plan()

    buffering = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=False,
        absorbs_weak_windows=False,
        restart_reread_buffer_regions=0,
    )

    owner, held_rounds = buffering.weak_holds[0]
    assert owner == decoding_records.WindowReads((1, 0))
    assert held_rounds == (
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 1),
        (3, 1),
    )
    assert buffering.minimum_live_rounds == held_rounds
    assert buffering.potential_holds == ()


def test_an_open_ended_stream_leaves_the_stores_capacity_unbounded():
    """A stream with no end has no sufficient live-round set to size on."""
    execution = overlapping_successor_plan()

    buffering = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=False,
        absorbs_weak_windows=False,
        restart_reread_buffer_regions=0,
        has_open_ended_dynamic_streams=True,
    )

    assert buffering.sufficient_live_rounds is None


def chained_sliding_windows():
    """Four d=3 sliding windows of one operation, each bounded by the last.

    Window 0 commits 1-3 and reads to 6, window 3 commits 10-12 and
    reads to 15.
    """
    windows = {}
    for index in range(4):
        commit_lo = 3 * index + 1
        commit_hi = commit_lo + 2
        buffer_hi = commit_lo + 5
        window = window_records.Window(
            1, index, commit_lo, commit_hi, buffer_hi, 6
        )
        if index:
            window.deps = [(1, index - 1)]
        windows[(1, index)] = window
    return window_records.WindowPlan(
        windows=windows,
        window_count={1: 4},
        op_windows={1: [0, 1, 2, 3]},
        successors={1: []},
        spatial_nodes={1: 1},
        rounds_by_operation={1: 15},
        code_names={1: "surface"},
        total_windows=4,
        windowed_by_operation={1: True},
        batch_preceding_idle_rounds_by_operation={1: False},
    )


def test_the_potential_restart_hold_covers_what_the_restart_re_reads():
    """The hold is exactly the restart window's re-read range.

    Toshio 2510.25222 Sec. III C: with no re-read the restarted weak
    window begins at the round after the strong region, so the hold
    reaches no round the strong region committed; with one buffer region
    of re-read it reaches one buffer region back, which is what the plan
    then re-slices the window onto.
    """
    execution = chained_sliding_windows()

    paper = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        absorbs_weak_windows=True,
        restart_reread_buffer_regions=0,
    )
    one_region = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        absorbs_weak_windows=True,
        restart_reread_buffer_regions=1,
    )

    owner = decoding_records.PotentialRestart((1, 3))
    paper_holds = dict(paper.weak_holds)
    one_region_holds = dict(one_region.weak_holds)
    paper_rounds = tuple((1, index) for index in range(10, 16))
    one_region_rounds = tuple((1, index) for index in range(7, 16))
    assert paper_holds[owner] == paper_rounds
    assert one_region_holds[owner] == one_region_rounds


def one_window_with_a_successor():
    """One window of a seven-round operation with a four-round successor."""
    window = window_records.Window(1, 0, 3, 4, 6, 7, buffer_lo=2)
    return window_records.WindowPlan(
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


def test_a_strong_context_hold_is_one_buffer_region_on_each_side():
    """Buffer 1 keeps the rounds an escalation of this window would read.

    Toshio 2510.25222 Sec. III C: the strong decoder is given the
    committed region with a buffer region on each side, so the plan
    places that hold on Buffer 1 whether or not the window escalates.
    """
    execution = one_window_with_a_successor()

    buffering = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        absorbs_weak_windows=False,
        restart_reread_buffer_regions=0,
    )

    owner, held_rounds = buffering.potential_holds[0]
    assert owner == decoding_records.PotentialStrong((1, 0))
    assert held_rounds == tuple((1, index) for index in range(1, 7))


def test_a_forward_windows_strong_hold_reaches_into_the_successor():
    """The wider region runs past the operation's end onto its successor."""
    execution = one_window_with_a_successor()

    buffering = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        absorbs_weak_windows=True,
        restart_reread_buffer_regions=0,
    )

    held_rounds = buffering.potential_holds[0][1]
    assert held_rounds == (
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


def test_the_strong_union_counts_a_shared_round_once():
    """The store's sufficient set is the union of holds, not their sum.

    Overlapping strong contexts do not each allocate a packet, so four
    windows whose contexts total thirty three round reads keep fifteen
    rounds alive, once each.
    """
    execution = chained_sliding_windows()

    buffering = planner._plan_syndrome_buffering(
        execution,
        retain_strong_context=True,
        absorbs_weak_windows=False,
        restart_reread_buffer_regions=0,
    )

    read_count = 0
    for _owner, held_rounds in buffering.potential_holds:
        read_count += len(held_rounds)
    sufficient = buffering.sb1_sufficient_live_rounds
    every_round = set()
    for index in range(1, 16):
        every_round.add((1, index))
    assert read_count == 33
    assert len(sufficient) == 15
    assert set(sufficient) == every_round


def test_the_boundary_graph_is_checked_on_the_boundary_edges():
    """The decode graph is the boundary edges, not the workload's order.

    Two operations that name each other at a boundary would wait for
    each other forever, and one operation planned twice would decode
    twice, so the plan refuses both before a window is built.
    """
    first = operation_of(1, decoder_boundary_predecessors=(2,))
    second = operation_of(2, decoder_boundary_predecessors=(1,))
    with pytest.raises(ValueError, match="cycle"):
        compiled_plan((first, second), (1, 2))
    only = operation_of(1)
    with pytest.raises(ValueError, match="duplicate operation id"):
        compiled_plan((only,), (1, 1))
