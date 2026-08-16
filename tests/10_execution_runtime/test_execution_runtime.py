"""Behavior contracts for the timing-only execution runtime."""

from types import SimpleNamespace

import pytest

import decsim.execution_runtime as execution_runtime
from decsim.execution_runtime import ExecutionRuntime
from decsim.message import Decision, ExecutionProgram, Operation, ResourceClaim


class RecordingEngine:
    def __init__(self):
        self.now = 0
        self.events = []
        self.scheduled = []

    def schedule(self, tick, callback, *, label):
        self.scheduled.append((tick, callback, label))

    def log(self, component, message):
        self.events.append(("log", component, message))

    def release_next(self):
        tick, callback, label = self.scheduled.pop(0)
        self.now = tick
        callback()
        return label


class RecordingController:
    def __init__(self, engine, *, round_ticks=1, gates=False):
        self.engine = engine
        self.round_ticks = round_ticks
        self.gates_start_on_round_boundaries = gates
        self.allowed = {}
        self.issued = []
        self.boundaries = []
        self.runtime = None
        self.observed_at_issue = []

    def round_ticks_for(self, operation):
        return self.round_ticks

    def can_start(self, operation):
        self.engine.events.append(("can_start", operation.id))
        return self.allowed.get(operation.id, True)

    def issue_operation(self, operation, idle_rounds):
        self.issued.append((operation, idle_rounds))
        if self.runtime is not None:
            self.observed_at_issue.append((
                operation.id,
                operation.id in self.runtime.started,
                self.runtime.op_start_time.get(operation.id),
                dict(self.runtime.idle_rounds_by_patch),
            ))
        self.engine.events.append(("issue", operation.id, idle_rounds))

    def before_successor_release(self, operation):
        self.engine.events.append(("before", operation.id))

    def after_successor_release(self, operation):
        self.engine.events.append(("after", operation.id))

    def note_round_boundary(self, patch):
        self.boundaries.append(patch)
        self.engine.events.append(("boundary", patch))


class RecordingFactory:
    def __init__(self, engine):
        self.engine = engine
        self.requests = []

    def request(self, operation_id, callback):
        self.engine.events.append(("factory_request", operation_id))
        self.requests.append((operation_id, callback))

    def release(self, index=0):
        self.requests[index][1]()


def make_operation(operation_id, *, name=None, qubits=(), **changes):
    return Operation(operation_id, name or f"operation-{operation_id}", qubits, **changes)


def claims_for(*operations):
    return {operation.id: [] for operation in operations}


def make_runtime(*operations, round_ticks=1, gates=False, claims=None):
    engine = RecordingEngine()
    controller = RecordingController(engine, round_ticks=round_ticks, gates=gates)
    factory = RecordingFactory(engine)
    runtime = ExecutionRuntime(
        engine,
        controller=controller,
        factory=factory,
        resource_claims_by_operation_id=claims if claims is not None else claims_for(*operations),
    )
    controller.runtime = runtime
    return runtime, engine, controller, factory


def mutable_runtime_state(runtime, engine, controller, factory):
    names = (
        "operations", "dependencies_remaining", "successors", "schedule_released",
        "busy_claims", "requested", "state_ready", "started", "done_bodies",
        "decode_released", "op_start_time", "body_done_time", "decode_release_time",
        "result_return_time_by_operation", "idle_rounds_by_patch",
    )
    copied = {}
    for name in names:
        value = getattr(runtime, name)
        copied[name] = value.copy()
    copied["program"] = runtime.program
    copied["last_finish_time"] = runtime.last_finish_time
    copied["engine_events"] = list(engine.events)
    copied["scheduled"] = list(engine.scheduled)
    copied["issued"] = list(controller.issued)
    copied["factory_requests"] = list(factory.requests)
    return copied


def test_construction_preserves_collaborators_and_shallow_copies_claim_mapping():
    """Construction preserves collaborators and shallow-copies only the claim mapping."""
    operation = make_operation(1)
    claim_list = []
    supplied_claims = {operation.id: claim_list}
    runtime, engine, controller, factory = make_runtime(operation, claims=supplied_claims)

    assert runtime.engine is engine
    assert runtime.controller is controller
    assert runtime.factory is factory
    assert runtime._claims[operation.id] is claim_list
    supplied_claims[2] = []
    assert 2 not in runtime._claims
    claim_list.append(ResourceClaim("qubit", frozenset({"q"})))
    assert runtime._claims[operation.id] == claim_list
    with pytest.raises(TypeError):
        runtime._claims[3] = []

    bare_engine = object()
    bare_controller = object()
    bare_factory = object()
    bare_runtime = ExecutionRuntime(
        bare_engine,
        controller=bare_controller,
        factory=bare_factory,
        resource_claims_by_operation_id={},
    )
    assert (bare_runtime.engine, bare_runtime.controller, bare_runtime.factory) == (
        bare_engine, bare_controller, bare_factory
    )


def test_construction_starts_with_the_documented_empty_lifecycle():
    """A new runtime starts unloaded with empty lifecycle state and a zero finish time."""
    runtime, _, _, _ = make_runtime()

    assert runtime.program is None
    assert runtime.workload_complete is False
    for name in (
        "operations", "dependencies_remaining", "successors", "busy_claims",
        "op_start_time", "body_done_time", "decode_release_time",
        "result_return_time_by_operation", "idle_rounds_by_patch",
    ):
        assert getattr(runtime, name) == {}
    for name in (
        "schedule_released", "requested", "state_ready", "started",
        "done_bodies", "decode_released",
    ):
        assert getattr(runtime, name) == set()
    assert runtime.last_finish_time == 0


def test_deleted_snapshot_projection_is_not_part_of_the_runtime_surface():
    """The deleted snapshot projection and record remain absent without compatibility shims."""
    runtime, _, _, _ = make_runtime()

    assert not hasattr(execution_runtime, "ExecutionSnapshot")
    assert not hasattr(runtime, "snapshot")


def test_empty_program_completes_physically_and_second_load_is_atomic():
    """An empty loaded program is complete, and a second load changes no lifecycle state."""
    runtime, engine, controller, factory = make_runtime()
    program = ExecutionProgram(
        (),
        decode_operations=(object(),),
        dynamic_streams=(object(),),
        protected_regions=(object(),),
    )
    runtime.load_program(program)

    assert runtime.program is program
    assert runtime.workload_complete is True
    before = mutable_runtime_state(runtime, engine, controller, factory)
    with pytest.raises(RuntimeError, match="already loaded"):
        runtime.load_program(ExecutionProgram((make_operation(9),)))
    assert mutable_runtime_state(runtime, engine, controller, factory) == before


def test_load_builds_ordered_dependency_edges_and_releases_successor_after_body():
    """Body completion consumes every ordered dependency edge before admitting a successor."""
    root = make_operation(1)
    successor = make_operation(2, predecessors=(1, 1))
    runtime, engine, controller, _ = make_runtime(root, successor)
    runtime.load_program(ExecutionProgram((root, successor)))

    assert runtime.operations == {1: root, 2: successor}
    assert runtime.dependencies_remaining == {1: 0, 2: 2}
    assert runtime.successors == {1: [2, 2], 2: []}
    assert [operation.id for operation, _ in controller.issued] == [1]

    engine.events.clear()
    engine.now = 4
    runtime.body_done(root)

    assert runtime.dependencies_remaining[2] == 0
    assert [operation.id for operation, _ in controller.issued] == [1, 2]
    assert runtime.body_done_time[1] == 4
    assert runtime.last_finish_time == 4


def test_load_deliberately_leaves_graph_validation_to_the_trusted_boundary():
    """Loading deliberately accepts duplicate identities and cycles while unknown predecessors fail naturally."""
    first = make_operation(1, name="first")
    replacement = make_operation(1, name="replacement")
    duplicate_runtime, _, _, _ = make_runtime(first, replacement)
    duplicate_runtime.load_program(ExecutionProgram((first, replacement)))
    assert duplicate_runtime.operations[1] is replacement

    left = make_operation(2, predecessors=(3,))
    right = make_operation(3, predecessors=(2,))
    cyclic_runtime, _, cyclic_controller, _ = make_runtime(left, right)
    cyclic_runtime.load_program(ExecutionProgram((left, right)))
    assert cyclic_runtime.requested == set()
    assert cyclic_controller.issued == []

    orphan = make_operation(4, predecessors=(99,))
    orphan_runtime, _, _, _ = make_runtime(orphan)
    with pytest.raises(KeyError):
        orphan_runtime.load_program(ExecutionProgram((orphan,)))


def test_schedule_release_uses_raw_cadence_product_without_local_validation():
    """Timing-only scheduling passes every nonzero raw cadence product to the engine unchanged."""
    immediate = make_operation(1, scheduled_start_round=0)
    negative = make_operation(2, scheduled_start_round=1)
    fractional = make_operation(3, scheduled_start_round=1.5)
    runtime, _, controller, _ = make_runtime(immediate, negative, fractional, round_ticks=-4)
    runtime.load_program(ExecutionProgram((immediate, negative, fractional)))

    assert runtime.schedule_released == {1}
    assert [(tick, label) for tick, _, label in runtime.engine.scheduled] == [
        (-4, "scheduled-start(operation-2)"),
        (-6.0, "scheduled-start(operation-3)"),
    ]
    assert [operation.id for operation, _ in controller.issued] == [1]


def test_scheduled_callback_is_locally_idempotent_after_admission():
    """A repeated scheduled-release callback does not request or issue an operation twice."""
    operation = make_operation(1, scheduled_start_round=2, clifford=False)
    runtime, _, controller, factory = make_runtime(operation, round_ticks=3)
    runtime.load_program(ExecutionProgram((operation,)))
    callback = runtime.engine.scheduled[0][1]

    callback()
    callback()

    assert runtime.schedule_released == {1}
    assert runtime.requested == {1}
    assert len(factory.requests) == 1
    factory.release()
    factory.release()
    assert [issued.id for issued, _ in controller.issued] == [1]


def test_magic_state_request_holds_resources_until_readiness_and_issues_once():
    """Magic-state admission claims first, waits for readiness, and issues at most once."""
    operation = make_operation(1, qubits=("data",), clifford=False)
    claims = {1: [ResourceClaim("qubit", frozenset({"data"}))]}
    runtime, engine, controller, factory = make_runtime(operation, claims=claims)
    runtime.load_program(ExecutionProgram((operation,)))

    assert runtime.busy_claims == {("qubit", "data"): 1}
    assert runtime.requested == {1}
    assert runtime.state_ready == set()
    assert controller.issued == []
    assert engine.events[-1] == ("factory_request", 1)

    runtime.idle_rounds_by_patch["data"] = 3
    engine.now = 7
    factory.release()
    factory.release()
    assert runtime.op_start_time == {1: 7}
    assert controller.issued == [(operation, 3)]
    assert controller.observed_at_issue == [(1, True, 7, {})]


def test_feedback_and_controller_cadence_are_independent_start_gates():
    """A blocked operation starts only after feedback readiness and controller cadence both allow it."""
    operation = make_operation(1, blocked_by=0)
    runtime, engine, controller, _ = make_runtime(operation)
    controller.allowed[1] = False
    runtime.load_program(ExecutionProgram((operation,)))

    assert runtime.state_ready == {1}
    assert runtime.started == set()
    engine.now = 3
    runtime.on_decision(Decision(1, releases_operation=True))
    assert runtime.decode_released == {1}
    assert runtime.decode_release_time == {1: 3}
    assert runtime.started == set()

    controller.allowed[1] = True
    engine.now = 5
    runtime._maybe_begin(operation)
    assert runtime.started == {1}
    assert runtime.op_start_time == {1: 5}


def test_readiness_callback_deliberately_does_not_recheck_schedule_or_claim_admission():
    """A direct readiness callback deliberately trusts its admission path instead of rechecking it."""
    operation = make_operation(1, scheduled_start_round=9)
    runtime, engine, controller, _ = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))
    assert runtime.requested == set()

    engine.now = 2
    runtime._state_became_ready(operation)

    assert runtime.started == {1}
    assert runtime.op_start_time == {1: 2}
    assert [issued.id for issued, _ in controller.issued] == [1]


def test_claim_publication_is_ordered_and_all_or_nothing():
    """Typed resource claims publish in claim and repr order only after the entire set passes."""
    operation = make_operation(1)
    claims = {
        1: [
            ResourceClaim("qubit", frozenset({"b", "a"})),
            ResourceClaim("ancilla", frozenset({2})),
        ]
    }
    runtime, _, _, _ = make_runtime(operation, claims=claims)
    runtime.operations[1] = operation
    runtime._claim_resources(operation)
    assert list(runtime.busy_claims) == [
        ("qubit", "a"), ("qubit", "b"), ("ancilla", 2)
    ]

    contender = make_operation(2)
    holder = make_operation(3)
    conflict_claims = {
        2: [
            ResourceClaim("qubit", frozenset({"new"})),
            ResourceClaim("qubit", frozenset({"busy"})),
        ]
    }
    conflict_runtime, _, _, _ = make_runtime(contender, claims=conflict_claims)
    conflict_runtime.operations.update({2: contender, 3: holder})
    conflict_runtime.busy_claims[("qubit", "busy")] = 3
    before = dict(conflict_runtime.busy_claims)
    with pytest.raises(RuntimeError, match="share qubit resource"):
        conflict_runtime._claim_resources(contender)
    assert conflict_runtime.busy_claims == before


def test_claim_rejects_duplicate_operands_and_duplicate_typed_keys_before_publication():
    """Duplicate operands or typed keys fail before any resource ownership is published."""
    duplicate_qubits = make_operation(1, qubits=("q", "q"))
    runtime, _, _, _ = make_runtime(duplicate_qubits)
    runtime.operations[1] = duplicate_qubits
    with pytest.raises(RuntimeError, match="more than once"):
        runtime._claim_resources(duplicate_qubits)
    assert runtime.busy_claims == {}

    duplicate_key = make_operation(2)
    duplicate_claims = {
        2: [
            ResourceClaim("qubit", frozenset({"q"})),
            ResourceClaim("qubit", frozenset({"q"})),
        ]
    }
    duplicate_runtime, _, _, _ = make_runtime(duplicate_key, claims=duplicate_claims)
    duplicate_runtime.operations[2] = duplicate_key
    with pytest.raises(RuntimeError, match="share qubit resource"):
        duplicate_runtime._claim_resources(duplicate_key)
    assert duplicate_runtime.busy_claims == {}


def test_claim_shape_and_mapping_completeness_use_natural_failures():
    """Missing claims and unhashable operands fail naturally while extra claim entries stay unused."""
    missing = make_operation(1)
    runtime, _, _, _ = make_runtime(missing, claims={99: []})
    runtime.operations[1] = missing
    with pytest.raises(KeyError):
        runtime._claim_resources(missing)
    with pytest.raises(KeyError):
        runtime._free_resources(missing)
    assert runtime.busy_claims == {}

    valid = make_operation(3)
    extra_runtime, _, _, _ = make_runtime(valid, claims={3: [], 99: object()})
    extra_runtime.operations[3] = valid
    extra_runtime._claim_resources(valid)
    assert extra_runtime.busy_claims == {}

    unhashable = make_operation(2, qubits=([],))
    unhashable_runtime, _, _, _ = make_runtime(unhashable)
    unhashable_runtime.operations[2] = unhashable
    with pytest.raises(TypeError):
        unhashable_runtime._claim_resources(unhashable)
    assert unhashable_runtime.busy_claims == {}


def test_free_requires_exact_holders_and_is_all_or_nothing():
    """Resource release verifies every exact holder before deleting any typed key."""
    operation = make_operation(1)
    claims = {
        1: [
            ResourceClaim("qubit", frozenset({"a", "b"})),
            ResourceClaim("qubit", frozenset({"a"})),
        ]
    }
    runtime, _, _, _ = make_runtime(operation, claims=claims)

    runtime.busy_claims = {("qubit", "a"): 1}
    before = dict(runtime.busy_claims)
    with pytest.raises(RuntimeError, match="unclaimed"):
        runtime._free_resources(operation)
    assert runtime.busy_claims == before

    runtime.busy_claims = {("qubit", "a"): 1, ("qubit", "b"): 9}
    before = dict(runtime.busy_claims)
    with pytest.raises(RuntimeError, match="held by operation"):
        runtime._free_resources(operation)
    assert runtime.busy_claims == before

    runtime.busy_claims = {("qubit", "a"): 1, ("qubit", "b"): 1}
    runtime._free_resources(operation)
    assert runtime.busy_claims == {}


def test_body_done_invalid_callbacks_leave_all_runtime_and_collaborator_state_unchanged():
    """Unknown, premature, and duplicate completion callbacks fail before any observable mutation."""
    scheduled = make_operation(1, scheduled_start_round=5)
    runtime, engine, controller, factory = make_runtime(scheduled)
    runtime.load_program(ExecutionProgram((scheduled,)))

    unknown = make_operation(99)
    before = mutable_runtime_state(runtime, engine, controller, factory)
    with pytest.raises(RuntimeError, match="unindexed"):
        runtime.body_done(unknown)
    assert mutable_runtime_state(runtime, engine, controller, factory) == before

    before = mutable_runtime_state(runtime, engine, controller, factory)
    with pytest.raises(RuntimeError, match="before it starts"):
        runtime.body_done(scheduled)
    assert mutable_runtime_state(runtime, engine, controller, factory) == before

    engine.release_next()
    engine.now = 8
    runtime.body_done(scheduled)
    before = mutable_runtime_state(runtime, engine, controller, factory)
    with pytest.raises(RuntimeError, match="already complete"):
        runtime.body_done(scheduled)
    assert mutable_runtime_state(runtime, engine, controller, factory) == before


def test_body_done_preserves_valid_release_hook_issue_and_completion_order():
    """A valid completion frees resources before successor issue and preserves both controller hooks."""
    root = make_operation(1, qubits=("shared",))
    successor = make_operation(2, qubits=("shared",), predecessors=(1,))
    claims = {
        1: [ResourceClaim("qubit", frozenset({"shared"}))],
        2: [ResourceClaim("qubit", frozenset({"shared"}))],
    }
    runtime, engine, controller, _ = make_runtime(root, successor, claims=claims)
    runtime.load_program(ExecutionProgram((root, successor)))
    engine.events.clear()

    engine.now = 11
    runtime.body_done(root)

    assert engine.events == [
        ("log", "ExecutionRuntime", "operation-1 body done"),
        ("before", 1),
        ("can_start", 2),
        ("issue", 2, 0),
        ("after", 1),
    ]
    assert runtime.busy_claims == {("qubit", "shared"): 2}
    assert runtime.workload_complete is False

    engine.events.clear()
    engine.now = 13
    runtime.body_done(successor)
    assert runtime.workload_complete is True
    assert runtime.last_finish_time == 13
    assert engine.events == [
        ("log", "ExecutionRuntime", "operation-2 body done"),
        ("before", 2),
        ("log", "ExecutionRuntime", "QPU finished. All 2 operations are physically complete; decoder may still be draining."),
        ("after", 2),
    ]


def test_waiting_blocked_successor_checks_only_direct_feedback_and_boundary_state():
    """The waiting query ignores nonfeedback readiness dimensions for direct successors."""
    predecessor = make_operation(1)
    blocked = make_operation(2, predecessors=(1,), blocked_by=1, scheduled_start_round=8)
    unblocked = make_operation(3, predecessors=(1,))
    runtime, _, controller, _ = make_runtime(predecessor, blocked, unblocked, gates=False)
    controller.allowed[2] = False
    runtime.load_program(ExecutionProgram((predecessor, blocked, unblocked)))

    assert runtime.waiting_blocked_successor(1) is True
    runtime.decode_released.add(2)
    assert runtime.waiting_blocked_successor(1) is False
    controller.gates_start_on_round_boundaries = True
    assert runtime.waiting_blocked_successor(1) is True
    runtime.started.add(2)
    assert runtime.waiting_blocked_successor(1) is False


def test_round_boundary_retry_is_inert_off_and_notes_each_eligible_retry_before_start():
    """Boundary cadence is inert when off and notes a supplied patch immediately before an eligible retry."""
    predecessor = make_operation(1)
    blocked = make_operation(2, predecessors=(1,), blocked_by=1)
    runtime, engine, controller, _ = make_runtime(predecessor, blocked, gates=False)
    runtime.load_program(ExecutionProgram((predecessor, blocked)))
    runtime.decode_released.add(2)
    runtime.state_ready.add(2)
    runtime.schedule_released.add(2)
    runtime.dependencies_remaining[2] = 0
    engine.events.clear()

    runtime.start_released_successors_on_boundary(1, patch="patch-a")
    assert engine.events == []
    assert runtime.started == {1}

    controller.gates_start_on_round_boundaries = True
    runtime.start_released_successors_on_boundary(1, patch="patch-a")
    assert engine.events == [
        ("boundary", "patch-a"),
        ("can_start", 2),
        ("issue", 2, 0),
    ]
    assert controller.boundaries == ["patch-a"]


def test_idle_round_consumption_prefers_patches_and_is_destructive():
    """Idle-round consumption prefers truthy patches, counts duplicates once, and removes used entries."""
    patched = make_operation(1, qubits=("q",), patches=("p", "p", "missing"))
    runtime, _, _, _ = make_runtime(patched)
    runtime.idle_rounds_by_patch = {"p": 3, "q": 8}
    assert runtime.consume_idle_rounds(patched) == 3
    assert runtime.idle_rounds_by_patch == {"q": 8}

    fallback = make_operation(2, qubits=("q",), patches=())
    assert runtime.consume_idle_rounds(fallback) == 8
    assert runtime.idle_rounds_by_patch == {}


def test_idle_round_consumption_deliberately_does_not_roll_back_earlier_pops():
    """An invalid later idle identity fails naturally without restoring an earlier consumed value."""
    operation = make_operation(1, patches=("valid", []))
    runtime, _, _, _ = make_runtime(operation)
    runtime.idle_rounds_by_patch.update({"valid": 4, "other": 1})

    with pytest.raises(TypeError):
        runtime.consume_idle_rounds(operation)
    assert runtime.idle_rounds_by_patch == {"other": 1}


def test_decisions_keep_release_and_result_timestamps_distinct_and_latched():
    """Timing-only decisions latch feedback separately from result return and preserve early release."""
    operation = make_operation(1, blocked_by=0, clifford=False)
    runtime, engine, controller, factory = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))

    engine.now = 2
    runtime.on_decision(Decision(1, releases_operation=1))
    assert runtime.decode_released == {1}
    assert runtime.decode_release_time == {1: 2}
    assert runtime.result_return_time_by_operation == {}
    assert runtime.started == set()

    engine.now = 4
    factory.release()
    assert runtime.op_start_time == {1: 4}
    assert [issued.id for issued, _ in controller.issued] == [1]

    engine.now = 6
    runtime.on_decision(Decision(1, releases_operation=0))
    assert runtime.result_return_time_by_operation == {1: 6}
    assert runtime.decode_release_time == {1: 2}

    engine.now = 8
    runtime.on_decision(Decision(1, releases_operation=False))
    runtime.on_decision(Decision(1, releases_operation=True))
    assert runtime.result_return_time_by_operation == {1: 8}
    assert runtime.decode_release_time == {1: 8}


def test_unknown_decision_target_fails_before_mutation_and_unblocked_release_is_recorded():
    """Unknown decisions fail naturally, while an unblocked release still records its timing state."""
    operation = make_operation(1)
    runtime, engine, controller, factory = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))
    before = mutable_runtime_state(runtime, engine, controller, factory)

    with pytest.raises(KeyError):
        runtime.on_decision(Decision(99))
    assert mutable_runtime_state(runtime, engine, controller, factory) == before

    engine.now = 9
    runtime.on_decision(Decision(1))
    assert runtime.decode_released == {1}
    assert runtime.decode_release_time == {1: 9}



def test_zero_finish_time_needs_physical_completion_state_for_interpretation():
    """A zero finish time can describe unloaded, empty, or tick-zero-completed timing state."""
    unloaded, _, _, _ = make_runtime()
    assert unloaded.last_finish_time == 0
    assert unloaded.workload_complete is False

    empty, _, _, _ = make_runtime()
    empty.load_program(ExecutionProgram(()))
    assert empty.last_finish_time == 0
    assert empty.workload_complete is True

    operation = make_operation(1)
    completed, _, _, _ = make_runtime(operation)
    completed.load_program(ExecutionProgram((operation,)))
    completed.body_done(operation)
    assert completed.last_finish_time == 0
    assert completed.workload_complete is True

def test_timing_endpoints_remain_distinct_from_physical_and_terminal_completion():
    """Timing-only start, body, release, and return endpoints retain distinct event meanings."""
    operation = make_operation(1, blocked_by=0)
    runtime, engine, _, _ = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))

    engine.now = 1
    runtime.on_decision(Decision(1, True))
    assert runtime.op_start_time == {1: 1}
    engine.now = 3
    runtime.body_done(operation)
    assert runtime.workload_complete is True
    assert runtime.body_done_time == {1: 3}
    assert runtime.last_finish_time == 3

    engine.now = 5
    runtime.on_decision(Decision(1, False))
    assert runtime.result_return_time_by_operation == {1: 5}
    assert runtime.decode_release_time == {1: 1}
    assert runtime.last_finish_time == 3
