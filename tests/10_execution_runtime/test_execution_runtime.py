"""Behavior contracts for the timing-only execution runtime."""

from types import SimpleNamespace

import pytest

import decsim.frontends.execution_runtime as execution_runtime
import decsim.observe.runtime_stamps as runtime_stamps_module
from decsim.frontends.execution_runtime import ExecutionRuntime
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
    def __init__(self, engine, *, round_ticks=1):
        self.engine = engine
        self.round_ticks = round_ticks
        self.allowed = {}
        self.issued = []
        self.runtime = None
        self.stamps = None
        self.observed_at_issue = []
        self.lifecycle_observations = []

    def observe_lifecycle(self, boundary, operation_id):
        if self.runtime is not None:
            self.lifecycle_observations.append((
                boundary,
                operation_id,
                frozenset(self.runtime.started_operation_ids),
                frozenset(self.runtime.finished_operation_ids),
                frozenset(self.runtime.released_operation_ids),
            ))

    def round_ticks_for(self, operation):
        return self.round_ticks

    def can_start(self, operation):
        self.observe_lifecycle("can_start", operation.id)
        self.engine.events.append(("can_start", operation.id))
        return self.allowed.get(operation.id, True)

    def issue_operation(self, operation, on_started):
        self.observe_lifecycle("issue_operation", operation.id)
        self.issued.append((operation, on_started))
        if self.runtime is not None:
            self.observed_at_issue.append((
                operation.id,
                operation.id in self.runtime.started_operation_ids,
                self.stamps.op_start.get(operation.id),
            ))
        self.engine.events.append(("issue", operation.id))
        return self.engine.now

    def before_successor_release(self, operation):
        self.observe_lifecycle("before_successor_release", operation.id)
        self.engine.events.append(("before", operation.id))

    def after_successor_release(self, operation, waits_for_blocked, is_workload_complete):
        self.observe_lifecycle("after_successor_release", operation.id)
        self.engine.events.append(("after", operation.id))


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


def make_runtime(*operations, round_ticks=1, claims=None):
    engine = RecordingEngine()
    controller = RecordingController(engine, round_ticks=round_ticks)
    factory = RecordingFactory(engine)
    runtime = ExecutionRuntime(
        engine,
        issuer=controller,
        factory=factory,
        resource_claims_by_operation_id=claims if claims is not None else claims_for(*operations),
    )
    controller.runtime = runtime
    stamps = runtime_stamps_module.RuntimeStamps()
    runtime.operation_issued.connect(stamps.operation_issued)
    runtime.operation_started.connect(stamps.operation_started)
    runtime.body_finished.connect(stamps.body_finished)
    runtime.decode_released.connect(stamps.decode_released)
    runtime.result_returned.connect(stamps.result_returned)
    controller.stamps = stamps
    return runtime, engine, controller, factory, stamps


def mutable_runtime_state(runtime, engine, controller, factory):
    names = (
        "operations", "dependencies_remaining", "successors", "schedule_released",
        "requested", "state_ready", "started_operation_ids",
        "finished_operation_ids", "released_operation_ids",
    )
    copied = {}
    for name in names:
        value = getattr(runtime, name)
        copied[name] = value.copy()
    stamp_names = (
        "op_start", "body_done", "decode_release", "result_return",
    )
    for name in stamp_names:
        copied[name] = getattr(controller.stamps, name).copy()
    copied["program"] = runtime.program
    copied["last_finish"] = controller.stamps.last_finish
    copied["engine_events"] = list(engine.events)
    copied["scheduled"] = list(engine.scheduled)
    copied["issued"] = list(controller.issued)
    copied["observed_at_issue"] = list(controller.observed_at_issue)
    copied["lifecycle_observations"] = list(controller.lifecycle_observations)
    copied["factory_requests"] = list(factory.requests)
    return copied


def test_construction_preserves_collaborators_and_shallow_copies_claim_mapping():
    """Construction preserves collaborators and shallow-copies only the claim mapping."""
    operation = make_operation(1)
    claim_list = []
    supplied_claims = {operation.id: claim_list}
    runtime, engine, controller, factory, _ = make_runtime(operation, claims=supplied_claims)

    assert runtime.engine is engine
    assert runtime.issuer is controller
    assert runtime.factory is factory
    assert runtime.resources.claims_by_operation_id[operation.id] is claim_list
    supplied_claims[2] = []
    assert 2 not in runtime.resources.claims_by_operation_id
    claim_list.append(ResourceClaim("qubit", frozenset({"q"})))
    assert runtime.resources.claims_by_operation_id[operation.id] == claim_list
    with pytest.raises(TypeError):
        runtime.resources.claims_by_operation_id[3] = []

    bare_engine = object()
    bare_controller = object()
    bare_factory = object()
    bare_runtime = ExecutionRuntime(
        bare_engine,
        issuer=bare_controller,
        factory=bare_factory,
        resource_claims_by_operation_id={},
    )
    assert (bare_runtime.engine, bare_runtime.issuer, bare_runtime.factory) == (
        bare_engine, bare_controller, bare_factory
    )


def test_construction_starts_with_the_documented_empty_lifecycle():
    """A new runtime starts unloaded with empty lifecycle state and a zero finish time."""
    runtime, _, _, _, stamps = make_runtime()

    assert runtime.program is None
    assert runtime.workload_complete is False
    for name in ("operations", "dependencies_remaining", "successors"):
        assert getattr(runtime, name) == {}
    for name in (
        "schedule_released", "requested", "state_ready",
        "started_operation_ids", "finished_operation_ids",
        "released_operation_ids",
    ):
        assert getattr(runtime, name) == set()
    for name in ("op_start", "body_done", "decode_release", "result_return"):
        assert getattr(stamps, name) == {}
    assert stamps.last_finish == 0


def test_deleted_snapshot_projection_is_not_part_of_the_runtime_surface():
    """The deleted snapshot projection and record remain absent without compatibility shims."""
    runtime, _, _, _, _ = make_runtime()

    assert not hasattr(execution_runtime, "ExecutionSnapshot")
    assert not hasattr(runtime, "snapshot")


def test_deleted_lifecycle_latches_are_not_part_of_the_runtime_surface():
    """Deleted lifecycle latch sets stay absent before and after a complete operation."""
    operation = make_operation(1)
    runtime, _, _, _, _ = make_runtime(operation)

    for name in ("started", "done_bodies"):
        assert not hasattr(runtime, name)
    runtime.load_program(ExecutionProgram((operation,)))
    runtime.body_done(operation)
    for name in ("started", "done_bodies"):
        assert not hasattr(runtime, name)


def test_the_lifecycle_id_sets_are_the_only_membership_record():
    """The id sets match lifecycle membership at every controller observation boundary."""
    root = make_operation(1)
    blocked_successor = make_operation(2, predecessors=(1,), blocked_by=1)
    runtime, engine, controller, _, stamps = make_runtime(root, blocked_successor, )
    controller.allowed[2] = False

    runtime.load_program(ExecutionProgram((root, blocked_successor)))
    engine.now = 1
    runtime.body_done(root)
    engine.now = 2
    runtime.on_decision(Decision(2, releases_operation=True))
    controller.allowed[2] = True
    runtime.retry_ready_operations()
    engine.now = 3
    runtime.body_done(blocked_successor)

    assert controller.lifecycle_observations == [
        ("can_start", 1, frozenset(), frozenset(), frozenset()),
        ("issue_operation", 1, frozenset({1}), frozenset(), frozenset()),
        ("before_successor_release", 1, frozenset({1}), frozenset({1}), frozenset()),
        ("after_successor_release", 1, frozenset({1}), frozenset({1}), frozenset()),
        ("can_start", 2, frozenset({1}), frozenset({1}), frozenset({2})),
        ("can_start", 2, frozenset({1}), frozenset({1}), frozenset({2})),
        ("issue_operation", 2, frozenset({1, 2}), frozenset({1}), frozenset({2})),
        ("before_successor_release", 2, frozenset({1, 2}), frozenset({1, 2}), frozenset({2})),
        ("after_successor_release", 2, frozenset({1, 2}), frozenset({1, 2}), frozenset({2})),
    ]
    assert runtime.started_operation_ids == {1, 2}
    assert runtime.finished_operation_ids == {1, 2}
    assert runtime.released_operation_ids == {2}
    assert set(stamps.op_start) == {1, 2}
    assert set(stamps.body_done) == {1, 2}
    assert set(stamps.decode_release) == {2}


def test_empty_program_completes_physically():
    """An empty loaded program is complete."""
    runtime, engine, controller, factory, _ = make_runtime()
    program = ExecutionProgram(
        (),
        decode_operations=(object(),),
        dynamic_streams=(object(),),
        protected_regions=(object(),),
    )
    runtime.load_program(program)

    assert runtime.program is program
    assert runtime.workload_complete is True


def test_load_builds_ordered_dependency_edges_and_releases_successor_after_body():
    """Body completion consumes every ordered dependency edge before admitting a successor."""
    root = make_operation(1)
    successor = make_operation(2, predecessors=(1, 1))
    runtime, engine, controller, _, stamps = make_runtime(root, successor)
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
    assert stamps.body_done[1] == 4
    assert stamps.last_finish == 4


def test_load_deliberately_leaves_graph_validation_to_the_trusted_boundary():
    """Loading deliberately accepts duplicate identities and cycles while unknown predecessors fail naturally."""
    first = make_operation(1, name="first")
    replacement = make_operation(1, name="replacement")
    duplicate_runtime, _, _, _, _ = make_runtime(first, replacement)
    duplicate_runtime.load_program(ExecutionProgram((first, replacement)))
    assert duplicate_runtime.operations[1] is replacement

    left = make_operation(2, predecessors=(3,))
    right = make_operation(3, predecessors=(2,))
    cyclic_runtime, _, cyclic_controller, _, _ = make_runtime(left, right)
    cyclic_runtime.load_program(ExecutionProgram((left, right)))
    assert cyclic_runtime.requested == set()
    assert cyclic_controller.issued == []

    orphan = make_operation(4, predecessors=(99,))
    orphan_runtime, _, _, _, _ = make_runtime(orphan)
    with pytest.raises(KeyError):
        orphan_runtime.load_program(ExecutionProgram((orphan,)))


def test_schedule_release_uses_raw_cadence_product_without_local_validation():
    """Timing-only scheduling passes every nonzero raw cadence product to the engine unchanged."""
    immediate = make_operation(1, scheduled_start_round=0)
    negative = make_operation(2, scheduled_start_round=1)
    fractional = make_operation(3, scheduled_start_round=1.5)
    runtime, _, controller, _, _ = make_runtime(immediate, negative, fractional, round_ticks=-4)
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
    runtime, _, controller, factory, _ = make_runtime(operation, round_ticks=3)
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
    runtime, engine, controller, factory, stamps = make_runtime(
        operation, claims=claims
    )
    runtime.load_program(ExecutionProgram((operation,)))

    assert runtime.resources.holder_by_resource == {("qubit", "data"): 1}
    assert runtime.requested == {1}
    assert runtime.state_ready == set()
    assert controller.issued == []
    assert engine.events[-1] == ("factory_request", 1)

    engine.now = 7
    factory.release()
    factory.release()
    assert stamps.op_start == {1: 7}
    assert [issued for issued, _ in controller.issued] == [operation]
    assert controller.observed_at_issue == [(1, True, 7)]


def test_feedback_and_controller_cadence_are_independent_start_gates():
    """A blocked operation starts only after feedback readiness and controller cadence both allow it."""
    operation = make_operation(1, blocked_by=0)
    runtime, engine, controller, _, stamps = make_runtime(operation)
    controller.allowed[1] = False
    runtime.load_program(ExecutionProgram((operation,)))

    assert runtime.state_ready == {1}
    assert stamps.op_start == {}
    engine.now = 3
    runtime.on_decision(Decision(1, releases_operation=True))
    assert stamps.decode_release == {1: 3}
    assert stamps.op_start == {}

    controller.allowed[1] = True
    engine.now = 5
    runtime._maybe_begin(operation)
    assert stamps.op_start == {1: 5}


def test_readiness_callback_deliberately_does_not_recheck_schedule_or_claim_admission():
    """A direct readiness callback deliberately trusts its admission path instead of rechecking it."""
    operation = make_operation(1, scheduled_start_round=9)
    runtime, engine, controller, _, stamps = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))
    assert runtime.requested == set()

    engine.now = 2
    runtime._state_became_ready(operation)

    assert stamps.op_start == {1: 2}
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
    runtime, _, _, _, _ = make_runtime(operation, claims=claims)
    runtime.operations[1] = operation
    runtime.resources.claim(operation, lambda holder_id: runtime.operations[holder_id].name)
    assert list(runtime.resources.holder_by_resource) == [
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
    conflict_runtime, _, _, _, _ = make_runtime(contender, claims=conflict_claims)
    conflict_runtime.operations.update({2: contender, 3: holder})
    conflict_runtime.resources.holder_by_resource[("qubit", "busy")] = 3
    before = dict(conflict_runtime.resources.holder_by_resource)
    with pytest.raises(RuntimeError, match="share qubit resource"):
        conflict_runtime.resources.claim(contender, lambda holder_id: conflict_runtime.operations[holder_id].name)
    assert conflict_runtime.resources.holder_by_resource == before


def test_claim_rejects_duplicate_operands_and_duplicate_typed_keys_before_publication():
    """Duplicate operands or typed keys fail before any resource ownership is published."""
    duplicate_qubits = make_operation(1, qubits=("q", "q"))
    runtime, _, _, _, _ = make_runtime(duplicate_qubits)
    runtime.operations[1] = duplicate_qubits
    with pytest.raises(RuntimeError, match="more than once"):
        runtime.resources.claim(duplicate_qubits, lambda holder_id: runtime.operations[holder_id].name)
    assert runtime.resources.holder_by_resource == {}

    duplicate_key = make_operation(2)
    duplicate_claims = {
        2: [
            ResourceClaim("qubit", frozenset({"q"})),
            ResourceClaim("qubit", frozenset({"q"})),
        ]
    }
    duplicate_runtime, _, _, _, _ = make_runtime(duplicate_key, claims=duplicate_claims)
    duplicate_runtime.operations[2] = duplicate_key
    with pytest.raises(RuntimeError, match="share qubit resource"):
        duplicate_runtime.resources.claim(duplicate_key, lambda holder_id: duplicate_runtime.operations[holder_id].name)
    assert duplicate_runtime.resources.holder_by_resource == {}


def test_claim_shape_and_mapping_completeness_use_natural_failures():
    """Missing claims and unhashable operands fail naturally while extra claim entries stay unused."""
    missing = make_operation(1)
    runtime, _, _, _, _ = make_runtime(missing, claims={99: []})
    runtime.operations[1] = missing
    with pytest.raises(KeyError):
        runtime.resources.claim(missing, lambda holder_id: runtime.operations[holder_id].name)
    with pytest.raises(KeyError):
        runtime.resources.release(missing)
    assert runtime.resources.holder_by_resource == {}

    valid = make_operation(3)
    extra_runtime, _, _, _, _ = make_runtime(valid, claims={3: [], 99: object()})
    extra_runtime.operations[3] = valid
    extra_runtime.resources.claim(valid, lambda holder_id: extra_runtime.operations[holder_id].name)
    assert extra_runtime.resources.holder_by_resource == {}

    unhashable = make_operation(2, qubits=([],))
    unhashable_runtime, _, _, _, _ = make_runtime(unhashable)
    unhashable_runtime.operations[2] = unhashable
    with pytest.raises(TypeError):
        unhashable_runtime.resources.claim(unhashable, lambda holder_id: unhashable_runtime.operations[holder_id].name)
    assert unhashable_runtime.resources.holder_by_resource == {}


def test_body_done_preserves_valid_release_hook_issue_and_completion_order():
    """A valid completion frees resources before successor issue and preserves both controller hooks."""
    root = make_operation(1, qubits=("shared",))
    successor = make_operation(2, qubits=("shared",), predecessors=(1,))
    claims = {
        1: [ResourceClaim("qubit", frozenset({"shared"}))],
        2: [ResourceClaim("qubit", frozenset({"shared"}))],
    }
    runtime, engine, controller, _, stamps = make_runtime(root, successor, claims=claims)
    runtime.load_program(ExecutionProgram((root, successor)))
    engine.events.clear()

    engine.now = 11
    runtime.body_done(root)

    assert engine.events == [
        ("log", "ExecutionRuntime", "operation-1 body done"),
        ("before", 1),
        ("can_start", 2),
        ("issue", 2),
        ("after", 1),
    ]
    assert runtime.resources.holder_by_resource == {("qubit", "shared"): 2}
    assert runtime.workload_complete is False

    engine.events.clear()
    engine.now = 13
    runtime.body_done(successor)
    assert runtime.workload_complete is True
    assert stamps.last_finish == 13
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
    runtime, _, controller, _, stamps = make_runtime(predecessor, blocked, unblocked, )
    controller.allowed[2] = False
    runtime.load_program(ExecutionProgram((predecessor, blocked, unblocked)))

    assert runtime.waiting_blocked_successor(1) is True
    runtime.released_operation_ids.add(2)
    assert runtime.waiting_blocked_successor(1) is False


def test_ready_retry_offers_state_ready_operations_in_identity_order():
    """A ready retry re-offers state-ready operations in identity order and starts only those the gates admit."""
    unloaded, _, _, _, _ = make_runtime()
    unloaded.retry_ready_operations()

    root = make_operation(1)
    never_released = make_operation(2, blocked_by=1)
    released_first = make_operation(9, blocked_by=1)
    released_second = make_operation(24, blocked_by=1)
    operations = (root, never_released, released_first, released_second)
    runtime, engine, controller, _, stamps = make_runtime(*operations)
    runtime.load_program(ExecutionProgram(operations))
    assert runtime.state_ready == {1, 2, 9, 24}
    assert runtime.started_operation_ids == {1}

    engine.events.clear()
    runtime.retry_ready_operations()
    assert engine.events == []
    assert runtime.state_ready == {1, 2, 9, 24}
    assert runtime.started_operation_ids == {1}

    runtime.released_operation_ids.update({9, 24})
    controller.allowed[24] = False
    engine.now = 6
    engine.events.clear()
    runtime.retry_ready_operations()
    assert engine.events == [
        ("can_start", 9),
        ("issue", 9),
        ("can_start", 24),
    ]
    assert runtime.started_operation_ids == {1, 9}
    assert stamps.op_start[9] == 6

    controller.allowed[24] = True
    engine.now = 8
    engine.events.clear()
    runtime.retry_ready_operations()
    assert engine.events == [("can_start", 24), ("issue", 24)]
    assert [operation.id for operation, _ in controller.issued] == [1, 9, 24]
    assert stamps.op_start[24] == 8


def test_decisions_keep_release_and_result_timestamps_distinct_and_latched():
    """Timing-only decisions latch feedback separately from result return and preserve early release."""
    operation = make_operation(1, blocked_by=0, clifford=False)
    runtime, engine, controller, factory, stamps = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))

    engine.now = 2
    runtime.on_decision(Decision(1, releases_operation=1))
    assert stamps.decode_release == {1: 2}
    assert stamps.result_return == {}
    assert stamps.op_start == {}

    engine.now = 4
    factory.release()
    assert stamps.op_start == {1: 4}
    assert [issued.id for issued, _ in controller.issued] == [1]

    engine.now = 6
    runtime.on_decision(Decision(1, releases_operation=0))
    assert stamps.result_return == {1: 6}
    assert stamps.decode_release == {1: 2}

    engine.now = 8
    runtime.on_decision(Decision(1, releases_operation=False))
    assert stamps.result_return == {1: 8}
    assert stamps.decode_release == {1: 2}


def test_unknown_decision_target_fails_before_mutation_and_blocked_release_is_recorded():
    """Unknown decisions fail naturally, while a valid blocked release records its timing state."""
    operation = make_operation(1, blocked_by=0)
    runtime, engine, controller, factory, stamps = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))
    before = mutable_runtime_state(runtime, engine, controller, factory)

    with pytest.raises(KeyError):
        runtime.on_decision(Decision(99))
    assert mutable_runtime_state(runtime, engine, controller, factory) == before

    engine.now = 9
    runtime.on_decision(Decision(1))
    assert stamps.decode_release == {1: 9}
    assert stamps.op_start == {1: 9}


def test_zero_finish_time_needs_physical_completion_state_for_interpretation():
    """A zero finish time can describe unloaded, empty, or tick-zero-completed timing state."""
    unloaded, _, _, _, unloaded_stamps = make_runtime()
    assert unloaded_stamps.last_finish == 0
    assert unloaded.workload_complete is False

    empty, _, _, _, empty_stamps = make_runtime()
    empty.load_program(ExecutionProgram(()))
    assert empty_stamps.last_finish == 0
    assert empty.workload_complete is True

    operation = make_operation(1)
    completed, _, _, _, completed_stamps = make_runtime(operation)
    completed.load_program(ExecutionProgram((operation,)))
    completed.body_done(operation)
    assert completed_stamps.last_finish == 0
    assert completed.workload_complete is True

def test_timing_endpoints_remain_distinct_from_physical_and_terminal_completion():
    """Timing-only start, body, release, and return endpoints retain distinct event meanings."""
    operation = make_operation(1, blocked_by=0)
    runtime, engine, _, _, stamps = make_runtime(operation)
    runtime.load_program(ExecutionProgram((operation,)))

    engine.now = 1
    runtime.on_decision(Decision(1, True))
    assert stamps.op_start == {1: 1}
    engine.now = 3
    runtime.body_done(operation)
    assert runtime.workload_complete is True
    assert stamps.body_done == {1: 3}
    assert stamps.last_finish == 3

    engine.now = 5
    runtime.on_decision(Decision(1, False))
    assert stamps.result_return == {1: 5}
    assert stamps.decode_release == {1: 1}
    assert stamps.last_finish == 3
