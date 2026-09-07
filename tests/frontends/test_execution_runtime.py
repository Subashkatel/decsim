"""Which operation runs when, and who holds which qubit while it runs.

The runtime is the sequencer of one workload: an operation starts when
its predecessors are done, its scheduled start round has passed, its
magic state is ready, its feedback release has arrived and the
controller's issuer lets it onto the QPU. Every gate is independent, and
the last one to open starts the operation.

Resource ownership is the safety law under it: a qubit or a patch has
one holder, claimed when the operation is requested and freed when its
body is done, so two operations never hold one resource without a
dependency edge between them (the same rule a compiler's register
allocator keeps, and the reason a lattice-surgery schedule is a
dependency graph and not a list). A claim is all or nothing, so a
refused operation leaves the ledger as it was.

The engine and the issuer here are recording stand-ins, so every tick a
test asserts is one the test set.
"""

import pytest

import decsim.frontends.execution_runtime as execution_runtime
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.records.program as program_records


class RecordingEngine:
    """A clock a test moves by hand, and the log lines it was given."""

    def __init__(self):
        self.now = 0
        self.calls = []
        self.scheduled = []

    def schedule(self, delay, action, *, label):
        self.scheduled.append((delay, action, label))

    def log(self, component, message):
        self.calls.append(("log", component, message))


class RecordingIssuer:
    """The controller's issuer: which operations it let onto the QPU."""

    def __init__(self, engine, round_ticks=1):
        self.engine = engine
        self.round_ticks = round_ticks
        self.allowed_by_operation_id = {}
        self.issued = []

    def round_ticks_for(self, operation):
        del operation
        return self.round_ticks

    def can_start(self, operation):
        self.engine.calls.append(("can_start", operation.id))
        return self.allowed_by_operation_id.get(operation.id, True)

    def issue_operation(self, operation, on_started):
        del on_started
        self.issued.append(operation.id)
        self.engine.calls.append(("issue", operation.id))
        return self.engine.now

    def before_successor_release(self, operation):
        self.engine.calls.append(("before", operation.id))

    def after_successor_release(
        self, operation, waits_for_blocked, is_workload_complete
    ):
        del waits_for_blocked, is_workload_complete
        self.engine.calls.append(("after", operation.id))


class RecordingFactory:
    """The magic state factory: the requests it holds, released by hand."""

    def __init__(self, engine):
        self.engine = engine
        self.callbacks = []

    def request(self, op_id, callback):
        self.engine.calls.append(("factory_request", op_id))
        self.callbacks.append(callback)

    def release_the_first_state(self):
        callback = self.callbacks[0]
        callback()


def operation_named(operation_id, **fields):
    name = f"operation-{operation_id}"
    return program_records.Operation(operation_id, name, (), **fields)


def no_claims_for(operations):
    claims = {}
    for operation in operations:
        claims[operation.id] = []
    return claims


def runtime_over(operations, claims=None, round_ticks=1):
    """The runtime, its engine, its issuer, its factory and its stamps."""
    engine = RecordingEngine()
    issuer = RecordingIssuer(engine, round_ticks)
    factory = RecordingFactory(engine)
    if claims is None:
        claims = no_claims_for(operations)
    runtime = execution_runtime.ExecutionRuntime(
        engine,
        issuer=issuer,
        factory=factory,
        resource_claims_by_operation_id=claims,
    )
    stamps = runtime_stamps_module.RuntimeStamps()
    runtime.operation_issued.connect(stamps.operation_issued)
    runtime.operation_started.connect(stamps.operation_started)
    runtime.body_finished.connect(stamps.body_finished)
    runtime.decode_released.connect(stamps.decode_released)
    runtime.result_returned.connect(stamps.result_returned)
    return runtime, engine, issuer, factory, stamps


def name_of_holder(runtime):
    """The name the ledger prints for a holding operation."""

    def name_of(holder_id):
        holder = runtime.operations[holder_id]
        return holder.name

    return name_of


def test_a_program_with_no_operations_is_complete_at_once():
    runtime, _engine, _issuer, _factory, _stamps = runtime_over(())
    program = program_records.ExecutionProgram(())
    runtime.load_program(program)
    assert runtime.workload_complete is True


def test_a_successor_waits_for_every_dependency_edge_it_declares():
    """An operation named twice as a predecessor is waited for twice."""
    root = operation_named(1)
    successor = operation_named(2, predecessors=(1, 1))
    operations = (root, successor)
    runtime, engine, issuer, _factory, stamps = runtime_over(operations)
    program = program_records.ExecutionProgram(operations)

    runtime.load_program(program)
    assert runtime.dependencies_remaining == {1: 0, 2: 2}
    assert issuer.issued == [1]

    engine.now = 4
    runtime.body_done(root)
    assert runtime.dependencies_remaining[2] == 0
    assert issuer.issued == [1, 2]
    assert stamps.body_done[1] == 4


def test_an_operation_with_a_scheduled_start_round_waits_for_that_round():
    """The release is one engine event at round times the cadence."""
    root = operation_named(1)
    late = operation_named(2, scheduled_start_round=4)
    operations = (root, late)
    runtime, engine, issuer, _factory, _stamps = runtime_over(
        operations, round_ticks=10
    )
    program = program_records.ExecutionProgram(operations)

    runtime.load_program(program)

    assert issuer.issued == [1]
    assert engine.scheduled == [
        (40, engine.scheduled[0][1], "scheduled-start(operation-2)")
    ]


def test_a_scheduled_release_that_fires_twice_issues_the_operation_once():
    operation = operation_named(1, scheduled_start_round=2)
    operations = (operation,)
    runtime, engine, issuer, _factory, _stamps = runtime_over(
        operations, round_ticks=3
    )
    program = program_records.ExecutionProgram(operations)
    runtime.load_program(program)
    release = engine.scheduled[0][1]

    release()
    release()

    assert issuer.issued == [1]


def test_a_magic_state_operation_claims_its_qubits_then_waits_for_one():
    """The claim comes first, so the state is made for a run that can use it."""
    operation = program_records.Operation(
        1, "t-gate", ("data",), clifford=False
    )
    data_qubits = frozenset({"data"})
    claim = program_records.ResourceClaim("qubit", data_qubits)
    claims = {1: [claim]}
    operations = (operation,)
    runtime, engine, issuer, factory, stamps = runtime_over(
        operations, claims=claims
    )
    program = program_records.ExecutionProgram(operations)

    runtime.load_program(program)
    assert runtime.resources.holder_by_resource == {("qubit", "data"): 1}
    assert issuer.issued == []
    assert engine.calls[-1] == ("factory_request", 1)

    engine.now = 7
    factory.release_the_first_state()
    factory.release_the_first_state()
    assert issuer.issued == [1]
    assert stamps.op_start == {1: 7}


def test_a_blocked_operation_starts_when_both_of_its_gates_are_open():
    """Feedback readiness and the controller's cadence are two gates."""
    operation = operation_named(1, blocked_by=0)
    operations = (operation,)
    runtime, engine, issuer, _factory, stamps = runtime_over(operations)
    issuer.allowed_by_operation_id[1] = False
    program = program_records.ExecutionProgram(operations)
    runtime.load_program(program)

    engine.now = 3
    release = program_records.Decision(1, releases_operation=True)
    runtime.on_decision(release)
    assert stamps.decode_release == {1: 3}
    assert stamps.op_start == {}

    issuer.allowed_by_operation_id[1] = True
    engine.now = 5
    runtime.retry_ready_operations()
    assert stamps.op_start == {1: 5}


def test_a_claim_publishes_every_resource_of_the_operation_or_none():
    holder = operation_named(3)
    contender = program_records.Operation(2, "contender", ())
    free_qubits = frozenset({"free"})
    free_claim = program_records.ResourceClaim("qubit", free_qubits)
    busy_qubits = frozenset({"busy"})
    busy_claim = program_records.ResourceClaim("qubit", busy_qubits)
    claims = {2: [free_claim, busy_claim]}
    operations = (contender,)
    runtime, _engine, _issuer, _factory, _stamps = runtime_over(
        operations, claims=claims
    )
    runtime.operations.update({2: contender, 3: holder})
    runtime.resources.holder_by_resource[("qubit", "busy")] = 3

    name_of = name_of_holder(runtime)
    with pytest.raises(RuntimeError, match="share qubit resource"):
        runtime.resources.claim(contender, name_of)

    assert runtime.resources.holder_by_resource == {("qubit", "busy"): 3}


def test_an_operation_that_lists_one_qubit_twice_is_refused():
    operation = program_records.Operation(1, "twice", ("q", "q"))
    operations = (operation,)
    runtime, _engine, _issuer, _factory, _stamps = runtime_over(operations)
    runtime.operations[1] = operation

    name_of = name_of_holder(runtime)
    with pytest.raises(RuntimeError, match="lists a qubit more than once"):
        runtime.resources.claim(operation, name_of)

    assert runtime.resources.holder_by_resource == {}


def test_a_body_frees_its_resources_before_its_successor_is_issued():
    """The successor holds the qubit its predecessor just gave up."""
    shared = frozenset({"shared"})
    root = program_records.Operation(1, "root", ("shared",))
    successor = program_records.Operation(
        2, "successor", ("shared",), predecessors=(1,)
    )
    claims = {
        1: [program_records.ResourceClaim("qubit", shared)],
        2: [program_records.ResourceClaim("qubit", shared)],
    }
    operations = (root, successor)
    runtime, engine, _issuer, _factory, _stamps = runtime_over(
        operations, claims=claims
    )
    program = program_records.ExecutionProgram(operations)
    runtime.load_program(program)
    engine.calls.clear()

    engine.now = 11
    runtime.body_done(root)

    assert runtime.resources.holder_by_resource == {("qubit", "shared"): 2}
    assert engine.calls == [
        ("log", "ExecutionRuntime", "root body done"),
        ("before", 1),
        ("can_start", 2),
        ("issue", 2),
        ("after", 1),
    ]
    assert runtime.workload_complete is False


def test_the_ready_retry_offers_the_waiting_operations_in_identity_order():
    """A retry is deterministic: the ids ascend, whatever released them."""
    root = operation_named(1)
    waiting_late = operation_named(24, blocked_by=1)
    waiting_early = operation_named(9, blocked_by=1)
    operations = (root, waiting_late, waiting_early)
    runtime, engine, issuer, _factory, _stamps = runtime_over(operations)
    program = program_records.ExecutionProgram(operations)
    runtime.load_program(program)
    runtime.released_operation_ids.update({9, 24})
    engine.calls.clear()

    runtime.retry_ready_operations()

    assert issuer.issued == [1, 9, 24]


def test_a_retry_starts_nothing_that_no_release_has_reached():
    root = operation_named(1)
    waiting = operation_named(2, blocked_by=1)
    operations = (root, waiting)
    runtime, engine, issuer, _factory, _stamps = runtime_over(operations)
    program = program_records.ExecutionProgram(operations)
    runtime.load_program(program)
    engine.calls.clear()

    runtime.retry_ready_operations()

    assert engine.calls == []
    assert issuer.issued == [1]


def test_a_release_and_a_result_return_are_stamped_apart():
    """One decision releases an operation; a later one returns its result."""
    operation = operation_named(1, blocked_by=0, clifford=False)
    operations = (operation,)
    runtime, engine, _issuer, factory, stamps = runtime_over(operations)
    program = program_records.ExecutionProgram(operations)
    runtime.load_program(program)

    engine.now = 2
    release = program_records.Decision(1, releases_operation=True)
    runtime.on_decision(release)
    assert stamps.decode_release == {1: 2}
    assert stamps.result_return == {}

    engine.now = 4
    factory.release_the_first_state()
    assert stamps.op_start == {1: 4}

    engine.now = 6
    result = program_records.Decision(1, releases_operation=False)
    runtime.on_decision(result)
    assert stamps.result_return == {1: 6}
    assert stamps.decode_release == {1: 2}
