"""Which operation runs when: readiness, resource ownership, timestamps.

An operation starts when its predecessors are done, its scheduled start
round has passed, its magic state (if any) is ready, its feedback release
(if any) has arrived and the controller's issuer lets it onto the QPU.
Resources (qubits, patches) are claimed at request and freed when the
body is done; two operations never hold one resource without a
dependency edge between them. The timestamp maps are the run's record of
every operation's life. The issuer hears each start through on_started
(SimPy's callback on the event): the runtime never receives a call back
from the controller.
"""

import functools
import types
from typing import Any, Callable, Protocol, runtime_checkable

import decsim.engine
import decsim.message as message


@runtime_checkable
class MagicStateFactory(Protocol):
    """Where a non-Clifford operation gets its magic state."""

    engine: Any

    def request(self, op_id: int, callback: Callable[[], None]):
        """Ask for one state; callback runs once it is ready."""

    def shutdown(self) -> None:
        """Stop producing; the workload is complete."""


class ResourceLedger:
    """Which operation holds which resource; one holder per resource."""

    def __init__(self, claims_by_operation_id):
        copied = dict(claims_by_operation_id)
        self.claims_by_operation_id = types.MappingProxyType(copied)
        self.holder_by_resource = {}

    def claim(self, operation: message.Operation, name_of: Callable) -> None:
        """Claim every resource of the operation, or none of them."""
        distinct_qubits = set(operation.qubits)
        if len(distinct_qubits) != len(operation.qubits):
            raise RuntimeError(
                f"{operation.name} lists a qubit more than once: "
                f"{operation.qubits}"
            )
        claims = self.claims_by_operation_id[operation.id]
        keys_to_claim = []
        for key in _resource_keys(claims):
            holder_id = self._holder_of(key, keys_to_claim, operation.id)
            if holder_id is None:
                keys_to_claim.append(key)
                continue
            holder_name = name_of(holder_id)
            kind, resource_id = key
            raise RuntimeError(
                f"{operation.name} and {holder_name} share {kind} "
                f"resource {resource_id!r} but have no dependency edge. "
                "The operation list is missing "
                "program-order wiring (run it through _wire_circuit / a "
                "frontend)"
            )
        for key in keys_to_claim:
            self.holder_by_resource[key] = operation.id

    def release(self, operation: message.Operation) -> None:
        """Free every resource the operation holds."""
        claims = self.claims_by_operation_id[operation.id]
        held_keys = _resource_keys(claims)
        for key in dict.fromkeys(held_keys):
            holder_id = self.holder_by_resource.get(key)
            assert holder_id == operation.id, (
                f"{operation.name} releases {key!r} held by {holder_id!r}"
            )
            del self.holder_by_resource[key]

    def _holder_of(self, key: tuple, prospective_keys: list, operation_id):
        """Who holds the key: a live holder, this claim itself, or nobody."""
        if key in self.holder_by_resource:
            return self.holder_by_resource[key]
        if key in prospective_keys:
            return operation_id
        return None


class ExecutionRuntime:
    """Own the program DAG, readiness, and timestamps.

    The ledger owns resources. The timestamp maps (op_start_time,
    body_done_time, decode_release_time, result_return_time_by_operation)
    are keyed by operation id; membership in a map is the operation's
    lifecycle state.
    """

    def __init__(
        self,
        engine: decsim.engine.Engine,
        *,
        issuer,
        factory,
        resource_claims_by_operation_id,
    ):
        self.engine = engine
        self.issuer = issuer
        self.factory = factory
        self.resources = ResourceLedger(resource_claims_by_operation_id)
        self.program = None
        self.operations = {}
        self.dependencies_remaining = {}
        self.successors = {}
        self.schedule_released = set()
        self.requested = set()
        self.state_ready = set()
        self.op_start_time = {}
        self.body_done_time = {}
        self.decode_release_time = {}
        self.result_return_time_by_operation = {}
        self.last_finish_time = 0

    @property
    def workload_complete(self) -> bool:
        """Every operation of the loaded program has finished its body."""
        if self.program is None:
            return False
        indexed = self.operations.keys()
        finished = self.body_done_time.keys()
        return indexed == finished

    def load_program(self, program: message.ExecutionProgram) -> None:
        """Index the operations, build the dependency graph, start the roots."""
        self.program = program
        for operation in program.operations:
            self.operations[operation.id] = operation
            self.dependencies_remaining[operation.id] = len(
                operation.predecessors
            )
            self.successors[operation.id] = []
        for operation in program.operations:
            for predecessor_id in operation.predecessors:
                self.successors[predecessor_id].append(operation.id)
        for operation in program.operations:
            self._release_at_scheduled_start(operation)
        for operation in program.operations:
            self._attempt_start(operation)

    def operation_started(
        self, operation: message.Operation, boundary_tick: int
    ) -> None:
        """The QPU has the operation's command: it starts at this boundary."""
        self.op_start_time[operation.id] = boundary_tick

    def body_done(self, operation: message.Operation) -> None:
        """A body finished: record it, free resources, release successors."""
        assert operation.id in self.operations, (
            f"operation {operation.id!r} is not in the program"
        )
        assert operation.id in self.op_start_time, (
            f"{operation.name} finished before it started"
        )
        assert operation.id not in self.body_done_time, (
            f"{operation.name} finished twice"
        )
        self.body_done_time[operation.id] = self.engine.now
        self.last_finish_time = max(self.last_finish_time, self.engine.now)
        self.engine.log("ExecutionRuntime", f"{operation.name} body done")
        self.resources.release(operation)
        self.issuer.before_successor_release(operation)
        for successor_id in self.successors[operation.id]:
            self.dependencies_remaining[successor_id] -= 1
            if self.dependencies_remaining[successor_id] == 0:
                successor = self.operations[successor_id]
                self._attempt_start(successor)
        if self.workload_complete:
            operation_count = len(self.operations)
            self.engine.log(
                "ExecutionRuntime",
                f"QPU finished. All {operation_count} operations are "
                "physically complete; decoder may still be draining.",
            )
        waits_for_blocked = self.waiting_blocked_successor(operation.id)
        self.issuer.after_successor_release(
            operation, waits_for_blocked, self.workload_complete
        )

    def waiting_blocked_successor(self, operation_id) -> bool:
        """True while a feedback-blocked successor awaits its decode release."""
        for successor_id in self.successors[operation_id]:
            successor = self.operations[successor_id]
            if successor.blocked_by is None:
                continue
            if successor.id in self.op_start_time:
                continue
            if successor.id not in self.decode_release_time:
                return True
        return False

    def retry_ready_operations(self) -> None:
        """Retry every state-ready operation after a cadence change."""
        for operation_id in sorted(self.state_ready):
            operation = self.operations[operation_id]
            self._maybe_begin(operation)

    def on_decision(self, decision: message.Decision) -> None:
        """A decision reached the controller: a release starts its operation.

        A result return is only recorded; a release is recorded and the
        blocked operation tries to start.
        """
        operation_id = decision.target_operation_id
        operation = self.operations[operation_id]
        if not decision.releases_operation:
            self.result_return_time_by_operation[operation_id] = self.engine.now
            self.engine.log(
                "ExecutionRuntime",
                f"received result return for {operation.name}",
            )
            return
        assert operation.blocked_by is not None, (
            f"a release targets {operation.name}, which is not feedback-blocked"
        )
        assert operation_id not in self.decode_release_time, (
            f"{operation.name} was released twice"
        )
        self.decode_release_time[operation_id] = self.engine.now
        self.engine.log(
            "ExecutionRuntime",
            f"CONSUMED release for {operation.name}; now trying to start",
        )
        self._maybe_begin(operation)

    def _release_at_scheduled_start(self, operation: message.Operation) -> None:
        round_ticks = self.issuer.round_ticks_for(operation)
        release_tick = operation.scheduled_start_round * round_ticks
        if release_tick == 0:
            self.schedule_released.add(operation.id)
            return
        release = functools.partial(self._release_scheduled, operation)
        self.engine.schedule(
            release_tick, release, label=f"scheduled-start({operation.name})"
        )

    def _release_scheduled(self, operation: message.Operation) -> None:
        self.schedule_released.add(operation.id)
        self._attempt_start(operation)

    def _attempt_start(self, operation: message.Operation) -> None:
        """Claim resources and ask for the magic state once the DAG allows."""
        if not self._is_admissible(operation):
            return
        self.resources.claim(operation, self._operation_name)
        self.requested.add(operation.id)
        if not operation.needs_magic_state:
            self._state_became_ready(operation)
            return
        self.engine.log(
            "ExecutionRuntime",
            f"{operation.name} needs a magic state; asking the factory",
        )
        on_ready = functools.partial(self._state_became_ready, operation)
        self.factory.request(operation.id, on_ready)

    def _is_admissible(self, operation: message.Operation) -> bool:
        """Predecessors done, schedule released, and not yet requested."""
        if self.dependencies_remaining[operation.id] != 0:
            return False
        if operation.id not in self.schedule_released:
            return False
        return operation.id not in self.requested

    def _operation_name(self, operation_id) -> str:
        return self.operations[operation_id].name

    def _state_became_ready(self, operation: message.Operation) -> None:
        self.state_ready.add(operation.id)
        self._maybe_begin(operation)

    def _maybe_begin(self, operation: message.Operation) -> None:
        """Issue the operation when every start gate is open."""
        if operation.id in self.op_start_time:
            return
        if operation.id not in self.state_ready:
            return
        is_blocked = operation.blocked_by is not None
        if is_blocked and operation.id not in self.decode_release_time:
            return
        if not self.issuer.can_start(operation):
            return
        # A provisional stamp marks the operation as started before the
        # issuer issues it: issuing can retry ready operations, and a
        # reentrant _maybe_begin must not issue this one twice. The QPU's
        # actual start boundary then replaces the stamp (operation_started).
        self.op_start_time[operation.id] = self.engine.now
        on_started = functools.partial(self.operation_started, operation)
        self.issuer.issue_operation(operation, on_started)


def _resource_keys(claims) -> list[tuple]:
    """(kind, resource id) for every claim, in a stable order."""
    keys = []
    for claim in claims:
        claim_keys = _claim_keys(claim)
        keys.extend(claim_keys)
    return keys


def _claim_keys(claim) -> list[tuple]:
    keys = []
    ordered_ids = sorted(claim.ids, key=repr)
    for resource_id in ordered_ids:
        keys.append((claim.kind, resource_id))
    return keys
