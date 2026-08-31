"""Track operation readiness, resource ownership, and execution timestamps.

An operation starts when its predecessors are done, its scheduled start
round has passed, its magic state (if any) is ready, its feedback release
(if any) has arrived and the controller lets it onto the QPU. Resources
(qubits, patches) are claimed at request and freed when the body is done;
two operations never hold one resource without a dependency edge between
them.
"""
from __future__ import annotations
from types import MappingProxyType

from ..message import Decision, ExecutionProgram


class ResourceLedger:
    """Which operation holds which resource; one holder per resource."""

    def __init__(self, claims_by_operation_id):
        self.claims = MappingProxyType(dict(claims_by_operation_id))
        self.busy_claims = {}

    def claim(self, operation, name_of) -> None:
        if len(set(operation.qubits)) != len(operation.qubits):
            raise RuntimeError(
                f"{operation.name} lists a qubit more than once: {operation.qubits}")
        keys_to_claim = []
        prospective_keys = set()
        for claim in self.claims[operation.id]:
            for resource_id in sorted(claim.ids, key=repr):
                key = (claim.kind, resource_id)
                if key in self.busy_claims:
                    holder_id = self.busy_claims[key]
                elif key in prospective_keys:
                    holder_id = operation.id
                else:
                    keys_to_claim.append(key)
                    prospective_keys.add(key)
                    continue
                raise RuntimeError(
                    f"{operation.name} and {name_of(holder_id)} share {claim.kind} "
                    f"resource {resource_id!r} but have no dependency edge. "
                    "The operation list is missing "
                    "program-order wiring (run it through _wire_circuit / a frontend)")
        for key in keys_to_claim:
            self.busy_claims[key] = operation.id

    def release(self, operation) -> None:
        release_keys = []
        unique_release_keys = set()
        for claim in self.claims[operation.id]:
            for resource_id in claim.ids:
                key = (claim.kind, resource_id)
                if key not in unique_release_keys:
                    release_keys.append(key)
                    unique_release_keys.add(key)
        missing_holder = object()
        for kind, resource_id in release_keys:
            holder_id = self.busy_claims.get((kind, resource_id), missing_holder)
            if holder_id is missing_holder:
                raise RuntimeError(
                    f"{operation.name} cannot release unclaimed {kind} "
                    f"resource {resource_id!r}")
            if holder_id != operation.id:
                raise RuntimeError(
                    f"{operation.name} cannot release {kind} resource "
                    f"{resource_id!r} held by operation {holder_id!r}")
        for key in release_keys:
            del self.busy_claims[key]


class ExecutionRuntime:
    """Own the program DAG, readiness, and timestamps; the ledger owns resources."""

    def __init__(self, engine, *, controller, factory,
                 resource_claims_by_operation_id):
        self.engine = engine
        self.controller = controller
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
        self.idle_rounds_by_patch = {}
        self.last_finish_time = 0

    @property
    def workload_complete(self):
        """Every operation of the loaded program has finished its body."""
        return (self.program is not None and
                self.operations.keys() == self.body_done_time.keys())

    def load_program(self, program: ExecutionProgram) -> None:
        """Index the operations, build the dependency graph, start the roots."""
        self.program = program
        for operation in program.operations:
            self.operations[operation.id] = operation
            self.dependencies_remaining[operation.id] = len(operation.predecessors)
            self.successors[operation.id] = []
        for operation in program.operations:
            for predecessor_id in operation.predecessors:
                self.successors[predecessor_id].append(operation.id)
        for operation in program.operations:
            release_tick = (operation.scheduled_start_round *
                            self.controller.round_ticks_for(operation))
            if release_tick == 0:
                self.schedule_released.add(operation.id)
            else:
                self.engine.schedule(
                    release_tick,
                    lambda ready=operation: self._release_scheduled(ready),
                    label=f"scheduled-start({operation.name})")
        for operation in program.operations:
            self._attempt_start(operation)

    def _release_scheduled(self, operation):
        self.schedule_released.add(operation.id)
        self._attempt_start(operation)

    def _attempt_start(self, operation):
        if (self.dependencies_remaining[operation.id] != 0 or
                operation.id not in self.schedule_released or
                operation.id in self.requested):
            return
        self.resources.claim(operation, lambda holder_id: self.operations[holder_id].name)
        self.requested.add(operation.id)
        if operation.needs_magic_state:
            self.engine.log("ExecutionRuntime",
                            f"{operation.name} needs a magic state; asking the factory")
            self.factory.request(operation.id,
                lambda ready=operation: self._state_became_ready(ready))
        else:
            self._state_became_ready(operation)

    def _state_became_ready(self, operation):
        self.state_ready.add(operation.id)
        self._maybe_begin(operation)

    def _maybe_begin(self, operation):
        if operation.id in self.op_start_time or operation.id not in self.state_ready:
            return
        if (operation.blocked_by is not None and
                operation.id not in self.decode_release_time):
            return
        if not self.controller.can_start(operation):
            return
        # the provisional stamp marks the operation as started before
        # issue_operation runs: issuing can retry ready operations, and a
        # reentrant _maybe_begin must not issue this one twice; the QPU's
        # actual start boundary then replaces the stamp
        self.op_start_time[operation.id] = self.engine.now
        idle_rounds = self.consume_idle_rounds(operation)
        self.op_start_time[operation.id] = self.controller.issue_operation(operation, idle_rounds)

    def body_done(self, operation):
        """The QPU finished a body: record it, release successors, free resources."""
        if operation.id not in self.operations:
            raise RuntimeError(
                f"cannot complete unindexed operation id {operation.id!r}")
        if operation.id not in self.op_start_time:
            raise RuntimeError(
                f"cannot complete operation {operation.name} before it starts")
        if operation.id in self.body_done_time:
            raise RuntimeError(
                f"operation {operation.name} body is already complete")
        self.body_done_time[operation.id] = self.engine.now
        self.last_finish_time = max(self.last_finish_time, self.engine.now)
        self.engine.log("ExecutionRuntime", f"{operation.name} body done")
        self.resources.release(operation)
        self.controller.before_successor_release(operation)
        for successor_id in self.successors[operation.id]:
            self.dependencies_remaining[successor_id] -= 1
            if self.dependencies_remaining[successor_id] == 0:
                self._attempt_start(self.operations[successor_id])
        if self.workload_complete:
            self.engine.log("ExecutionRuntime",
                            f"QPU finished. All {len(self.operations)} operations are "
                            "physically complete; decoder may still be draining.")
        self.controller.after_successor_release(operation)

    def waiting_blocked_successor(self, operation_id):
        """True while a feedback-blocked successor of this operation awaits its decode release."""
        for successor_id in self.successors[operation_id]:
            successor = self.operations[successor_id]
            if successor.blocked_by is None or successor.id in self.op_start_time:
                continue
            if successor.id not in self.decode_release_time:
                return True
        return False

    def retry_ready_operations(self):
        """Retry every state-ready operation after a controller cadence change."""
        for operation_id in sorted(self.state_ready):
            self._maybe_begin(self.operations[operation_id])

    def record_idle_round(self, patch):
        """Account one emitted idle round against the patch that idled."""
        self.idle_rounds_by_patch[patch] = self.idle_rounds_by_patch.get(patch, 0) + 1

    def consume_idle_rounds(self, operation):
        """Take the idle rounds emitted on this operation's patches since the last consumer."""
        patches = operation.patches if operation.patches else operation.qubits
        return sum(self.idle_rounds_by_patch.pop(patch, 0) for patch in patches)

    def on_decision(self, decision: Decision):
        """A decode result arrived at the controller: record it, and release the blocked operation if it says so."""
        operation_id = decision.target_operation_id
        operation = self.operations[operation_id]
        if not decision.releases_operation:
            self.result_return_time_by_operation[operation_id] = self.engine.now
            self.engine.log("ExecutionRuntime",
                            f"received result return for {operation.name}")
            return
        if operation.blocked_by is None:
            raise RuntimeError(
                f"release decision targets {operation.name}, "
                "which is not feedback-blocked")
        if operation_id in self.decode_release_time:
            raise RuntimeError(
                f"{operation.name} was already released by an earlier decision")
        self.decode_release_time[operation_id] = self.engine.now
        self.engine.log("ExecutionRuntime",
                        f"CONSUMED release for {operation.name}; now trying to start")
        self._maybe_begin(operation)
