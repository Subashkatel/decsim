"""Conditional release: letting go of the operations that waited on a result.

When an operation's final result is in, each operation blocked on it
gets a release decision; when none waits but the QPU needs the outcome,
the decision is a result return.

The outcome's value never matters: a conditional operation starts once
its dependency is decoded, whatever it decoded to. Caune et al.
2410.05202 stall on the decoder's status register and pad the
conditional gate to the same time either way; Sivak et al. 2211.09116
broadcast the decision bits to every control card, which branch on them
in lockstep. So nothing here reads the result's bits.
"""

import decsim.ports as ports
import decsim.records.program as program_records


class ConditionalRelease:
    """Turns one operation's final result into the decisions it unblocks."""

    dispatch = ports.Port(ports.DecisionDispatch)
    # the decision is delivered into the runtime at the QPU, one hop
    # after the dispatch sends it
    runtime = ports.Port(ports.OperationRuntime)

    def __init__(self) -> None:
        self.waiting_by_blocker: dict[int, list[int]] = {}

    def register_blocked_operation(
        self, blocked_operation_id: int, blocking_operation_id: int
    ) -> None:
        """Note that one operation waits on another's logical measurement."""
        waiting = self.waiting_by_blocker.setdefault(blocking_operation_id, [])
        waiting.append(blocked_operation_id)

    def release_waiters(self, operation: program_records.Operation) -> None:
        """A final result arrived: send every decision it releases."""
        deliver = self.runtime.on_decision
        for decision in self.decisions_for(operation):
            # the decision leaves by the frame's end of the
            # frame-to-controller path, which executes that send
            self.dispatch.dispatch_decision(decision, deliver)

    def decisions_for(
        self, operation: program_records.Operation
    ) -> list[program_records.Decision]:
        """The decisions one final result releases.

        One release per waiting operation; else a result return when the
        QPU needs the outcome; else nothing.
        """
        waiting = self.waiting_by_blocker.pop(operation.id, [])
        if waiting:
            return [
                program_records.Decision(operation_id)
                for operation_id in waiting
            ]
        if operation.requires_result_return_to_qpu:
            return [
                program_records.Decision(operation.id, releases_operation=False)
            ]
        return []
