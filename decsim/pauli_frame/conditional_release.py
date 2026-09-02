"""Conditional release: letting go of the operations that waited on a result.

When an operation's final result is in, every operation blocked on it may
start; that is one "conditional release" decision per waiting operation.
When nothing waits but the QPU itself needs the outcome, the decision is a
"result return". Every decision travels to the controller over the
frame-to-controller path, and a release then sends the real operation
command through the controller and on to the QPU.

The value of the outcome never matters here. Timing does not branch on it
(SWIPER's rule: a conditional instruction starts once its dependency is
fully decoded), so nothing in this file reads the result's bits.
"""

from typing import Callable, Optional

import decsim.message as message


class ConditionalRelease:
    """Turns one operation's final result into the decisions it unblocks."""

    def __init__(self, engine):
        self.engine = engine
        self.waiting_by_blocker: dict[int, list[int]] = {}
        self.controller = None
        self.deliver_decision: Optional[Callable] = None

    def connect(self, controller, deliver_decision: Callable) -> None:
        """Wire the return path: the controller relays each decision."""
        self.controller = controller
        self.deliver_decision = deliver_decision

    def register_blocked_operation(
        self, blocked_operation_id: int, blocking_operation_id: int
    ) -> None:
        """Note that one operation waits on another's logical measurement."""
        waiting = self.waiting_by_blocker.setdefault(blocking_operation_id, [])
        waiting.append(blocked_operation_id)

    def release_waiters(self, operation: message.Operation) -> None:
        """A final result arrived: send every decision it releases."""
        for decision in self.decisions_for(operation):
            if decision.releases_operation:
                instruction = "conditional release"
            else:
                instruction = "result return"
            self.engine.log(
                "PauliFrame",
                f"DISPATCH {instruction} for op#{decision.target_operation_id} "
                f"-> controller -> controller sequencer",
            )
            self.controller.relay_instruction(decision, self.deliver_decision)

    def decisions_for(
        self, operation: message.Operation
    ) -> list[message.Decision]:
        """The decisions one final result releases.

        One release per waiting operation; else a result return when the
        QPU needs the outcome; else nothing.
        """
        waiting = self.waiting_by_blocker.pop(operation.id, [])
        if waiting:
            return [message.Decision(operation_id) for operation_id in waiting]
        if operation.requires_result_return_to_qpu:
            return [message.Decision(operation.id, releases_operation=False)]
        return []
