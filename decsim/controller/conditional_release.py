"""Conditional release: letting go of the operations that waited on a result.

When an operation's final result is in, every operation blocked on it may
start; that is one "conditional release" decision per waiting operation.
When nothing waits but the QPU itself needs the outcome, the decision is a
"result return". Every decision travels to the controller over the
frame-to-controller path, and a release then sends the real operation
command through the controller and on to the QPU.

The value of the outcome never matters here: a conditional operation
starts once its dependency is fully decoded, whatever it decoded to.
Caune et al. 2410.05202 stalls the program on the decoder's status
register until decoding completes and then executes the second program
"conditionally on the received result" (lines 364-367, 1255-1262), and
the conditional gate costs the same either way, the qubit idling "for a
time equal to the gate's duration" when the result is 0. Sivak et al.
2211.09116 broadcasts the two decision bits to every control card,
which "run independent but synchronized control flows that include
conditional branching on these bits" (lines 1051-1056). So nothing in
this file reads the result's bits.
"""

from typing import Callable, Optional

import decsim.records.log_sources as log_sources
import decsim.records.program as program_records


class ConditionalRelease:
    """Turns one operation's final result into the decisions it unblocks."""

    def __init__(self, engine):
        self.engine = engine
        self.waiting_by_blocker: dict[int, list[int]] = {}
        self.controller = None
        self.deliver_decision: Optional[Callable] = None

    def connect(self, controller, deliver_decision: Callable) -> None:
        """Wire the return path, after both ends exist.

        The wiring is two-phase because the two ends need each other: a
        release has no way to reach the QPU except through the
        controller's instruction output, and that output is built with
        the execution runtime whose decision callback the release
        delivers into. One of the two has to be constructed first, so
        the release is constructed knowing nothing and told its
        controller here, once the root has both.
        """
        self.controller = controller
        self.deliver_decision = deliver_decision

    def register_blocked_operation(
        self, blocked_operation_id: int, blocking_operation_id: int
    ) -> None:
        """Note that one operation waits on another's logical measurement."""
        waiting = self.waiting_by_blocker.setdefault(blocking_operation_id, [])
        waiting.append(blocked_operation_id)

    def release_waiters(self, operation: program_records.Operation) -> None:
        """A final result arrived: send every decision it releases."""
        for decision in self.decisions_for(operation):
            if decision.releases_operation:
                instruction = "conditional release"
            else:
                instruction = "result return"
            # the decision leaves the frame's end of the
            # frame-to-controller path, so the line is sourced there
            self.engine.log(
                log_sources.PAULI_FRAME,
                f"DISPATCH {instruction} for op#{decision.target_operation_id} "
                f"-> controller -> controller sequencer",
            )
            self.controller.relay_instruction(decision, self.deliver_decision)

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
