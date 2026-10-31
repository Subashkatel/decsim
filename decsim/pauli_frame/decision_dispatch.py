"""The frame's end of the frame-to-controller path.

A decision leaves the frame side, so the send is executed here: gem5
bills a transfer to the port it left by (src/mem/coherent_xbar.cc:354-357)
and OMNeT++ refuses a module that sends a message it does not own
(src/sim/csimplemodule.cc:333-334). A card that prices no path has no
fabric: the decision is at the controller at once, and only the
controller's decision-to-pulse cost stands. A notice that a result is
final rides the same path and stops at the controller, which holds the
decisions waiting on it (controller/final_result_gate.py); it drives no
pulse.
"""

import functools
from collections.abc import Callable

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records


class DecisionDispatch:
    """Sends one released decision over frame_to_controller."""

    # a card that prices no path leaves this unbound, and the decision
    # is at the controller in the same instant
    link = ports.Port(ports.Link, optional=True)
    instruction_output = ports.Port(ports.InstructionReceiver)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine

    def dispatch_decision(
        self,
        decision: program_records.Decision,
        deliver: Callable[[program_records.Decision], None],
    ) -> None:
        """Send one decision to the controller; deliver runs at the QPU."""
        self._log_dispatch(decision)
        landed = functools.partial(self._at_the_controller, decision, deliver)
        self._send(decision.target_operation_id, landed)

    def dispatch_final_result(
        self, operation_id: int, landed: Callable[[], None]
    ) -> None:
        """Send one final-result notice; landed runs at the controller."""
        self.engine.log(
            log_sources.PAULI_FRAME,
            f"DISPATCH final result for op#{operation_id} -> controller",
        )
        at_the_controller = functools.partial(_call_landed, landed)
        self._send(operation_id, at_the_controller)

    def _send(self, operation_id: int, landed) -> None:
        """Put one message on frame_to_controller; landed runs on arrival."""
        if self.link is None:
            self.engine.schedule(0, landed, label="pauli frame->controller")
            return
        attribution = transfer_records.TransferAttribution(
            owner_id=operation_id,
            patch_ids=(),
            window_id=None,
            first_round=None,
            last_round=None,
        )
        self.link.send(
            transfer_records.FRAME_TO_CONTROLLER,
            None,
            self.engine.now,
            attribution,
            landed,
        )

    def _at_the_controller(self, decision, deliver, _transfer=None) -> None:
        """The decision is at the controller: its output takes it on."""
        self.instruction_output.relay_instruction(decision, deliver)

    def _log_dispatch(self, decision: program_records.Decision) -> None:
        """Narrate the decision at the end it leaves by."""
        if decision.releases_operation:
            instruction = "conditional release"
        else:
            instruction = "result return"
        self.engine.log(
            log_sources.PAULI_FRAME,
            f"DISPATCH {instruction} for op#{decision.target_operation_id} "
            f"-> controller -> controller sequencer",
        )


def _call_landed(landed, _transfer=None) -> None:
    """Run a notice's landing, with or without the transfer that carried it."""
    landed()
