"""The frame's end of the frame-to-controller path.

A decision leaves the frame side and lands at the controller, so the
send is executed here: gem5 bills a transfer to the port it left by
(tmp/resources/gem5/src/mem/packet.hh:424-431), and OMNeT++ refuses a
module that sends a message it does not own
(tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334). The line that
narrates the dispatch is sourced at the end that executes it. What the
controller then does with the decision is its own
(controller/instruction_output.py), reached at the landing.

A card that prices no path has no fabric at all: the decision is then
available at the controller in the same instant, and only the
controller's own decision-to-pulse cost stands.
"""

import functools

import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records


class DecisionDispatch:
    """Sends one released decision over frame_to_controller."""

    def __init__(self, engine, link, controller) -> None:
        self.engine = engine
        self.link = link
        self.controller = controller

    def dispatch_decision(self, decision, deliver) -> None:
        """Send one decision to the controller; deliver runs at the QPU."""
        self._log_dispatch(decision)
        landed = functools.partial(self._at_the_controller, decision, deliver)
        if self.link is None:
            self.engine.schedule(0, landed, label="pauli frame->controller")
            return
        attribution = transfer_records.TransferAttribution(
            operation_id=decision.target_operation_id,
            patch_ids=(),
            window_id=None,
            first_round=None,
            last_round=None,
        )
        self.link.send(
            transfer_records.LinkPath.FRAME_TO_CONTROLLER,
            None,
            self.engine.now,
            attribution,
            landed,
        )

    def _at_the_controller(self, decision, deliver, _transfer=None) -> None:
        """The decision is at the controller: its output takes it on."""
        self.controller.relay_instruction(decision, deliver)

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
