"""The decision leaves by the frame's end and lands at the controller.

gem5 bills a transfer to the port it left by (packet.hh:424-431) and
OMNeT++ refuses a module that sends a message it does not own
(csimplemodule.cc:333-334), so the frame side executes the
frame_to_controller send and narrates it; the controller's instruction
output is reached at the landing, one crossing later. A card that
prices no path has no fabric: the decision is then at the controller in
the same instant, and only the controller's own costs stand
(tests/controller/test_instruction_output.py).
"""

import dataclasses

import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.pauli_frame.decision_dispatch as decision_dispatch
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph_layout as graph_layout
import tests.experiments.run_files as run_files


def _reference_layout():
    """A weak pool escalating to one on the host, on the reference card."""
    switching = run_files.switching_machine()
    reference = fabrics.ReferenceFabric()
    on_the_reference = dataclasses.replace(switching, fabric=reference)
    at_task = on_the_reference.at_task()
    return graph_layout.layout_of(at_task)


REFERENCE_LAYOUT = _reference_layout()


class RecordingController:
    """The landing: which decisions arrived, when, and over what path."""

    def __init__(self, engine):
        self.engine = engine
        self.relayed = []

    def relay_instruction(self, decision, deliver):
        self.relayed.append((self.engine.now, decision, deliver))


def dispatch_over(link):
    """The unit on one fabric, with a recording controller behind it.

    A card that prices no path gives no fabric, and the link port is
    then left unbound.
    """
    engine = engine_module.Engine()
    controller = RecordingController(engine)
    unit = decision_dispatch.DecisionDispatch(engine)
    unit.instruction_output = controller
    if link is not None:
        unit.link = link
    return engine, unit, controller


def priced_fabric(engine):
    reference = REFERENCE_LAYOUT.card
    return fabric_module.LinkFabric(reference, engine)


def test_the_decision_reaches_the_controller_one_crossing_later():
    engine = engine_module.Engine()
    link = priced_fabric(engine)
    controller = RecordingController(engine)
    unit = decision_dispatch.DecisionDispatch(engine)
    unit.instruction_output = controller
    unit.link = link
    crossing_ticks = link.expected_delay_ticks(
        transfer_records.FRAME_TO_CONTROLLER, None, 0
    )
    release = program_records.Decision(2, releases_operation=True)
    deliver = object()

    unit.dispatch_decision(release, deliver)
    engine.run()

    assert controller.relayed == [(crossing_ticks, release, deliver)]
    assert crossing_ticks > 0


def test_the_crossing_is_billed_to_the_frame_to_controller_path():
    """The send is one transfer on the hop the decision rides."""
    engine = engine_module.Engine()
    link = priced_fabric(engine)
    controller = RecordingController(engine)
    unit = decision_dispatch.DecisionDispatch(engine)
    unit.instruction_output = controller
    unit.link = link
    delivered = []
    link.trace.transfer_delivered.connect(delivered.append)
    decision = program_records.Decision(7, releases_operation=False)
    deliver = object()

    unit.dispatch_decision(decision, deliver)
    engine.run()

    (record,) = delivered
    assert record.path == transfer_records.FRAME_TO_CONTROLLER
    assert record.attribution.owner_id == 7


def test_with_no_fabric_the_decision_is_at_the_controller_at_once():
    """An unpriced card drops the crossing, not the decision."""
    engine, unit, controller = dispatch_over(None)
    decision = program_records.Decision(9, releases_operation=False)
    deliver = object()

    unit.dispatch_decision(decision, deliver)
    engine.run()

    ticks = [tick for tick, _decision, _deliver in controller.relayed]
    assert ticks == [0]
