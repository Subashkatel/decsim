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

import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.observe.log_writers as log_writers
import decsim.pauli_frame.decision_dispatch as decision_dispatch
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records


class RecordingController:
    """The landing: which decisions arrived, when, and over what path."""

    def __init__(self, engine):
        self.engine = engine
        self.relayed = []

    def relay_instruction(self, decision, deliver):
        self.relayed.append((self.engine.now, decision, deliver))


def dispatch_over(link):
    """The unit on one fabric, with a recording controller behind it."""
    engine = engine_module.Engine()
    controller = RecordingController(engine)
    unit = decision_dispatch.DecisionDispatch(engine, link, controller)
    return engine, unit, controller


def priced_fabric(engine):
    reference = link_profiles.logical_reference_profile()
    return fabric_module.LinkFabric(reference, engine)


def test_the_decision_reaches_the_controller_one_crossing_later():
    engine = engine_module.Engine()
    link = priced_fabric(engine)
    controller = RecordingController(engine)
    unit = decision_dispatch.DecisionDispatch(engine, link, controller)
    crossing_ticks = link.expected_delay_ticks(
        transfer_records.LinkPath.FRAME_TO_CONTROLLER, None, 0
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
    unit = decision_dispatch.DecisionDispatch(engine, link, controller)
    delivered = []
    link.trace.transfer_delivered.connect(delivered.append)
    decision = program_records.Decision(7, releases_operation=False)
    deliver = object()

    unit.dispatch_decision(decision, deliver)
    engine.run()

    (record,) = delivered
    assert record.path == transfer_records.LinkPath.FRAME_TO_CONTROLLER
    assert record.attribution.operation_id == 7


def test_with_no_fabric_the_decision_is_at_the_controller_at_once():
    """An unpriced card drops the crossing, not the decision."""
    engine, unit, controller = dispatch_over(None)
    decision = program_records.Decision(9, releases_operation=False)
    deliver = object()

    unit.dispatch_decision(decision, deliver)
    engine.run()

    ticks = [tick for tick, _decision, _deliver in controller.relayed]
    assert ticks == [0]


def test_a_release_is_narrated_at_the_end_it_leaves_by():
    engine, unit, _controller = dispatch_over(None)
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    release = program_records.Decision(12, releases_operation=True)
    deliver = object()

    unit.dispatch_decision(release, deliver)

    assert log.lines == [
        "[  0.000 us] PauliFrame: DISPATCH conditional release for op#12 "
        "-> controller -> controller sequencer"
    ]


def test_a_result_return_is_narrated_as_a_return():
    engine, unit, _controller = dispatch_over(None)
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    decision = program_records.Decision(9, releases_operation=False)
    deliver = object()

    unit.dispatch_decision(decision, deliver)

    assert log.lines == [
        "[  0.000 us] PauliFrame: DISPATCH result return for op#9 "
        "-> controller -> controller sequencer"
    ]
