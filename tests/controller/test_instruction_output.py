"""The output: decisions and commands reach the QPU after the pulse cost.

A release is consumed at the controller as soon as it crosses
frame_to_controller; a result return without an operation pays the
decision-to-pulse cost (17 ticks here; QubiC's 8 clocks at 500 MHz are
16 ns, 2110.00557) and the controller_to_qpu crossing before it is
available at the QPU. The link law is the channel's
(tests/links/test_channel.py); here the reference card prices both hops.
"""

import decsim.controller.instruction_output as instruction_output
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.message as message
import decsim.observe.round_events as round_events

PULSE_TICKS = 17


def test_a_release_is_delivered_when_it_reaches_the_controller():
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    link = fabric_module.LinkFabric(reference, engine)
    recorder = round_events.RoundEventRecorder(engine)
    output = instruction_output.InstructionOutput(
        engine, link, None, PULSE_TICKS
    )
    output.output_event.connect(recorder.output)
    crossing_ticks = link.expected_delay_ticks(
        message.LinkPath.FRAME_TO_CONTROLLER, None, 0
    )
    release = message.Decision(2, releases_operation=True)
    delivered = []

    def deliver(decision):
        delivered.append((engine.now, decision))

    output.relay_instruction(release, deliver)
    engine.run()

    assert delivered == [(crossing_ticks, release)]
    kinds_and_ticks = [
        (event.kind, event.tick) for event in recorder.output_events
    ]
    assert kinds_and_ticks == [("DECISION_AVAILABLE", crossing_ticks)]


def test_a_result_return_pays_the_pulse_cost_and_the_crossing_to_the_qpu():
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    link = fabric_module.LinkFabric(reference, engine)
    recorder = round_events.RoundEventRecorder(engine)
    output = instruction_output.InstructionOutput(
        engine, link, None, PULSE_TICKS
    )
    output.output_event.connect(recorder.output)
    to_controller = link.expected_delay_ticks(
        message.LinkPath.FRAME_TO_CONTROLLER, None, 0
    )
    to_qpu = link.expected_delay_ticks(
        message.LinkPath.CONTROLLER_TO_QPU, None, 0
    )
    result = message.Decision(2, releases_operation=False)
    delivered = []

    def deliver(decision):
        delivered.append((engine.now, decision))

    output.relay_instruction(result, deliver)
    engine.run()

    issued_tick = to_controller + PULSE_TICKS
    assert delivered == [(issued_tick + to_qpu, result)]
    kinds_and_ticks = [
        (event.kind, event.tick) for event in recorder.output_events
    ]
    assert kinds_and_ticks == [
        ("DECISION_AVAILABLE", to_controller),
        ("CONTROL_DECISION_ISSUED", issued_tick),
    ]
