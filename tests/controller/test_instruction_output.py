"""The output: decisions and commands reach the QPU after the pulse cost.

A release is consumed at the controller the instant it lands from
frame_to_controller, whose crossing the frame side executes and prices
(tests/pauli_frame/test_decision_dispatch.py); a result return without
an operation pays the decision-to-pulse cost (17 ticks here; QubiC's 8
clocks at 500 MHz are 16 ns, 2110.00557) and the controller_to_qpu
crossing before it is available at the QPU. The link law is the
channel's (tests/links/test_channel.py); here the reference card prices
the output hop.

The pulse cost is the control processor's own work and stands even with
no fabric at all (QubiC holds the conditional jump and the pulse on the
core beside the qubit, Fruitwala et al. 2404.15260). The whole-run law
at the end of the file composes the three stages the declared card of
tests/declared_run.py prices, the closed loop Yang et al. 2605.04892
measure end to end as the sum of its stages (550 ns from the end of the
readout pulse to the start of the feedback pulse).
"""

import decsim.config as config
import decsim.controller.instruction_output as instruction_output
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.observe.round_events as round_events
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import tests.declared_run as declared_run

PULSE_TICKS = 17


def test_a_release_is_consumed_where_it_lands():
    """The crossing is already paid: the release is available at once."""
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    link = fabric_module.LinkFabric(reference, engine)
    recorder = round_events.RoundEventRecorder(engine)
    output = instruction_output.InstructionOutput(
        engine, link, None, PULSE_TICKS
    )
    output.trace.output_event.connect(recorder.output)
    release = program_records.Decision(2, releases_operation=True)
    delivered = []

    def deliver(decision):
        delivered.append((engine.now, decision))

    output.relay_instruction(release, deliver)
    engine.run()

    assert delivered == [(0, release)]
    kinds_and_ticks = [
        (event.kind, event.tick) for event in recorder.output_events
    ]
    assert kinds_and_ticks == [("DECISION_AVAILABLE", 0)]


def test_a_result_return_pays_the_pulse_cost_and_the_crossing_to_the_qpu():
    engine = engine_module.Engine()
    reference = link_profiles.logical_reference_profile()
    link = fabric_module.LinkFabric(reference, engine)
    recorder = round_events.RoundEventRecorder(engine)
    output = instruction_output.InstructionOutput(
        engine, link, None, PULSE_TICKS
    )
    output.trace.output_event.connect(recorder.output)
    to_qpu = link.expected_delay_ticks(
        transfer_records.LinkPath.CONTROLLER_TO_QPU, None, 0
    )
    result = program_records.Decision(2, releases_operation=False)
    delivered = []

    def deliver(decision):
        delivered.append((engine.now, decision))

    output.relay_instruction(result, deliver)
    engine.run()

    assert delivered == [(PULSE_TICKS + to_qpu, result)]
    kinds_and_ticks = [
        (event.kind, event.tick) for event in recorder.output_events
    ]
    assert kinds_and_ticks == [
        ("DECISION_AVAILABLE", 0),
        ("CONTROL_DECISION_ISSUED", PULSE_TICKS),
    ]


def test_a_result_return_with_no_link_still_pays_the_pulse_cost():
    """An unpriced fabric drops the crossing, not the local work.

    The decision-to-pulse cost is the control processor's own work
    between the decision and the pulse it triggers, which QubiC runs on
    the core beside the qubit (Fruitwala et al. 2404.15260); a card
    that prices no path therefore still charges it, and the result is
    available at the QPU one pulse cost after the decision.
    """
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    output = instruction_output.InstructionOutput(
        engine, None, None, PULSE_TICKS
    )
    output.trace.output_event.connect(recorder.output)
    result = program_records.Decision(9, releases_operation=False)
    delivered = []

    def deliver(decision):
        delivered.append((engine.now, decision))

    output.relay_instruction(result, deliver)
    engine.run()

    assert delivered == [(PULSE_TICKS, result)]
    kinds_and_ticks = [
        (event.kind, event.tick) for event in recorder.output_events
    ]
    assert kinds_and_ticks == [
        ("DECISION_AVAILABLE", 0),
        ("CONTROL_DECISION_ISSUED", PULSE_TICKS),
    ]


# The whole loop: the frame's commit, the release, and the command the
# release starts, on the declared card of tests/declared_run.py.


def commit_tick_of(machine, window_key):
    """The tick the frame committed one window's correction."""
    snapshot = machine.pauli_frame.snapshot()
    for record in snapshot.records:
        if record.window_key == window_key:
            return record.committed_ticks
    raise AssertionError(f"no frame record for window {window_key}")


def command_arrivals_for(machine, operation_id):
    """Every command that reached the QPU for one operation."""
    arrivals = []
    for event in machine.observation.command_events.events:
        if event.kind != "ARRIVED":
            continue
        if event.command.operation.id != operation_id:
            continue
        arrivals.append(event)
    return arrivals


def test_a_result_return_is_reported_to_the_execution_runtime():
    """The modelled destination of a result return is the runtime.

    The return pays frame_to_controller, the pulse cost and
    controller_to_qpu, and the object that takes the outcome is the
    execution runtime, which stands for the classical program: Caune et
    al. 2410.05202 stalls that program on the decoder's status register
    and goes on (lines 364-367), and QubiC executes the branch on the
    control processor beside the qubit (Fruitwala et al. 2404.15260
    Sec. III and IV). The QPU device runs no command for it, so the only
    command that reached it is the operation's own start.
    """
    operation = declared_run.memory_operation(
        1, requires_result_return_to_qpu=True
    )
    machine = declared_run.weak_only_run(rounds=6, operations=(operation,))
    commit = commit_tick_of(machine, (1, 0))
    to_controller = config.microseconds_to_ticks(2.0)
    to_qpu = config.microseconds_to_ticks(2.0)
    stamps = machine.observation.runtime_stamps
    arrivals = command_arrivals_for(machine, 1)

    assert stamps.result_return == {1: commit + to_controller + to_qpu}
    assert [arrival.tick for arrival in arrivals] == [0]


def test_the_feedback_chain_is_the_frame_commit_plus_each_stage_once():
    """A blocked successor pays frame_to_controller, the pulse, then cq.

    The closed loop is the sum of its stages and nothing else (Yang et
    al. 2605.04892 measure 550 ns from the end of the readout pulse to
    the start of the feedback pulse by adding up the stages of their
    feedback module), so the release is at the blocker's commit plus
    frame_to_controller 2, and the successor's own command reaches the
    QPU one decision-to-pulse cost of 3 and one controller_to_qpu 2
    later. The command is the sequencer's operation, not a copy of it:
    the QPU runs what the workload declared.
    """
    first = declared_run.memory_operation(1)
    successor = declared_run.memory_operation(2, blocked_by=1)
    operations = (first, successor)
    controller = declared_run.declared_controller(
        decision_to_pulse_microseconds=3.0
    )
    machine = declared_run.weak_only_run(
        rounds=6, operations=operations, controller=controller
    )
    blocker_commit = commit_tick_of(machine, (1, 0))
    to_controller = config.microseconds_to_ticks(2.0)
    pulse_ticks = config.microseconds_to_ticks(3.0)
    to_qpu = config.microseconds_to_ticks(2.0)
    expected_release = blocker_commit + to_controller
    output_path_ticks = pulse_ticks + to_qpu
    expected_start = expected_release + output_path_ticks
    stamps = machine.observation.runtime_stamps
    sequencer = machine.execution_runtime
    (arrival,) = command_arrivals_for(machine, 2)

    assert blocker_commit == config.microseconds_to_ticks(33.0)
    assert stamps.decode_release[2] == expected_release
    assert stamps.op_start[2] == expected_start
    assert arrival.tick == expected_start
    assert arrival.command.operation is sequencer.schedule.operations[2]
