"""Physical controller-boundary controls.

The model does not synthesize ADC or DAC waveforms. It must nevertheless
preserve the real data crossing each declared boundary and charge each
online stage in causal order.
"""

from decsim.config import TimingConfig, us
from decsim.controller.controller import Controller
from decsim.controller.syndrome_packing import SyndromePacking
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.engine import Engine
from decsim.message import Decision, QPUReadout, RunOperationBody, WINDOW_INPUT_ROUTE
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec


class _WindowInputReceiver:
    def __init__(self):
        self.packets = []

    def accept_window_input(self, packet):
        self.packets.append(packet)
        return True


def test_measurement_signal_path_preserves_classified_bits_and_exact_latency(fabric):
    """QC transport plus the analog-readout/classification abstraction
    produces the same bit values, neither early nor altered."""
    engine = Engine(verbose=False)
    receiver = _WindowInputReceiver()
    links = fabric["declared_profile"](cwb=False, csb=False).resolve()
    packing = SyndromePacking(
        engine, links=links, t_pack=0, packing_context_capacity=None,
        window_input_receiver=receiver, feedback_memory_receiver=None)
    controller = Controller(
        engine, qpu=None, window_manager=None, syndrome_packing=packing,
        measurement_signal_to_classical_bits_ticks=us(3), links=links,
        resolved_operations=(), resolved_patches=(), idle_policy=None,
        feedback_streams=None)

    controller.accept_qpu_readout(
        QPUReadout(7, "patch-a", 4, bits=[True, False, 1, 0], size_bits=4),
        WINDOW_INPUT_ROUTE)
    engine.run()

    assert [(event.kind, event.tick) for event in packing.round_events[:3]] == [
        ("EMITTED", 0),
        ("BINARY_AVAILABLE", us(2 + 3)),
        ("PACKED", us(2 + 3)),
    ]
    assert len(receiver.packets) == 1
    assert receiver.packets[0].fragments[0].bits == (1, 0, 1, 0)
    assert receiver.packets[0].fragments[0].size_bits == 4
    qc_transfer = next(transfer for transfer in links.snapshot().transfers
                       if transfer.path.value == "qc")
    assert qc_transfer.reservation.payload_bits == 4


def _feedback_run(fabric, *, controller_output_us):
    operations = (
        fabric["memory_op"](1),
        fabric["memory_op"](2, blocked_by=1),
    )
    timing = TimingConfig(
        round_us=1.0,
        measurement_signal_to_classical_bits_us=fabric["DECLARED_US"]["binary"],
        instruction_or_decision_to_analog_control_pulse_us=controller_output_us,
        t_pack_us=fabric["DECLARED_US"]["pack"],
    )
    return RunSpec(
        ops=operations, d=3, rounds_policy=FixedRounds(6),
        decoder=PresetLatencyDecoder(fabric["DECLARED_US"]["weak"]),
        links=fabric["declared_profile"](cwb=True, csb=False),
        timing=timing,
        pauli_frame=PauliFrameConfig(commit_us=fabric["DECLARED_US"]["frame"]),
    ).build()


def test_feedback_operation_command_traverses_controller_output_and_cq(fabric):
    """The returned decision reaches the controller over OC; the actual
    blocked operation command then pays output processing and CQ before the
    QPU receives exactly that command."""
    completed = _feedback_run(fabric, controller_output_us=3.0)
    completed.engine.run()
    runtime = completed.execution_runtime
    blocker_commit = next(
        record.committed_ticks for record in completed.pauli_frame.snapshot().records
        if record.window_key == (1, 0))

    assert runtime.decode_release_time[2] == blocker_commit + us(2)  # OC
    assert runtime.op_start_time[2] == blocker_commit + us(2 + 3 + 2)

    arrivals = [event for event in completed.qpu.command_events
                if event.kind == "ARRIVED" and event.command.operation.id == 2]
    assert len(arrivals) == 1
    assert arrivals[0].tick == blocker_commit + us(2 + 3 + 2)
    assert isinstance(arrivals[0].command, RunOperationBody)
    assert arrivals[0].command.operation == runtime.operations[2]

    output = [event for event in completed.controller.output_events
              if event.operation_id == 2]
    assert [(event.kind, event.tick) for event in output] == [
        ("DECISION_AVAILABLE", blocker_commit + us(2)),
        ("CONTROL_PULSE_COMMAND_ISSUED", blocker_commit + us(2 + 3)),
    ]
    assert output[0].payload.target_operation_id == 2
    assert output[1].payload is arrivals[0].command


def test_output_latency_changes_arrival_but_not_decision_availability(fabric):
    zero = _feedback_run(fabric, controller_output_us=0.0)
    delayed = _feedback_run(fabric, controller_output_us=3.0)
    zero.engine.run()
    delayed.engine.run()

    assert (zero.execution_runtime.decode_release_time[2]
            == delayed.execution_runtime.decode_release_time[2])
    assert (delayed.execution_runtime.op_start_time[2]
            - zero.execution_runtime.op_start_time[2]) == us(3)


def test_non_aligned_controller_arrival_waits_for_next_qec_boundary(fabric):
    completed = _feedback_run(fabric, controller_output_us=0.016)
    completed.engine.run()
    arrival = next(event.tick for event in completed.qpu.command_events
                   if event.kind == "ARRIVED" and event.command.operation.id == 2)
    start = completed.execution_runtime.op_start_time[2]

    assert arrival % us(1.0) == us(0.016)
    assert start == ((arrival // us(1.0)) + 1) * us(1.0)


def test_preloaded_program_command_is_not_charged_as_online_feedback(fabric):
    """A command staged before t=0 still uses the controller/QPU command
    type, but a dynamic controller delay must not move the root operation."""
    completed = _feedback_run(fabric, controller_output_us=3.0)
    completed.engine.run()

    root_arrival = next(event for event in completed.qpu.command_events
                        if event.kind == "ARRIVED" and event.command.operation.id == 1)
    assert root_arrival.tick == 0
    assert completed.execution_runtime.op_start_time[1] == 0
    assert any(event.kind == "PRELOADED_COMMAND" and event.operation_id == 1
               for event in completed.controller.output_events)


def test_result_return_carries_the_same_decision_through_output_and_cq(fabric):
    operation = fabric["memory_op"](1, requires_result_return=True)
    timing = TimingConfig(
        round_us=1.0,
        measurement_signal_to_classical_bits_us=fabric["DECLARED_US"]["binary"],
        t_pack_us=fabric["DECLARED_US"]["pack"],
        instruction_or_decision_to_analog_control_pulse_us=3.0)
    completed = RunSpec(
        ops=(operation,), d=3, rounds_policy=FixedRounds(6),
        decoder=PresetLatencyDecoder(fabric["DECLARED_US"]["weak"]),
        links=fabric["declared_profile"](cwb=True, csb=False), timing=timing,
        pauli_frame=PauliFrameConfig(commit_us=fabric["DECLARED_US"]["frame"]),
    ).build()
    completed.engine.run()
    commit = next(record.committed_ticks
                  for record in completed.pauli_frame.snapshot().records)

    output = [event for event in completed.controller.output_events
              if event.operation_id == 1 and event.kind != "PRELOADED_COMMAND"]
    assert [(event.kind, event.tick) for event in output] == [
        ("DECISION_AVAILABLE", commit + us(2)),
        ("CONTROL_DECISION_ISSUED", commit + us(2 + 3)),
    ]
    assert output[1].payload is output[0].payload
    assert completed.execution_runtime.result_return_time_by_operation[1] == \
        commit + us(2 + 3 + 2)


def test_controller_output_without_a_link_still_pays_local_processing():
    engine = Engine(verbose=False)
    delivered = []
    controller = Controller(
        engine, qpu=None, window_manager=None, syndrome_packing=None,
        instruction_or_decision_to_analog_control_pulse_ticks=17,
        links=None, resolved_operations=(), resolved_patches=(),
        idle_policy=None, feedback_streams=None)
    decision = Decision(9, releases_operation=False)

    controller.relay_instruction(decision, delivered.append)
    engine.run()

    assert delivered == [decision]
    assert engine.now == 17
    assert [(event.kind, event.tick, event.payload)
            for event in controller.output_events] == [
        ("DECISION_AVAILABLE", 0, decision),
        ("CONTROL_DECISION_ISSUED", 17, decision),
    ]


def test_qubic_500_mhz_eight_cycle_controller_fixture_is_parameter_driven():
    """QubiC's conservative jump_fproc setting is 8 clocks at 500 MHz:
    16 ns. The reference number is a test input, not production logic."""
    clock_hz = 500_000_000
    cycles = 8
    controller_output_us = cycles / clock_hz * 1_000_000
    timing = TimingConfig(
        instruction_or_decision_to_analog_control_pulse_us=controller_output_us)

    assert controller_output_us == 0.016
    assert timing.ticks("instruction_or_decision_to_analog_control_pulse") == us(0.016)


def test_qubicml_500_mhz_27_cycle_discriminator_fixture_is_parameter_driven():
    """QubiCML reports 27 FPGA clocks from normalized data through its
    inference result: 54 ns at 500 MHz. The separate 500 ns acquisition
    window is already part of the QPU measurement, not silently added here."""
    clock_hz = 500_000_000
    inference_cycles = 27
    classification_us = inference_cycles / clock_hz * 1_000_000
    timing = TimingConfig(
        measurement_signal_to_classical_bits_us=classification_us)

    assert classification_us == 0.054
    assert timing.ticks("measurement_signal_to_classical_bits") == us(0.054)
