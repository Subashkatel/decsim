"""Physical controller-boundary controls.

The model does not synthesize ADC or DAC waveforms. It must nevertheless
preserve the real data crossing each declared boundary and charge each
online stage in causal order.
"""

import decsim.records.program as program_records
from decsim.config import microseconds_to_ticks
from decsim.controller.instruction_output import InstructionOutput
from decsim.controller.settings import ControllerSettings
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.settings import DecoderSettings
from decsim.engine import Engine
from decsim.frontends.settings import WorkloadSettings
from decsim.links.fabric import LinkFabric
from decsim.machine import MachineSettings
from decsim.observe.link_traffic import TrafficLedger
from decsim.observe.round_events import RoundEventRecorder
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.settings import QpuSettings


def _feedback_run(fabric, *, controller_output_us):
    operations = (
        fabric["memory_op"](1),
        fabric["memory_op"](2, blocked_by=1),
    )
    controller = ControllerSettings(
        readout_to_bits_microseconds=fabric["DECLARED_US"]["binary"],
        decision_to_pulse_microseconds=controller_output_us,
        packing_microseconds_per_round=fabric["DECLARED_US"]["pack"],
    )
    settings = MachineSettings(
        workload=WorkloadSettings(
            operations=operations, rounds_policy=FixedRounds(6)
        ),
        qpu=QpuSettings(distance=3, round_period_microseconds=1.0),
        weak_decoder=DecoderSettings(
            decoder=PresetLatencyDecoder(fabric["DECLARED_US"]["weak"])
        ),
        links=fabric["declared_profile"](
            controller_to_weak_buffer=True, controller_to_strong_buffer=False
        ),
        controller=controller,
        pauli_frame=PauliFrameConfig(
            commit_microseconds=fabric["DECLARED_US"]["frame"]
        ),
    )
    return fabric["run_machine"](settings)


def test_results_and_decisions_cross_links_at_their_own_size(fabric):
    """A decoder result reaches the frame as one bit per logical
    observable (Caune et al. 2410.05202 return one Boolean per decode;
    Google 2408.13687 an observable bitmask; PECOS XORs an observable mask
    into its frame); a decision crosses oc as one 32-bit bus word (the
    decoder sequencer's WISHBONE interface, Caune et al. 2410.05202) and a
    command crosses cq as one 128-bit instruction word (QubiC's distributed
    processor, Fruitwala et al. 2404.15260). The reference card carries no
    system-wide aggregate on these paths."""
    completed = _feedback_run(fabric, controller_output_us=0.0)
    transfers = completed.observation.traffic.traffic_json_value()["transfers"]
    payload_by_path = {}
    for transfer in transfers:
        payload_by_path.setdefault(transfer["path"], set()).add(
            transfer["payload_bits"]
        )
    assert payload_by_path["weak_decoder_to_frame"] == {1}
    assert payload_by_path["frame_to_controller"] == {32}
    assert payload_by_path["controller_to_qpu"] == {128}


def test_feedback_operation_command_traverses_controller_output_and_cq(fabric):
    """The returned decision reaches the controller over OC; the actual
    blocked operation command then pays output processing and CQ before the
    QPU receives exactly that command."""
    completed = _feedback_run(fabric, controller_output_us=3.0)
    completed.engine.run()
    runtime = completed.execution_runtime
    blocker_commit = next(
        record.committed_ticks
        for record in completed.pauli_frame.snapshot().records
        if record.window_key == (1, 0)
    )

    stamps = completed.observation.runtime_stamps
    assert stamps.decode_release[2] == blocker_commit + microseconds_to_ticks(
        2
    )  # OC
    assert stamps.op_start[2] == blocker_commit + microseconds_to_ticks(
        2 + 3 + 2
    )

    arrivals = [
        event
        for event in completed.observation.command_events.events
        if event.kind == "ARRIVED" and event.command.operation.id == 2
    ]
    assert len(arrivals) == 1
    assert arrivals[0].tick == blocker_commit + microseconds_to_ticks(2 + 3 + 2)
    assert isinstance(arrivals[0].command, program_records.RunOperationBody)
    assert arrivals[0].command.operation == runtime.operations[2]

    output = [
        event
        for event in completed.observation.round_events.output_events
        if event.operation_id == 2
    ]
    assert [(event.kind, event.tick) for event in output] == [
        ("DECISION_AVAILABLE", blocker_commit + microseconds_to_ticks(2)),
        (
            "CONTROL_PULSE_COMMAND_ISSUED",
            blocker_commit + microseconds_to_ticks(2 + 3),
        ),
    ]
    assert output[0].payload.target_operation_id == 2
    assert output[1].payload is arrivals[0].command


def test_output_latency_changes_arrival_but_not_decision_availability(fabric):
    zero = _feedback_run(fabric, controller_output_us=0.0)
    delayed = _feedback_run(fabric, controller_output_us=3.0)
    zero.engine.run()
    delayed.engine.run()

    assert (
        zero.observation.runtime_stamps.decode_release[2]
        == delayed.observation.runtime_stamps.decode_release[2]
    )
    assert (
        delayed.observation.runtime_stamps.op_start[2]
        - zero.observation.runtime_stamps.op_start[2]
    ) == microseconds_to_ticks(3)


def test_non_aligned_controller_arrival_waits_for_next_qec_boundary(fabric):
    completed = _feedback_run(fabric, controller_output_us=0.016)
    completed.engine.run()
    arrival = next(
        event.tick
        for event in completed.observation.command_events.events
        if event.kind == "ARRIVED" and event.command.operation.id == 2
    )
    start = completed.observation.runtime_stamps.op_start[2]

    assert arrival % microseconds_to_ticks(1.0) == microseconds_to_ticks(0.016)
    assert start == (
        (arrival // microseconds_to_ticks(1.0)) + 1
    ) * microseconds_to_ticks(1.0)


def test_preloaded_program_command_is_not_charged_as_online_feedback(fabric):
    """A command staged before t=0 still uses the controller/QPU command
    type, but a dynamic controller delay must not move the root operation."""
    completed = _feedback_run(fabric, controller_output_us=3.0)
    completed.engine.run()

    root_arrival = next(
        event
        for event in completed.observation.command_events.events
        if event.kind == "ARRIVED" and event.command.operation.id == 1
    )
    assert root_arrival.tick == 0
    assert completed.observation.runtime_stamps.op_start[1] == 0
    assert any(
        event.kind == "PRELOADED_COMMAND" and event.operation_id == 1
        for event in completed.observation.round_events.output_events
    )


def test_result_return_carries_the_same_decision_through_output_and_cq(fabric):
    operation = fabric["memory_op"](1, requires_result_return=True)
    controller = ControllerSettings(
        readout_to_bits_microseconds=fabric["DECLARED_US"]["binary"],
        packing_microseconds_per_round=fabric["DECLARED_US"]["pack"],
        decision_to_pulse_microseconds=3.0,
    )
    settings = MachineSettings(
        workload=WorkloadSettings(
            operations=(operation,), rounds_policy=FixedRounds(6)
        ),
        qpu=QpuSettings(distance=3, round_period_microseconds=1.0),
        weak_decoder=DecoderSettings(
            decoder=PresetLatencyDecoder(fabric["DECLARED_US"]["weak"])
        ),
        links=fabric["declared_profile"](
            controller_to_weak_buffer=True, controller_to_strong_buffer=False
        ),
        controller=controller,
        pauli_frame=PauliFrameConfig(
            commit_microseconds=fabric["DECLARED_US"]["frame"]
        ),
    )
    completed = fabric["run_machine"](settings)
    completed.engine.run()
    commit = next(
        record.committed_ticks
        for record in completed.pauli_frame.snapshot().records
    )

    output = [
        event
        for event in completed.observation.round_events.output_events
        if event.operation_id == 1 and event.kind != "PRELOADED_COMMAND"
    ]
    assert [(event.kind, event.tick) for event in output] == [
        ("DECISION_AVAILABLE", commit + microseconds_to_ticks(2)),
        ("CONTROL_DECISION_ISSUED", commit + microseconds_to_ticks(2 + 3)),
    ]
    assert output[1].payload is output[0].payload
    assert completed.observation.runtime_stamps.result_return[
        1
    ] == commit + microseconds_to_ticks(2 + 3 + 2)


def test_controller_output_without_a_link_still_pays_local_processing():
    engine = Engine()
    delivered = []
    recorder = RoundEventRecorder(engine)
    output = InstructionOutput(engine, None, None, 17)
    output.output_event.connect(recorder.output)
    decision = program_records.Decision(9, releases_operation=False)

    output.relay_instruction(decision, delivered.append)
    engine.run()

    assert delivered == [decision]
    assert engine.now == 17
    assert [
        (event.kind, event.tick, event.payload)
        for event in recorder.output_events
    ] == [
        ("DECISION_AVAILABLE", 0, decision),
        ("CONTROL_DECISION_ISSUED", 17, decision),
    ]


def test_qubic_500_mhz_eight_cycle_controller_fixture_is_parameter_driven():
    """QubiC's conservative jump_fproc setting is 8 clocks at 500 MHz:
    16 ns. The reference number is a test input, not production logic."""
    clock_hz = 500_000_000
    cycles = 8
    controller_output_us = cycles / clock_hz * 1_000_000
    timing = ControllerSettings(
        decision_to_pulse_microseconds=controller_output_us
    )

    assert controller_output_us == 0.016
    assert timing.decision_to_pulse_ticks() == microseconds_to_ticks(0.016)


def test_qubicml_500_mhz_27_cycle_discriminator_fixture_is_parameter_driven():
    """QubiCML reports 27 FPGA clocks from normalized data through its
    inference result: 54 ns at 500 MHz. The separate 500 ns acquisition
    window is already part of the QPU measurement, not silently added here."""
    clock_hz = 500_000_000
    inference_cycles = 27
    classification_us = inference_cycles / clock_hz * 1_000_000
    timing = ControllerSettings(readout_to_bits_microseconds=classification_us)

    assert classification_us == 0.054
    assert timing.readout_to_bits_ticks() == microseconds_to_ticks(0.054)
