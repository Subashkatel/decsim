"""Live Stim records agree with Stim's complete-circuit converter.

Stim's generated surface memory supplies first, repeated and terminal
instructions. TableauSimulator.do preserves the physical state between them.
"""

import dataclasses

import numpy
import pytest
import stim

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.records.circuits as circuit_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import tests.declared_run as declared_run
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


@pytest.mark.parametrize("round_count", [1, 2, 7])
def test_property_live_records_match_stim_at_different_stop_lengths(
    round_count: int,
) -> None:
    source, owner = _source()
    events = _execute(source, owner, round_count)
    circuit = source.executed_circuit(owner.id)
    measurements = source.sampled_measurements(owner.id)
    raw = numpy.array([measurements], dtype=numpy.bool_)
    converter = circuit.compile_m2d_converter()
    expected_events, expected_truth = converter.convert(
        measurements=raw, separate_observables=True
    )
    numpy.testing.assert_array_equal(events, expected_events[0])
    truth = source.logical_observable_truth(owner.id)
    numpy.testing.assert_array_equal(truth, expected_truth[0])
    assert source.sampled_detection_events(owner.id) == tuple(events)
    source.validate_stream_length(owner, round_count)


def test_a_live_table_reaches_as_far_back_as_its_final_fragment_reads():
    """Two rounds in, the next may be the final one, three rounds back.

    After the readout no round follows.
    """
    program = _final_reaching_program()
    source, owner = _source(program=program)
    _run_round(source, owner, 1, is_final=False)
    _run_round(source, owner, 2, is_final=False)
    running = source.formation_table(owner.id)
    _run_round(source, owner, 3, is_final=True)
    read_out = source.formation_table(owner.id)

    assert running.live_reach == 3
    assert read_out.live_reach is None


def test_a_program_whose_round_circuit_keeps_the_fragments_is_declared():
    """A surface-code round reads the one before it: a reach of one."""
    declared = memory_programs.memory_program()
    program = _DeclaredFragments(
        declared.first_round,
        declared.repeated_round,
        declared.final_round,
        declared.single_round,
    )

    source, owner = _source(program=program)
    _run_round(source, owner, 1, is_final=False)

    running = source.formation_table(owner.id)
    assert running.live_reach == 1


def test_a_later_stop_keeps_the_executed_nonterminal_prefix() -> None:
    shorter, owner = _source()
    longer, _ = _source()
    _execute(shorter, owner, 7)
    _execute(longer, owner, 13)
    short_record = shorter.sampled_measurements(owner.id)
    long_record = longer.sampled_measurements(owner.id)
    prefix_measurement_count = 6 * 8
    assert (
        short_record[:prefix_measurement_count]
        == long_record[:prefix_measurement_count]
    )


def test_window_lookahead_does_not_execute_physical_measurements() -> None:
    source, owner = _source()
    window = window_records.Window(owner.id, 0, 1, 3, 6, 6)
    model = source.window_model_for_stream(owner.id, window)
    assert model.detector_ids
    assert source.sampled_measurements(owner.id) == ()
    assert source.logical_observable_truth(owner.id) is None
    executed = source.executed_circuit(owner.id)
    empty = stim.Circuit()
    assert executed == empty


def test_a_finalized_stream_refuses_another_physical_round() -> None:
    source, owner = _source()
    _execute(source, owner, 1)
    with pytest.raises(RuntimeError, match="live Stim stream is already"):
        source.idle_round_payloads(
            owner,
            owner.id,
            2,
            is_final=False,
            round_period_ticks=1_100_000,
        )


def test_a_final_readout_without_observables_has_no_accuracy_truth() -> None:
    program = memory_programs.memory_program()
    removed_output = program.single_round.pop()
    assert removed_output.name == "OBSERVABLE_INCLUDE"
    source = streaming_stim_device.StreamingStimDevice(programs={1: program})
    owner = program_records.Operation(1, "memory", (0,), patches=(0,))
    requirement = fault_models.DecoderFaultModelRequirement(
        fault_models.GRAPHLIKE_ONLY
    )
    source.declare_stream(owner, 0)
    source.register_dynamic_stream(
        owner, 0, fault_model_requirement=requirement
    )
    source.idle_round_payloads(
        owner, 1, 1, is_final=True, round_period_ticks=1_100_000
    )
    assert source.logical_observable_truth(owner.id) is None


def test_idle_first_refuses_a_measurement_closed_owner() -> None:
    program = memory_programs.memory_program()
    source = streaming_stim_device.StreamingStimDevice(programs={1: program})
    owner = program_records.Operation(
        1,
        "memory",
        (0,),
        patches=(0,),
        feedback_boundary_mode="measurement_closed",
    )
    with pytest.raises(ValueError, match="live Stim memory requires"):
        source.declare_stream(owner, 0)
    assert source.logical_observable_truth(owner.id) is None


def test_a_segment_cannot_relabel_the_physical_patch() -> None:
    source, owner = _source()
    segment = dataclasses.replace(
        owner,
        id="prefix",
        stream_id=owner.id,
        stream_offset=0,
        patches=("other",),
    )
    with pytest.raises(ValueError, match="live Stim segment patches"):
        source.begin_operation(segment, 1, 0, round_period_ticks=1_100_000)
    assert source.sampled_measurements(owner.id) == ()


def test_a_segment_refuses_a_one_tick_period_mismatch() -> None:
    source, owner = _source(1.1)
    with pytest.raises(ValueError, match="physical circuit period differs"):
        source.begin_operation(owner, 1, 0, round_period_ticks=1_100_001)
    assert source.sampled_measurements(owner.id) == ()


def test_idle_first_refuses_a_period_mismatch_before_sampling() -> None:
    source, owner = _source(1.1)
    with pytest.raises(ValueError, match="physical circuit period differs"):
        source.idle_round_payloads(
            owner,
            owner.id,
            1,
            is_final=True,
            round_period_ticks=1_100_001,
        )
    assert source.sampled_measurements(owner.id) == ()


def test_a_subdisplay_precision_period_executes_at_its_exact_tick_count() -> (
    None
):
    source, owner = _source(1.100001)
    packets = source.idle_round_payloads(
        owner,
        owner.id,
        1,
        is_final=True,
        round_period_ticks=1_100_001,
    )
    assert packets[0].size_bits == 17
    assert source.logical_observable_truth(owner.id) is not None


def test_a_longer_feedback_wait_moves_the_actual_physical_readout() -> None:
    faster, first_source = _protected_machine(4.0)
    slower, second_source = _protected_machine(8.0)
    first_packets = []
    second_packets = []
    faster.qpu.device.trace.round_emitted.connect(first_packets.append)
    slower.qpu.device.trace.round_emitted.connect(second_packets.append)
    first_result = faster.run()
    second_result = slower.run()
    assert first_result.terminal_status == "complete"
    assert second_result.terminal_status == "complete"
    assert len(second_packets) > len(first_packets)
    assert (
        second_result.execution_done_ticks > first_result.execution_done_ticks
    )
    first_circuit = first_source.executed_circuit(100)
    second_circuit = second_source.executed_circuit(100)
    assert second_circuit.num_measurements > first_circuit.num_measurements
    assert first_packets[-1].size_bits == 17
    assert second_packets[-1].size_bits == 17
    assert first_result.operation_results[-1].observable_truth is not None
    assert second_result.operation_results[-1].observable_truth is not None
    assert first_result.decode_work_settled
    assert second_result.event_queue_empty


@pytest.mark.parametrize(
    "period_microseconds, noise_parameters",
    [
        (0.7, {"noise_model": "sd6"}),
        (
            1.25,
            {
                "noise_model": "physical",
                "relaxation_time_microseconds": 20.0,
                "dephasing_time_microseconds": 30.0,
            },
        ),
    ],
)
def test_deltakit_rounds_use_the_same_live_machine_and_record_oracle(
    period_microseconds: float, noise_parameters: dict
) -> None:
    pytest.importorskip("deltakit_explorer")
    program = deltakit.memory_rounds(
        "rotated_surface",
        3,
        "Z",
        0.003,
        round_period_microseconds=period_microseconds,
        **noise_parameters,
    )
    machine, source = _protected_machine(4.0, program, period_microseconds)
    packets = []
    machine.qpu.device.trace.round_emitted.connect(packets.append)
    result = machine.run()
    circuit = source.executed_circuit(100)
    measurements = source.sampled_measurements(100)
    raw = numpy.array([measurements], dtype=numpy.bool_)
    converter = circuit.compile_m2d_converter()
    events, truth = converter.convert(
        measurements=raw, separate_observables=True
    )
    actual_events = source.sampled_detection_events(100)
    actual_truth = source.logical_observable_truth(100)
    numpy.testing.assert_array_equal(actual_events, events[0])
    numpy.testing.assert_array_equal(actual_truth, truth[0])
    period_ticks = config.microseconds_to_ticks(period_microseconds)
    physical_duration_ticks = len(packets) * period_ticks
    assert result.execution_done_ticks == physical_duration_ticks
    assert result.terminal_status == "complete"
    assert result.decode_work_settled
    assert result.event_queue_empty


@pytest.mark.parametrize("round_count", [1, 3, 7])
def test_property_entangled_blocks_preserve_bell_parity_across_random_shots(
    round_count: int,
) -> None:
    """Shared-tableau Bell correlation, with a complete Stim record oracle."""
    program = memory_programs.joint_repetition_program(True, 0.0)
    owner = program_records.Operation(
        100, "joint", (0, 1), patches=("left", "right")
    )
    logical_values = set()
    for seed in range(16):
        source = streaming_stim_device.StreamingStimDevice({100: program}, seed)
        source.declare_stream(owner, 0)
        events = _execute(source, owner, round_count)
        record = source.sampled_measurements(100)
        assert record[-6] == record[-3]
        logical_values.add(record[-6])
        assert not any(events)
        assert source.logical_observable_truth(100) == (0,)
        _assert_complete_record_matches_stim(source, 100, events)
    assert logical_values == {0, 1}


def _source(round_period_microseconds=None, program=None):
    if program is None:
        program = memory_programs.memory_program()
    program = dataclasses.replace(
        program, round_period_microseconds=round_period_microseconds
    )
    source = streaming_stim_device.StreamingStimDevice(
        programs={"memory": program}, seed=37
    )
    owner = program_records.Operation(
        "memory", "memory", ("patch",), patches=("patch",)
    )
    requirement = fault_models.DecoderFaultModelRequirement(
        fault_models.GRAPHLIKE_ONLY
    )
    source.declare_stream(owner, 0)
    source.register_dynamic_stream(
        owner, 0, fault_model_requirement=requirement
    )
    return source, owner


def _run_round(source, owner, round_index: int, is_final: bool) -> None:
    """Run one round of the stream at the QPU's cadence."""
    source.idle_round_payloads(
        owner,
        owner.id,
        round_index,
        is_final=is_final,
        round_period_ticks=1_100_000,
    )


def _execute(source, owner, round_count):
    """Every round's events, formed at the controller as the stream grows."""
    at_the_controller = event_settings.DetectionEventSettings()
    placement = formation.SeatedFormation(source, at_the_controller)
    events = []
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        assert source.logical_observable_truth(owner.id) is None
        is_final = round_index == round_count
        packets = source.idle_round_payloads(
            owner,
            owner.id,
            round_index,
            is_final=is_final,
            round_period_ticks=1_100_000,
        )
        (formed,) = placement.form_at("controller", tuple(packets))
        events.extend(formed.bits)
    return events


def _protected_machine(
    feedback_microseconds, program=None, period_microseconds=1.1
):
    if program is None:
        program = memory_programs.memory_program()
    source = streaming_stim_device.StreamingStimDevice(programs={100: program})
    workload = _protected_workload()
    source_record = declared_run.GivenSource(source)
    qpu = qpu_settings.QpuSettings(
        distance=3,
        source=source_record,
        round_period_microseconds=period_microseconds,
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    links = link_profiles.logical_reference_profile()
    feedback_ticks = config.microseconds_to_ticks(feedback_microseconds)
    channel = dataclasses.replace(
        links.frame_to_controller.channel,
        propagation_latency_ticks=feedback_ticks,
    )
    path = dataclasses.replace(links.frame_to_controller, channel=channel)
    links = dataclasses.replace(links, frame_to_controller=path)
    observation = observation_settings.ObservationSettings(trace="chrome")
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=decoder,
        links=links,
        observation=observation,
    )
    machine = machine_module.Machine.build(settings, 17)
    return machine, source


def _protected_workload():
    """Keep the complete dependency graph together to expose its ownership."""
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    prefix = program_records.Operation(
        1, "prefix", (0,), patches=(0,), stream_id=100, stream_offset=0
    )
    begin = program_records.Operation(
        2,
        "protect",
        (0,),
        patches=(0,),
        predecessors=(1,),
        emits_detector_data=False,
    )
    resume = program_records.Operation(
        3,
        "resume",
        (0,),
        patches=(0,),
        predecessors=(2,),
        blocked_by=1,
        emits_detector_data=False,
    )
    finish = program_records.Operation(
        4,
        "readout",
        (0,),
        patches=(0,),
        predecessors=(3,),
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 2, 4)
    policy = round_policies.PerOperationRounds(
        ((100, 0), (1, 3), (2, 0), (3, 1), (4, 0))
    )
    return workload_settings.WorkloadSettings(
        operations=(prefix, begin, resume, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _assert_complete_record_matches_stim(source, stream_id, events) -> None:
    circuit = source.executed_circuit(stream_id)
    measurements = source.sampled_measurements(stream_id)
    raw = numpy.array([measurements], dtype=numpy.bool_)
    converter = circuit.compile_m2d_converter()
    expected_events, expected_truth = converter.convert(
        measurements=raw, separate_observables=True
    )
    numpy.testing.assert_array_equal(events, expected_events[0])
    truth = source.logical_observable_truth(stream_id)
    numpy.testing.assert_array_equal(truth, expected_truth[0])


@dataclasses.dataclass(frozen=True)
class _DeclaredFragments(circuit_records.RepeatedStimCircuit):
    """Overrides round_circuit and returns the declared fragment unchanged."""

    def round_circuit(self, round_index: int, is_final: bool) -> stim.Circuit:
        """The fragment the base class declares."""
        declared = circuit_records.RepeatedStimCircuit.round_circuit
        return declared(self, round_index, is_final)


def _final_reaching_program() -> circuit_records.RepeatedStimCircuit:
    """Round 1 reads out four times; the final round's rec[-4] reads back.

    Repeated rounds read only themselves.
    """
    first = stim.Circuit("R 0\nREPEAT 4 {\nM 0\nDETECTOR rec[-1]\n}")
    repeated = stim.Circuit("M 0\nDETECTOR rec[-1]")
    readout = stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]")
    final = stim.Circuit("M 0\nDETECTOR rec[-1] rec[-4]") + readout
    single = first + readout
    return circuit_records.RepeatedStimCircuit(first, repeated, final, single)
