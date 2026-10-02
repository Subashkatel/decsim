"""Physical Stim histories remain independent of decoder model selection.

Stim's complete-circuit measurement converter checks the single raw record
emitted across an explicit prefix and feedback-dependent protected rounds.
"""

import dataclasses
import types
from typing import Optional

import numpy
import pytest
import stim

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.ports as ports
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.settings as machine_settings
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


@pytest.mark.parametrize("producer", ["stim", "deltakit"])
def test_separate_live_models_never_execute_a_physical_history(
    producer: str,
) -> None:
    program = _program(producer)
    source = streaming_stim_device.StreamingStimDevice({100: program})
    models = streaming_stim_device.StreamingStimDevice({100: program})
    samples = []
    model_samples = []
    source.shot_sampled.connect(lambda *sample: samples.append(sample))
    models.shot_sampled.connect(lambda *sample: model_samples.append(sample))
    workload = _workload()
    machine = _machine(source, models, workload, 4.0)
    assert samples == []
    assert source.sampled_measurements(100) == ()
    assert source.logical_observable_truth(100) is None
    empty = stim.Circuit()
    assert source.executed_circuit(100) == empty
    packets = []
    machine.qpu.device.trace.round_emitted.connect(packets.append)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert result.decode_work_settled
    assert result.event_queue_empty
    assert len(samples) == 1
    assert model_samples == []
    assert models.logical_observable_truth(100) is None
    with pytest.raises(KeyError):
        models.sampled_measurements(100)
    circuit = source.executed_circuit(100)
    _assert_record(source, circuit, packets)
    truth = source.logical_observable_truth(100)
    assert result.operation_results[-1].observable_truth == truth


@pytest.mark.parametrize("producer", ["stim", "deltakit"])
def test_separate_models_preserve_the_prefix_across_feedback_waits(
    producer: str,
) -> None:
    program = _program(producer)
    faster, shorter = _live_machine(program, 4.0)
    slower, longer = _live_machine(program, 8.0)
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
    first_record = shorter.sampled_measurements(100)
    second_record = longer.sampled_measurements(100)
    before_final_count = len(first_record) - first_packets[-1].size_bits
    assert (
        first_record[:before_final_count] == second_record[:before_final_count]
    )
    assert first_packets[-1].size_bits > first_packets[-2].size_bits
    assert second_packets[-1].size_bits > second_packets[-2].size_bits


@pytest.mark.parametrize("model_horizon", ["finite", "unbounded"])
def test_finite_source_keeps_one_shot_with_separate_models(
    model_horizon: str,
) -> None:
    program = memory_programs.memory_program()
    circuit, _ = program.assemble(24)
    source = stim_device.StimDevice()
    models = _models(program, model_horizon)
    samples = []
    model_samples = []
    source.shot_sampled.connect(lambda *sample: samples.append(sample))
    models.shot_sampled.connect(lambda *sample: model_samples.append(sample))
    workload = _workload(circuit, 24, 24)
    machine = _machine(source, models, workload, 4.0)
    assert source.sampled_truth() == {}
    assert samples == []
    packets = []
    machine.qpu.device.trace.round_emitted.connect(packets.append)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert result.decode_work_settled
    assert result.event_queue_empty
    assert len(packets) == 24
    assert len(samples) == 1
    assert model_samples == []
    _assert_record(source, circuit, packets)
    truth = source.logical_observable_truth(100)
    assert source.logical_observable_truth(1) == truth
    assert result.operation_results[-1].observable_truth == truth
    assert models.logical_observable_truth(100) is None


def test_unbounded_models_cannot_remove_the_finite_source_seal_limit() -> None:
    program = memory_programs.memory_program()
    circuit, _ = program.assemble(24)
    source = stim_device.StimDevice()
    models = streaming_stim_device.StreamingStimDevice({100: program})
    workload = _workload(circuit, 24, 10)
    machine = _machine(source, models, workload, 4.0)
    with pytest.raises(
        RuntimeError, match="sealed at .*registered for 24 rounds"
    ):
        machine.run()


def test_a_live_stream_sealed_short_of_its_finite_models_stops_the_run():
    """The live source ends at 12 rounds; the models were built for 24.

    The source's own length check passes, since its history really did
    end there, and the decoder input then misses the models' rows.
    """
    program = memory_programs.memory_program()
    circuit, _ = program.assemble(24)
    source = streaming_stim_device.StreamingStimDevice({100: program})
    models = stim_device.StimDevice()
    workload = _workload(circuit, 24, 10)
    machine = _machine(source, models, workload, 4.0)
    with pytest.raises(RuntimeError):
        machine.run()


def test_unbounded_models_cannot_extend_the_finite_physical_source() -> None:
    program = memory_programs.memory_program()
    circuit, _ = program.assemble(24)
    source = stim_device.StimDevice()
    models = streaming_stim_device.StreamingStimDevice({100: program})
    workload = _workload(circuit, 24, 30)
    machine = _machine(source, models, workload, 4.0)
    with pytest.raises(
        ValueError, match="idle round is outside the finite source"
    ):
        machine.run()


def test_conflicting_stream_limits_are_refused_before_sampling() -> None:
    program = memory_programs.memory_program()
    circuit, _ = program.assemble(24)
    source = stim_device.StimDevice()
    models = _FixedHorizonModels(7)
    samples = []
    source.shot_sampled.connect(lambda *sample: samples.append(sample))
    workload = _workload(circuit, 24, 24)
    message = "physical and model stream round limits must agree"
    with pytest.raises(ValueError, match=message):
        _machine(source, models, workload, 4.0)
    assert samples == []
    assert source.sampled_truth() == {}
    assert models.sampled_truth() == {}


def test_timing_only_source_preserves_the_finite_models_circuit_copy() -> None:
    circuit = memory_programs.memory_circuit(24)
    code = code_geometry.SurfaceCodeModel(distance=3)
    source = syndrome_devices.TimingOnlyDevice(code)
    models = stim_device.StimDevice()
    workload = _workload(circuit, 24, 24)
    machine = _machine(source, models, workload, 4.0)
    copied = machine.plan.all_operations[0].circuit
    assert copied == circuit
    assert copied is not circuit
    assert models.sampled_truth() == {}
    assert source.logical_observable_truth(100) is None
    circuit.append("X", [0])
    assert copied != circuit


@pytest.mark.parametrize(
    "declaration",
    [
        {"operation_circuit_scope": None},
        {"operation_circuit_scope": "shared"},
    ],
)
def test_model_circuit_scope_is_required_at_the_root_boundary(
    declaration: dict,
) -> None:
    code = code_geometry.SurfaceCodeModel(distance=3)
    source = syndrome_devices.TimingOnlyDevice(code)
    models = types.SimpleNamespace(**declaration)
    workload = _workload()
    message = (
        "model provider operation_circuit_scope must be none or per_operation"
    )
    with pytest.raises(ValueError, match=message):
        _machine(source, models, workload, 4.0)


def _models(program, model_horizon):
    if model_horizon == "finite":
        return stim_device.StimDevice()
    return streaming_stim_device.StreamingStimDevice({100: program})


def _program(producer: str) -> circuit_records.RepeatedStimCircuit:
    if producer == "stim":
        return memory_programs.memory_program()
    pytest.importorskip("deltakit_explorer")
    return deltakit.memory_rounds(
        "rotated_surface", 3, "Z", 0.003, round_period_microseconds=1.1
    )


def _live_machine(program, feedback_microseconds):
    source = streaming_stim_device.StreamingStimDevice({100: program})
    models = streaming_stim_device.StreamingStimDevice({100: program})
    workload = _workload()
    machine = _machine(source, models, workload, feedback_microseconds)
    return machine, source


def _machine(
    source: ports.SyndromeSource,
    models: ports.WindowModelSource,
    workload: workload_settings.WorkloadSettings,
    feedback_microseconds: float,
) -> machine_module.Machine:
    qpu = qpu_settings.QpuSettings(
        distance=3,
        device=source,
        error_model_provider=models,
        round_period_microseconds=1.1,
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
    return machine_module.Machine.build(settings, 17)


def _workload(circuit=None, source_round_count=0, final_round=0):
    """Keep the dependency graph together so the shared source is visible."""
    owner = program_records.Operation(
        100, "memory", (0,), patches=(0,), circuit=circuit
    )
    prefix = program_records.Operation(
        1,
        "prefix",
        (0,),
        patches=(0,),
        circuit=circuit,
        stream_id=100,
        stream_offset=0,
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
        scheduled_start_round=final_round,
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 2, 4)
    counts = {100: source_round_count, 1: 3, 2: 0, 3: 1, 4: 0}
    policy = round_policies.PerOperationRounds(counts)
    return workload_settings.WorkloadSettings(
        operations=(prefix, begin, resume, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _assert_record(source, circuit: stim.Circuit, packets: list) -> None:
    measurements = _measurements(packets)
    assert len(measurements) == circuit.num_measurements
    raw = numpy.array([measurements], dtype=numpy.bool_)
    converter = circuit.compile_m2d_converter()
    events, truth = converter.convert(
        measurements=raw, separate_observables=True
    )
    actual_events = source.sampled_detection_events(100)
    actual_truth = source.logical_observable_truth(100)
    numpy.testing.assert_array_equal(actual_events, events[0])
    numpy.testing.assert_array_equal(actual_truth, truth[0])


def _measurements(packets: list[round_records.QPUReadout]) -> tuple[int, ...]:
    return tuple(bit for packet in packets for bit in packet.bits)


class _FixedHorizonModels(stim_device.StimDevice):
    """Represent an independently calibrated finite decoder model horizon."""

    def __init__(self, round_count: int) -> None:
        parent = super()
        parent.__init__()
        self.round_count = round_count

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> Optional[int]:
        del round_count
        circuit = memory_programs.memory_circuit(self.round_count)
        model_owner = dataclasses.replace(stream_operation, circuit=circuit)
        parent = super()
        return parent.register_dynamic_stream(
            model_owner,
            self.round_count,
            fault_model_requirement=fault_model_requirement,
        )
