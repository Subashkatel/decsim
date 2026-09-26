"""A terminal live readout must reach decoders with its actual final model.

The normal zero-latency fabric exercises same-tick delivery at the protected
seal boundary, using Stim's generated memory and functional PyMatching.
"""

import dataclasses

import numpy
import pytest

import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings
import tests.qpu.memory_programs as memory_programs


@pytest.mark.parametrize("final_round", [4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("data_hop_ticks", [0, 1])
def test_final_readout_uses_the_final_model_before_same_tick_decoding(
    final_round: int, data_hop_ticks: int
) -> None:
    machine, source = _machine(final_round, data_hop_ticks)
    packets = []
    machine.qpu.trace.round_emitted.connect(packets.append)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert result.decode_work_settled
    assert result.event_queue_empty
    assert len(packets) == final_round
    assert packets[-1].size_bits == 17
    assert packets[-1].round_index == final_round
    _assert_complete_record(source, final_round)
    owner_result = result.operation_results[-1]
    truth = source.logical_observable_truth(100)
    assert owner_result.operation_id == 100
    assert owner_result.observable_truth == truth
    assert owner_result.logical_failure is not None


@pytest.mark.parametrize("idle_policy", ["separate_decode_jobs", "ignore"])
@pytest.mark.parametrize("waiting_patch", [0, 1])
def test_a_closed_stream_boundary_releases_as_a_finite_operation_does(
    idle_policy: str, waiting_patch: int
) -> None:
    """The stream's last window is queued before the seal reaches it.

    A measurement_closed boundary lets the prefix's window decode on its
    own rounds, so the waiting operation starts at the same tick as when
    the prefix is an ordinary finite operation.
    """
    stream_run = _closed_boundary_run(True, idle_policy, waiting_patch)
    finite_run = _closed_boundary_run(False, idle_policy, waiting_patch)
    assert stream_run == finite_run


def _closed_boundary_run(
    is_stream: bool, idle_policy: str, waiting_patch: int
) -> tuple:
    """When the prefix ends, the release lands and the run finishes."""
    prefix = program_records.Operation(1, "prefix", (0,), patches=(0,))
    streams = ()
    counts = {1: 3, 2: 1}
    if is_stream:
        prefix = dataclasses.replace(prefix, stream_id=100, stream_offset=0)
        stream = program_records.Operation(100, "memory", (0,), patches=(0,))
        streams = (stream,)
        counts[100] = 0
    patches = (waiting_patch,)
    waiting = program_records.Operation(
        2, "waiting", patches, patches=patches, predecessors=(1,), blocked_by=1
    )
    policy = round_policies.PerOperationRounds(counts)
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, waiting),
        dynamic_streams=streams,
        rounds_policy=policy,
        feedback_boundary_mode="measurement_closed",
    )
    clock = config.Clock(1000)
    decoder = decoder_settings.DecoderSettings(kind=0.1, engine_clock=clock)
    idle = controller_settings.IdlePolicySettings(kind=idle_policy)
    settings = machine_settings.MachineSettings(
        workload=workload, weak_decoder=decoder, idle_policy=idle
    )
    machine = machine_module.Machine.build(settings, 0)
    events = []
    trace = machine.execution_runtime.trace
    trace.body_finished.connect(lambda *event: events.append(event))
    trace.decode_released.connect(lambda *event: events.append(event))
    trace.operation_started.connect(lambda *event: events.append(event))
    result = machine.run()
    return tuple(events), result.fully_done_ticks


def _machine(
    final_round: int, data_hop_ticks: int
) -> tuple[machine_module.Machine, streaming_stim_device.StreamingStimDevice]:
    program = memory_programs.memory_program()
    source = streaming_stim_device.StreamingStimDevice(programs={100: program})
    workload = _workload(final_round)
    qpu = qpu_settings.QpuSettings(
        distance=3, device=source, round_period_microseconds=1.0
    )
    clock = config.Clock(1000)
    decoder = decoder_settings.DecoderSettings(kind=0.1, engine_clock=clock)
    links = _zero_delay_links(data_hop_ticks)
    observation = observation_settings.ObservationSettings(
        trace="chrome", data_movement=True
    )
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=decoder,
        links=links,
        observation=observation,
    )
    machine = machine_module.Machine.build(settings, 17)
    return machine, source


def _workload(final_round: int) -> workload_settings.WorkloadSettings:
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    begin = program_records.Operation(
        1, "protect", (0,), patches=(0,), emits_detector_data=False
    )
    finish = program_records.Operation(
        2,
        "readout",
        (0,),
        patches=(0,),
        predecessors=(1,),
        scheduled_start_round=final_round,
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 1, 2)
    policy = round_policies.PerOperationRounds({100: 0, 1: 0, 2: 0})
    return workload_settings.WorkloadSettings(
        operations=(begin, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _zero_delay_links(data_hop_ticks: int) -> link_settings.FabricSettings:
    reference = link_profiles.logical_reference_profile()
    paths = {}
    for path in transfer_records.LinkPath:
        settings = reference.path_settings(path)
        channel = dataclasses.replace(
            settings.channel, propagation_latency_ticks=0
        )
        paths[path.value] = dataclasses.replace(settings, channel=channel)
    readout = paths["qpu_to_controller"]
    channel = dataclasses.replace(
        readout.channel, propagation_latency_ticks=data_hop_ticks
    )
    paths["qpu_to_controller"] = dataclasses.replace(readout, channel=channel)
    return dataclasses.replace(reference, **paths)


def _assert_complete_record(
    source: streaming_stim_device.StreamingStimDevice, final_round: int
) -> None:
    circuit = source.executed_circuit(100)
    expected = memory_programs.memory_circuit(final_round)
    flattened = circuit.flattened()
    expected_flattened = expected.flattened()
    assert flattened == expected_flattened
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
