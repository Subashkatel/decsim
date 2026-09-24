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
    own rounds, so the waiting operation is released as when the prefix
    is an ordinary finite operation, less the finite prefix's closing
    layer: its last round reads the data qubits out and forms c/2 more
    events, which the open stream does not, and they cross two links
    first. Both start the waiting operation at the same round boundary.
    The stream run then also decodes the rounds its idle patch kept
    producing, so it ends later.
    """
    closed = "measurement_closed"
    stream_run = _feedback_run(True, idle_policy, waiting_patch, closed)
    finite_run = _feedback_run(False, idle_policy, waiting_patch, closed)
    stream_events, stream_done_tick = stream_run
    finite_events, finite_done_tick = finite_run
    expected_events = _released_later(stream_events, 2, CLOSING_LAYER_TICKS)
    assert expected_events == finite_events
    assert stream_done_tick > finite_done_tick


# c/2 = 4 events at d=3, over the controller-to-buffer and
# buffer-to-decoder links at 8000 bits a microsecond, 125 ticks a bit
# (link_profiles.logical_reference_profile)
CLOSING_LAYER_TICKS = 2 * 4 * 125


def _released_later(events: tuple, operation_id: int, ticks: int) -> tuple:
    """The events with one operation's release moved later by ticks."""
    moved = []
    for kind, identity, tick in events:
        if (kind, identity) == ("released", operation_id):
            tick += ticks
        moved.append((kind, identity, tick))
    return tuple(moved)


def _recorder(events: list, kind: str):
    """A listener that keeps each event under its kind."""

    def listener(*event) -> None:
        kept = (kind, *event)
        events.append(kept)

    return listener


@pytest.mark.parametrize("waiting_patch", [0, 1])
def test_a_waiting_stream_decision_reads_its_buffer_from_the_idle_patch(
    waiting_patch: int,
) -> None:
    """The prefix's patch keeps its qubit in memory while the release waits.

    Each idle cycle is the stream's next round whatever the idle policy,
    so the prefix's last window reads its three buffer rounds from them:
    the release follows the third idle round, under both policies alike.
    """
    trailing = "trailing_buffer"
    charged = _feedback_run(
        True, "separate_decode_jobs", waiting_patch, trailing
    )
    ignored = _feedback_run(True, "ignore", waiting_patch, trailing)
    events = charged[0]
    assert events == ignored[0]
    ticks = {(kind, operation): tick for kind, operation, tick in events}
    wait_ticks = ticks[("released", 2)] - ticks[("finished", 1)]
    assert wait_ticks > 3 * 1_100_000


def _feedback_run(
    is_stream: bool, idle_policy: str, waiting_patch: int, mode: str
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
        feedback_boundary_mode=mode,
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    decoder = decoder_settings.DecoderSettings(kind=0.1, engine=engine)
    idle = controller_settings.IdlePolicySettings(kind=idle_policy)
    settings = machine_settings.MachineSettings(
        workload=workload, weak_decoder=decoder, idle_policy=idle
    )
    machine = machine_module.Machine.build(settings, 0)
    return _events_and_end(machine)


def _events_and_end(machine: machine_module.Machine) -> tuple:
    """Run to the end; the runtime's events and the fully-done tick."""
    guard = _refuse_past(1_000_000_000)
    machine.engine.action_done.connect(guard)
    events = []
    trace = machine.execution_runtime.trace
    finished = _recorder(events, "finished")
    released = _recorder(events, "released")
    started = _recorder(events, "started")
    trace.body_finished.connect(finished)
    trace.decode_released.connect(released)
    trace.operation_started.connect(started)
    result = machine.run()
    return tuple(events), result.fully_done_ticks


def _refuse_past(limit_ticks: int):
    """A listener that fails a run still going past the limit.

    A decision that never gets its buffer leaves the idle rounds
    ticking forever, so the run fails here instead of hanging.
    """

    def listener(now: int) -> None:
        if now > limit_ticks:
            raise AssertionError(f"the run did not end by tick {limit_ticks}")

    return listener


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
    engine = decoder_settings.EngineSettings(clock=clock)
    decoder = decoder_settings.DecoderSettings(kind=0.1, engine=engine)
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
