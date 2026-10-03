"""A terminal live readout must reach decoders with its actual final model.

The normal zero-latency fabric exercises same-tick delivery at the protected
seal boundary, using Stim's generated memory and functional PyMatching.
"""

import dataclasses
from typing import Optional, Union

import numpy
import pytest

import decsim.config as config
import decsim.controller.policies as idle_policies
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings
import tests.declared_run as declared_run
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the default QPU round period, 1.1 microseconds, in ticks
ROUND_TICKS = 1_100_000

IdlePolicySettings = Union[
    idle_policies.SeparateDecodeJobsSettings, idle_policies.IgnoreSettings
]

# each test below runs once under each idle policy
SEPARATE_DECODE_JOBS = idle_policies.SeparateDecodeJobsSettings()
IGNORE = idle_policies.IgnoreSettings()
BOTH_IDLE_POLICIES = pytest.mark.parametrize(
    "idle_policy",
    [SEPARATE_DECODE_JOBS, IGNORE],
    ids=["separate_decode_jobs", "ignore"],
)


@pytest.fixture(params=["live", "finite"])
def stream_source(request: pytest.FixtureRequest) -> tuple:
    """(owner, owner round count, source) of a live or a finite stream."""
    builder_by_kind = {"live": _live_stream, "finite": _finite_stream}
    build = builder_by_kind[request.param]
    return build()


@pytest.mark.parametrize("final_round", [4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("data_hop_ticks", [0, 1])
def test_final_readout_uses_the_final_model_before_same_tick_decoding(
    final_round: int, data_hop_ticks: int
) -> None:
    workload = _workload(final_round)
    machine, source = _machine(workload, data_hop_ticks)
    packets = []
    machine.qpu.device.trace.round_emitted.connect(packets.append)
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


@pytest.mark.parametrize("segment_count", [1, 2])
def test_a_live_stream_that_no_region_ends_is_refused_at_its_seal(
    segment_count: int,
) -> None:
    """Only a protected region's end reads a live stream out.

    A workload whose live stream has no region never asks for the
    destructive readout, so the seal at the workload's end is refused
    rather than reported with no logical outcome.
    """
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    first = _segment(1, 0, ())
    second = _segment(2, 3, (1,))
    segments = (first, second)[:segment_count]
    policy = round_policies.PerOperationRounds(((100, 0), (1, 3), (2, 3)))
    workload = workload_settings.WorkloadSettings(
        operations=segments, dynamic_streams=(owner,), rounds_policy=policy
    )
    machine, _ = _machine(workload, 0)
    with pytest.raises(RuntimeError):
        machine.run()


@pytest.mark.parametrize("readout_round", [8, 10])
def test_a_region_a_released_operation_starts_keeps_the_qpu_cycle(
    readout_round: int,
) -> None:
    """The release lands between two cycle edges; the region's rounds do not.

    The waiting operation starts the region the moment its decode
    releases it, and its readout ends the region on a cycle edge. Every
    cycle reads one round out on its edge, and the executed record is
    Stim's own memory circuit of that many rounds.
    """
    workload = _released_region_workload(readout_round)
    machine, source = _machine(workload, 0)
    emission_ticks = []
    listener = _emission_recorder(machine, emission_ticks)
    machine.qpu.device.trace.round_emitted.connect(listener)
    result = machine.run()
    assert result.terminal_status == "complete"
    after_last_round = readout_round + 1
    cycles = range(1, after_last_round)
    every_cycle_edge = [cycle * 1_000_000 for cycle in cycles]
    assert emission_ticks == every_cycle_edge
    _assert_complete_record(source, readout_round)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("round_count", [3, 4])
def test_a_finite_source_closed_at_its_end_keeps_its_own_windows(
    round_count: int, distance: int
) -> None:
    """The source fixed the stream's windows, and its end closes them.

    A feedback source that spans the whole finite circuit closes the
    stream on its last round, so measurement_closed and trailing_buffer
    commit the same windows.
    """
    closed = _finite_windows(round_count, distance, "measurement_closed")
    trailing = _finite_windows(round_count, distance, "trailing_buffer")
    assert closed == trailing


def test_a_finished_finite_stream_leaves_its_idle_patch_to_the_policy() -> None:
    """A three-round circuit fed back in full has no fourth round to give.

    The waiting operation on another patch is released by the prefix's
    closed boundary, and the prefix's patch idles past the circuit's
    end; those idle rounds are the idle policy's, as for any patch.
    """
    circuit = memory_programs.memory_circuit(3, 3)
    owner = program_records.Operation(
        100, "memory", (0,), patches=(0,), circuit=circuit
    )
    prefix = dataclasses.replace(
        owner, id=1, name="prefix", stream_id=100, stream_offset=0
    )
    waiting = program_records.Operation(
        2,
        "waiting",
        (1,),
        patches=(1,),
        predecessors=(1,),
        blocked_by=1,
        emits_detector_data=False,
    )
    policy = round_policies.PerOperationRounds(((100, 3), (1, 3), (2, 2)))
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, waiting),
        dynamic_streams=(owner,),
        rounds_policy=policy,
        feedback_boundary_mode="measurement_closed",
    )
    device = stim_device.StimDevice()
    machine = _stim_machine(workload, device)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert rounds == [1, 2, 3]


def test_a_finite_group_stream_gives_its_last_round_once() -> None:
    """Both patches of a finite stream read its last round out together.

    The two-patch prefix is fed back in full; its group idles past the
    circuit's nine rounds, and the policy takes the same idle rounds
    from each patch once the stream has none left.
    """
    workload, device = _finite_group_workload()
    machine = _stim_machine(workload, device)
    idle_rounds = {0: [], 1: []}

    def listener(operation_id, patch, round_index) -> None:
        del operation_id
        idle_rounds[patch].append(round_index)

    machine.control.idle_rounds.trace.idle_round_emitted.connect(listener)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert idle_rounds[0] == idle_rounds[1]


def test_an_issued_continuation_takes_the_streams_next_round() -> None:
    """A continuation follows the idle round read out as it starts.

    The continuation is issued at round 8, before that cycle's edge; the
    edge reads the idle patch's round 8 out first, so the continuation
    runs rounds 9 to 11 and the stream executes every cycle, in order.
    """
    owner, owner_round_count, source = _live_stream()
    resumed = _continuation(owner, 2, 1, scheduled_start_round=8)
    protect, readout = _protected_tail(2, None, 15)
    counts = {100: owner_round_count, 1: 3, 2: 3, 3: 0, 4: 0}
    later = (resumed, protect, readout)
    workload = _region_workload(owner, later, counts)
    machine = _stim_machine(workload, source)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert rounds == list(range(1, 16))


def test_a_second_continuation_follows_the_idle_rounds_before_it() -> None:
    """The patch idles on its stream between two start-bound segments.

    The first continuation runs rounds 9 to 11; the patch then reads
    rounds 12 and 13 out while it idles, so the second, started at
    round 13, runs rounds 14 to 16.
    """
    owner, owner_round_count, source = _live_stream()
    resumed = _continuation(owner, 2, 1, scheduled_start_round=8)
    again = _continuation(owner, 5, 2, scheduled_start_round=13)
    protect, readout = _protected_tail(5, None, 20)
    counts = {100: owner_round_count, 1: 3, 2: 3, 3: 0, 4: 0, 5: 3}
    later = (resumed, again, protect, readout)
    workload = _region_workload(owner, later, counts)
    machine = _stim_machine(workload, source)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    binding = machine.control.issuer.stream_binding_for(5)
    assert result.terminal_status == "complete"
    assert rounds == list(range(1, 21))
    assert binding.stream_offset == 13


def test_a_continuation_gives_its_decision_the_stream_s_buffer() -> None:
    """The patch idles on its stream after a start-bound continuation.

    Protect waits for the continuation's decision, and the continuation's
    last window reads its buffer from the rounds the idle patch reads
    out, so the decision arrives and the run ends.
    """
    owner, owner_round_count, source = _live_stream()
    resumed = _continuation(owner, 2, 1, scheduled_start_round=6)
    protect, readout = _protected_tail(2, 2, 15)
    counts = {100: owner_round_count, 1: 3, 2: 3, 3: 0, 4: 0}
    later = (resumed, protect, readout)
    workload = _region_workload(owner, later, counts)
    machine = _stim_machine(workload, source)
    guard = _refuse_past(100_000_000)
    machine.engine.action_done.connect(guard)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert rounds == list(range(1, 16))


def test_a_released_continuation_s_result_is_its_own_windows(
    stream_source: tuple,
) -> None:
    """A continuation after idle rounds starts a window of its stream.

    The prefix's decision releases the continuation at round 7, after
    four idle rounds, so it runs rounds 8 to 10; protect waits for its
    result, which is the sum over the windows of those rounds alone.
    """
    owner, owner_round_count, source = stream_source
    resumed = _continuation(owner, 2, 1, blocked_by=1)
    protect, readout = _protected_tail(2, 2, 24)
    counts = {100: owner_round_count, 1: 3, 2: 3, 3: 0, 4: 0}
    later = (resumed, protect, readout)
    workload = _region_workload(owner, later, counts)
    machine = _stim_machine(workload, source)
    guard = _refuse_past(100_000_000)
    machine.engine.action_done.connect(guard)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    binding = machine.control.issuer.stream_binding_for(2)
    assert result.terminal_status == "complete"
    assert rounds == list(range(1, 25))
    assert binding.stream_offset == 7


def test_a_continuation_s_last_round_ends_a_window_of_its_source() -> None:
    """A continuation's result is its own windows though idle rounds follow.

    The fifteen-round prefix's decision releases the continuation at
    round 19, so it runs rounds 20 to 22 of the 24-round source, whose
    last window the scheme lays over rounds 20 to 24; readout waits for
    the continuation's result and then idles out the source.
    """
    owner, owner_round_count, source = _finite_stream()
    prefix = dataclasses.replace(
        owner, id=1, name="prefix", stream_id=100, stream_offset=0
    )
    resumed = _continuation(owner, 2, 1, blocked_by=1)
    readout = program_records.Operation(
        4,
        "readout",
        (0,),
        patches=(0,),
        predecessors=(2,),
        blocked_by=2,
        emits_detector_data=False,
    )
    counts = {100: owner_round_count, 1: 15, 2: 3, 4: 0}
    count_items = counts.items()
    count_pairs = tuple(count_items)
    policy = round_policies.PerOperationRounds(count_pairs)
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, resumed, readout),
        dynamic_streams=(owner,),
        rounds_policy=policy,
    )
    machine = _stim_machine(workload, source)
    guard = _refuse_past(100_000_000)
    machine.engine.action_done.connect(guard)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    binding = machine.control.issuer.stream_binding_for(2)
    assert result.terminal_status == "complete"
    assert rounds == list(range(1, 25))
    assert binding.stream_offset == 19


def test_a_continuation_keeps_a_later_declared_segment_s_windows() -> None:
    """A continuation bound at start keeps a declared segment's cuts.

    Segment 5 declares rounds 19 to 21 of the 36-round source at load;
    the continuation released at round 7 replans the windows after it,
    and protect waits for segment 5's result, its own windows.
    """
    circuit = memory_programs.memory_circuit(36, 3)
    owner = program_records.Operation(
        100, "memory", (0,), patches=(0,), circuit=circuit
    )
    resumed = _continuation(owner, 2, 1, blocked_by=1)
    declared = dataclasses.replace(
        owner,
        id=5,
        name="declared",
        stream_id=100,
        stream_offset=18,
        predecessors=(2,),
        scheduled_start_round=19,
    )
    protect, readout = _protected_tail(5, 5, 37)
    counts = {100: 36, 1: 3, 2: 3, 5: 3, 3: 0, 4: 0}
    later = (resumed, declared, protect, readout)
    workload = _region_workload(owner, later, counts)
    source = stim_device.StimDevice()
    machine = _stim_machine(workload, source)
    guard = _refuse_past(100_000_000)
    machine.engine.action_done.connect(guard)
    rounds = _emitted_rounds(machine)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert rounds == list(range(1, 37))


@BOTH_IDLE_POLICIES
def test_a_stream_group_shrunk_to_one_patch_runs_to_completion(
    idle_policy: IdlePolicySettings,
) -> None:
    """A stream on patches 0 and 1, then an operation on patch 0 alone.

    Once the operation starts, patch 1 holds no stream and idles for the
    policy, and the run completes.
    """
    ticks = _shrunk_group_run(None, idle_policy)
    assert ("started", 2) in ticks


@BOTH_IDLE_POLICIES
def test_a_shrunk_group_gives_the_waiting_decision_its_buffer(
    idle_policy: IdlePolicySettings,
) -> None:
    """While the operation on patch 0 waits, both patches stay idle.

    Both continue the stream, so its last window reads its three buffer
    rounds from them and the release follows the third.
    """
    ticks = _shrunk_group_run(1, idle_policy)
    wait_ticks = ticks[("released", 2)] - ticks[("finished", 1)]
    three_rounds = 3 * ROUND_TICKS
    four_rounds = 4 * ROUND_TICKS
    assert three_rounds < wait_ticks < four_rounds


@pytest.mark.parametrize("distance", [3, 5])
@BOTH_IDLE_POLICIES
@pytest.mark.parametrize("waiting_patch", [0, 1])
def test_a_closed_stream_boundary_releases_as_a_finite_operation_does(
    idle_policy: IdlePolicySettings, waiting_patch: int, distance: int
) -> None:
    """The stream's last window is queued before the seal reaches it.

    A measurement_closed boundary lets the prefix's window decode on its
    own rounds, so the waiting operation is released as when the prefix
    is an ordinary finite operation, less the finite prefix's closing
    layer: its last round reads the data qubits out and forms c/2 more
    events, which the open stream does not, and they cross two links
    first. Both start the waiting operation at the same round boundary.
    At distance 5 the three-round prefix ends inside a commit region,
    and that window commits through the boundary as the finite
    operation's one window does. The stream run then also decodes the
    rounds its idle patch kept producing, so it ends later.
    """
    closed = "measurement_closed"
    stream_prefix = _stream_prefix()
    finite_prefix = _finite_prefix()
    stream_run = _feedback_run(
        stream_prefix, idle_policy, waiting_patch, closed, distance
    )
    finite_run = _feedback_run(
        finite_prefix, idle_policy, waiting_patch, closed, distance
    )
    stream_events, stream_done_tick = stream_run
    finite_events, finite_done_tick = finite_run
    stream_ticks = _ticks_by_event(stream_events)
    finite_ticks = _ticks_by_event(finite_events)
    closing_ticks = _closing_layer_ticks(distance)
    stream_ticks[("released", 2)] += closing_ticks
    assert stream_ticks == finite_ticks
    assert stream_done_tick > finite_done_tick


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
    stream_prefix = _stream_prefix()
    charged = _feedback_run(
        stream_prefix, SEPARATE_DECODE_JOBS, waiting_patch, trailing
    )
    ignored = _feedback_run(stream_prefix, IGNORE, waiting_patch, trailing)
    events = charged[0]
    assert events == ignored[0]
    ticks = _ticks_by_event(events)
    wait_ticks = ticks[("released", 2)] - ticks[("finished", 1)]
    three_rounds = 3 * ROUND_TICKS
    assert wait_ticks > three_rounds


@pytest.mark.parametrize("waiting_patch", [0, 1])
def test_a_zero_round_step_does_not_hide_the_blocked_operation(
    waiting_patch: int,
) -> None:
    """The prefix's boundary closes when anything is blocked by it.

    A zero-round operation between the prefix and the operation it
    releases leaves the release where it is without that step.
    """
    closed = "measurement_closed"
    direct_operations = _direct_operations(waiting_patch)
    stepped_operations = _stepped_operations(waiting_patch)
    direct = _prefix_run(direct_operations, closed)
    stepped = _prefix_run(stepped_operations, closed)
    assert stepped == direct
    release_wait_ticks, _ = stepped
    assert release_wait_ticks < 1_100_000


@pytest.mark.parametrize("waiting_patch", [0, 1])
def test_an_operation_blocked_elsewhere_leaves_the_boundary_open(
    waiting_patch: int,
) -> None:
    """A successor blocked by another operation does not close the prefix.

    The prefix is no feedback source, so measurement_closed leaves its
    window as trailing_buffer does: the release and the commit agree.
    """
    operations = _elsewhere_operations(waiting_patch)
    closed = _prefix_run(operations, "measurement_closed")
    trailing = _prefix_run(operations, "trailing_buffer")
    assert closed == trailing


def _released_region_workload(
    readout_round: int,
) -> workload_settings.WorkloadSettings:
    """A prefix, then a region its decision starts and a readout ends."""
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    prefix = _segment(1, 0, ())
    waiting = program_records.Operation(
        2,
        "waiting",
        (0,),
        patches=(0,),
        predecessors=(1,),
        blocked_by=1,
        emits_detector_data=False,
    )
    readout = program_records.Operation(
        3,
        "readout",
        (0,),
        patches=(0,),
        predecessors=(2,),
        scheduled_start_round=readout_round,
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 2, 3)
    policy = round_policies.PerOperationRounds(
        ((100, 0), (1, 3), (2, 1), (3, 0))
    )
    return workload_settings.WorkloadSettings(
        operations=(prefix, waiting, readout),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _emission_recorder(machine: machine_module.Machine, emission_ticks: list):
    """A listener that records the tick each stream round is read out."""

    def listener(readout) -> None:
        del readout
        emission_ticks.append(machine.engine.now)

    return listener


def _finite_windows(round_count: int, distance: int, mode: str) -> list:
    """The committed windows of a finite Stim stream fed back in full."""
    circuit = memory_programs.memory_circuit(round_count, distance)
    owner = program_records.Operation(
        100, "memory", (0,), patches=(0,), circuit=circuit
    )
    prefix = dataclasses.replace(
        owner, id=1, name="prefix", stream_id=100, stream_offset=0
    )
    waiting = program_records.Operation(
        2,
        "waiting",
        (1,),
        patches=(1,),
        predecessors=(1,),
        blocked_by=1,
        emits_detector_data=False,
    )
    after_circuit = memory_programs.memory_circuit(3, distance)
    after = program_records.Operation(
        3, "after", (0,), patches=(0,), predecessors=(1,), circuit=after_circuit
    )
    counts = {100: round_count, 1: round_count, 2: 1, 3: 3}
    count_items = counts.items()
    count_pairs = tuple(count_items)
    policy = round_policies.PerOperationRounds(count_pairs)
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, waiting, after),
        dynamic_streams=(owner,),
        rounds_policy=policy,
        feedback_boundary_mode=mode,
    )
    return _committed_windows(workload, distance)


def _committed_windows(
    workload: workload_settings.WorkloadSettings, distance: int
) -> list:
    """The spans of the windows a Stim run of the workload commits."""
    device = stim_device.StimDevice()
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(distance=distance, source=source)
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=decoder
    )
    machine = machine_module.Machine.build(settings, 5)
    windows = []
    sources = machine.windows.window_manager.window_sources()
    recorder = _window_recorder(windows)
    sources.window_committed.connect(recorder)
    result = machine.run()
    assert result.terminal_status == "complete"
    return windows


def _window_recorder(windows: list):
    """A listener that records each committed window's span."""

    def listener(window, contribution) -> None:
        del contribution
        span = (window.key, window.commit_hi, window.buffer_hi)
        windows.append(span)

    return listener


def _finite_group_workload() -> tuple:
    """(workload, source) of a nine-round two-patch circuit, three fed back.

    A feedback-blocked operation on a third patch waits for the prefix;
    the detectors' rounds are the ones Stim's own record gives them.
    """
    program = memory_programs.joint_repetition_program(False)
    circuit, measurement_rounds = program.assemble(9)
    table = detector_formation.build_formation_table(
        circuit, 9, measurement_rounds=measurement_rounds
    )
    detector_rounds = table.detector_rounds()
    group = (0, 1)
    owner = program_records.Operation(
        100, "memory", group, patches=group, circuit=circuit
    )
    prefix = dataclasses.replace(
        owner, id=1, name="prefix", stream_id=100, stream_offset=0
    )
    waiting = program_records.Operation(
        2,
        "waiting",
        (2,),
        patches=(2,),
        predecessors=(1,),
        blocked_by=1,
        emits_detector_data=False,
    )
    policy = round_policies.PerOperationRounds(((100, 9), (1, 3), (2, 8)))
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, waiting),
        dynamic_streams=(owner,),
        rounds_policy=policy,
    )
    device = stim_device.StimDevice(
        measurement_rounds={100: measurement_rounds},
        detector_rounds={100: detector_rounds},
    )
    return workload, device


def _live_stream() -> tuple:
    """(owner, owner round count, source) of an open-ended live stream."""
    program = memory_programs.memory_program()
    source = streaming_stim_device.StreamingStimDevice(programs={100: program})
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    return owner, 0, source


def _finite_stream() -> tuple:
    """(owner, owner round count, source) of a 24-round Stim circuit."""
    circuit = memory_programs.memory_circuit(24, 3)
    owner = program_records.Operation(
        100, "memory", (0,), patches=(0,), circuit=circuit
    )
    source = stim_device.StimDevice()
    return owner, 24, source


def _continuation(
    owner: program_records.Operation,
    operation_id: int,
    predecessor_id: int,
    *,
    scheduled_start_round: int = 0,
    blocked_by: Optional[int] = None,
) -> program_records.Operation:
    """A segment of the owner's stream that declares no offset."""
    return dataclasses.replace(
        owner,
        id=operation_id,
        name=f"resumed{operation_id}",
        stream_id=100,
        predecessors=(predecessor_id,),
        scheduled_start_round=scheduled_start_round,
        blocked_by=blocked_by,
    )


def _protected_tail(
    last_id: int, blocked_by: Optional[int], readout_round: int
) -> tuple:
    """(protect, readout): the region that reads stream 100 out."""
    protect = program_records.Operation(
        3,
        "protect",
        (0,),
        patches=(0,),
        predecessors=(last_id,),
        blocked_by=blocked_by,
        emits_detector_data=False,
    )
    readout = dataclasses.replace(
        protect,
        id=4,
        name="readout",
        predecessors=(3,),
        blocked_by=None,
        scheduled_start_round=readout_round,
    )
    return protect, readout


def _region_workload(
    owner: program_records.Operation, later: tuple, counts: dict
) -> workload_settings.WorkloadSettings:
    """A three-round prefix of the owner's stream, then the later operations.

    Protect (3) and readout (4) bound the stream's protected region.
    """
    prefix = dataclasses.replace(
        owner, id=1, name="prefix", stream_id=100, stream_offset=0
    )
    region = program_records.ProtectedRegion(100, 3, 4)
    count_items = counts.items()
    count_pairs = tuple(count_items)
    policy = round_policies.PerOperationRounds(count_pairs)
    return workload_settings.WorkloadSettings(
        operations=(prefix, *later),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _stim_machine(
    workload: workload_settings.WorkloadSettings, device
) -> machine_module.Machine:
    """A d=3 run of the workload on a Stim source, one-microsecond rounds."""
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source, round_period_microseconds=1.0
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=decoder
    )
    return machine_module.Machine.build(settings, 5)


def _emitted_rounds(machine: machine_module.Machine) -> list:
    """The index of every round the QPU reads out, in order."""
    rounds = []

    def listener(packet) -> None:
        rounds.append(packet.round_index)

    machine.qpu.device.trace.round_emitted.connect(listener)
    return rounds


def _shrunk_group_run(blocked_by, idle_policy: IdlePolicySettings) -> dict:
    """The runtime's ticks for a two-patch stream then a patch-0 operation."""
    group = (0, 1)
    owner = program_records.Operation(100, "memory", group, patches=group)
    segment = dataclasses.replace(
        owner, id=1, name="segment", stream_id=100, stream_offset=0
    )
    later = program_records.Operation(
        2, "later", (0,), patches=(0,), predecessors=(1,), blocked_by=blocked_by
    )
    policy = round_policies.PerOperationRounds(((100, 0), (1, 3), (2, 1)))
    workload = workload_settings.WorkloadSettings(
        operations=(segment, later),
        dynamic_streams=(owner,),
        rounds_policy=policy,
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    settings = machine_settings.MachineSettings(
        workload=workload, weak_decoder=decoder, idle_policy=idle_policy
    )
    machine = machine_module.Machine.build(settings, 0)
    events, _ = _events_and_end(machine)
    return _ticks_by_event(events)


def _segment(
    operation_id: int, stream_offset: int, predecessors: tuple
) -> program_records.Operation:
    """A three-round segment of stream 100 on patch 0."""
    return program_records.Operation(
        operation_id,
        f"segment{operation_id}",
        (0,),
        patches=(0,),
        stream_id=100,
        stream_offset=stream_offset,
        predecessors=predecessors,
    )


def _closing_layer_ticks(distance: int) -> int:
    """The closing layer's c/2 events over two links, 125 ticks a bit.

    The controller-to-buffer and buffer-to-decoder links move 8000 bits
    a microsecond (link_profiles.logical_reference_profile).
    """
    data_qubit_count = distance * distance
    check_count = data_qubit_count - 1
    closing_events = check_count // 2
    link_bits = 2 * closing_events
    return link_bits * 125


def _ticks_by_event(events: tuple) -> dict:
    """The tick of each (kind, operation) event."""
    return {(kind, operation): tick for kind, operation, tick in events}


def _recorder(events: list, kind: str):
    """A listener that keeps each event under its kind."""

    def listener(*event) -> None:
        kept = (kind, *event)
        events.append(kept)

    return listener


def _prefix_run(operations: tuple, mode: str) -> tuple:
    """Ticks from the prefix's end to the release, and to its commit."""
    stream = program_records.Operation(100, "memory", (0,), patches=(0,))
    policy = round_policies.PerOperationRounds(
        ((100, 0), (1, 3), (2, 0), (3, 1), (4, 3))
    )
    workload = workload_settings.WorkloadSettings(
        operations=operations,
        dynamic_streams=(stream,),
        rounds_policy=policy,
        feedback_boundary_mode=mode,
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    settings = machine_settings.MachineSettings(
        workload=workload, weak_decoder=decoder
    )
    machine = machine_module.Machine.build(settings, 0)
    commit_ticks = {}
    sources = machine.windows.window_manager.window_sources()
    recorder = _commit_recorder(machine, commit_ticks)
    sources.window_committed.connect(recorder)
    events, _ = _events_and_end(machine)
    ticks = _ticks_by_event(events)
    prefix_end_tick = ticks[("finished", 1)]
    release_wait_ticks = ticks[("released", 3)] - prefix_end_tick
    commit_wait_ticks = commit_ticks[(100, 0)] - prefix_end_tick
    return release_wait_ticks, commit_wait_ticks


def _prefix_and_waiting(waiting_patch: int) -> tuple:
    """(prefix, waiting): the waiting operation follows the prefix."""
    prefix = program_records.Operation(
        1, "prefix", (0,), patches=(0,), stream_id=100, stream_offset=0
    )
    patches = (waiting_patch,)
    waiting = program_records.Operation(
        3,
        "waiting",
        patches,
        patches=patches,
        predecessors=(1,),
        blocked_by=1,
    )
    return prefix, waiting


def _direct_operations(waiting_patch: int) -> tuple:
    """The waiting operation, blocked by the prefix, follows it."""
    return _prefix_and_waiting(waiting_patch)


def _stepped_operations(waiting_patch: int) -> tuple:
    """A zero-round step sits between the prefix and the waiting one."""
    prefix, waiting = _prefix_and_waiting(waiting_patch)
    step = program_records.Operation(
        2,
        "step",
        (0,),
        patches=(0,),
        predecessors=(1,),
        emits_detector_data=False,
    )
    waiting = dataclasses.replace(waiting, predecessors=(2,))
    return prefix, step, waiting


def _elsewhere_operations(waiting_patch: int) -> tuple:
    """The waiting operation is blocked by a third operation."""
    prefix, waiting = _prefix_and_waiting(waiting_patch)
    other = program_records.Operation(4, "other", (2,), patches=(2,))
    waiting = dataclasses.replace(waiting, predecessors=(1, 4), blocked_by=4)
    return prefix, other, waiting


def _commit_recorder(machine: machine_module.Machine, commit_ticks: dict):
    """A listener that records when each window first commits."""

    def listener(window, contribution) -> None:
        del contribution
        commit_ticks.setdefault(window.key, machine.engine.now)

    return listener


def _feedback_run(
    prefix_shape: tuple,
    idle_policy: IdlePolicySettings,
    waiting_patch: int,
    mode: str,
    distance: int = 3,
) -> tuple:
    """When the prefix ends, the release lands and the run finishes."""
    prefix, streams, counts = prefix_shape
    patches = (waiting_patch,)
    waiting = program_records.Operation(
        2, "waiting", patches, patches=patches, predecessors=(1,), blocked_by=1
    )
    count_items = counts.items()
    count_pairs = tuple(count_items)
    policy = round_policies.PerOperationRounds(count_pairs)
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, waiting),
        dynamic_streams=streams,
        rounds_policy=policy,
        feedback_boundary_mode=mode,
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    qpu = qpu_settings.QpuSettings(distance=distance)
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=decoder,
        idle_policy=idle_policy,
    )
    machine = machine_module.Machine.build(settings, 0)
    return _events_and_end(machine)


def _stream_prefix() -> tuple:
    """(prefix, streams, round counts): a prefix on stream 100."""
    prefix = program_records.Operation(
        1, "prefix", (0,), patches=(0,), stream_id=100, stream_offset=0
    )
    stream = program_records.Operation(100, "memory", (0,), patches=(0,))
    return prefix, (stream,), {100: 0, 1: 3, 2: 1}


def _finite_prefix() -> tuple:
    """(prefix, streams, round counts): an ordinary three-round prefix."""
    prefix = program_records.Operation(1, "prefix", (0,), patches=(0,))
    return prefix, (), {1: 3, 2: 1}


def _events_and_end(machine: machine_module.Machine) -> tuple:
    """Run to the end; the runtime's events and the fully-done tick."""
    guard = _refuse_past(1_000_000_000)
    machine.engine.action_done.connect(guard)
    events = []
    trace = machine.control.execution_runtime.trace
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
        assert now <= limit_ticks, f"the run did not end by tick {limit_ticks}"

    return listener


def _machine(
    workload: workload_settings.WorkloadSettings, data_hop_ticks: int
) -> tuple[machine_module.Machine, streaming_stim_device.StreamingStimDevice]:
    program = memory_programs.memory_program()
    source = streaming_stim_device.StreamingStimDevice(programs={100: program})
    source_record = declared_run.GivenSource(source)
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source_record, round_period_microseconds=1.0
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
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
    policy = round_policies.PerOperationRounds(((100, 0), (1, 0), (2, 0)))
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
