"""The windows facade's laws through its public surface.

The status of a window lives on its record (gem5's CacheBlk carries
its own status bits, src/mem/cache/cache_blk.hh): the facade keeps no
set that mirrors it. The run is a six-round d=3 memory on the
timing-only device with a preset-latency weak decoder.
"""

import types

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.windows.built_window_models as built_window_models
import decsim.windows.settings as window_settings
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner
import decsim.windows.windowing_schemes as windowing_schemes


def _weak_run():
    memory = program_records.Operation(0, "M", (0,), clifford=True)
    six_rounds = round_policies.FixedRounds(6)
    workload = workload_settings.WorkloadSettings(
        operations=[memory], rounds_policy=six_rounds
    )
    code = code_geometry.SurfaceCodeModel(distance=3)
    qpu = qpu_settings.QpuSettings(code=code, round_period_microseconds=1.0)
    decoder = decoders.PresetLatencyDecoder(2.0)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder, units=1)
    settings = machine_module.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )
    machine = machine_module.Machine.build(settings, 0)
    machine.run()
    return machine


def test_a_committed_window_publishes_the_request_that_decoded_it():
    machine = _weak_run()
    windows = machine.observation.windows.windows
    assert windows
    for key, window in windows.items():
        assert window.committed
        assert window.published_request_key is not None
        assert (
            window.published_request_key.tier is window_records.DecoderTier.WEAK
        )
        assert window.published_request_key.operation_id == key[0]
        assert window.published_request_key.window_id == key[1]


def test_no_window_of_a_weak_run_is_absorbed():
    machine = _weak_run()
    for window in machine.observation.windows.windows.values():
        assert window.is_absorbed is False


def test_a_window_is_final_once_its_request_is_published():
    """A committed window whose request is unpublished still awaits strong."""
    window = window_records.Window(
        operation_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=5, n_rounds=5
    )
    window.committed = True
    assert window.published_request_key is None
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 3
    )
    window.published_request_key = request_key
    assert (
        window.published_request_key.tier is window_records.DecoderTier.STRONG
    )


def test_a_streams_later_window_waits_on_the_previous_ones_boundary():
    """Each later overlapping stream window depends on its predecessor."""
    manager = object.__new__(window_manager_module.WindowManager)
    manager.window_interaction = window_interactions.DefaultWindowInteraction(0)
    manager.planner = _stream_planner()
    manager.tracker = types.SimpleNamespace(is_sealed=lambda _stream_id: False)
    manager.courier = window_boundaries.BoundaryCourier(
        manager.planner, None, manager.window_interaction, None, _no_boundary
    )
    manager.retention = types.SimpleNamespace(
        register_window=lambda _key, _window: None
    )
    stream = program_records.Operation("stream", "stream", (0,))
    manager.planner.register_stream(stream)

    manager._grow_stream("stream", 4, None)

    first, second = manager.planner.windows_of("stream")
    assert (first.commit_lo, first.commit_hi, first.buffer_hi) == (1, 3, 5)
    assert (second.commit_lo, second.commit_hi, second.buffer_hi) == (4, 6, 8)
    assert second.deps == [("stream", 0)]
    assert second.deps_remaining == 1
    assert first.dependents == [("stream", 1)]


def _no_boundary(_key, _is_unblocked) -> None:
    """No delivery lands in this test."""


def _stream_planner() -> window_planner.WindowPlanner:
    plan = window_records.WindowPlan(
        windows={},
        window_count={},
        op_windows={},
        successors={},
        spatial_nodes={},
        rounds_by_operation={},
        code_names={},
        total_windows=0,
        windowed_by_operation={},
        batch_preceding_idle_rounds_by_operation={},
    )
    geometry = types.SimpleNamespace(
        commit_round_count=3, buffer_round_count=2, code_name="surface"
    )
    resolved = types.SimpleNamespace(
        operation_id="stream",
        code_geometry=geometry,
        round_count=9,
        spatial_node_count=17,
    )
    built = built_window_models.BuiltWindowModels()
    models = window_planner.WindowModels(None, lambda _code_name: None, built)
    scheme = windowing_schemes.SlidingWindowScheme()
    return window_planner.WindowPlanner(
        scheme, [resolved], plan, models, planned_operations=()
    )


def _tan_memory_circuit():
    """A 27-round d=3 rotated memory at the reference noise strength."""
    stim = pytest.importorskip("stim")
    probability = 0.003
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=27,
        distance=3,
        after_clifford_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
        before_round_data_depolarization=probability,
    )


def _tan_sandwich_run(unit_count):
    """A 27-round d=3 Stim memory under Tan's sandwich schedule.

    A type-2 seam window fills before the two cores beside it and reads
    both of their boundaries (Tan et al. 2209.09219, supplement S8), so
    it becomes ready first and must wait in its input slot rather than on
    the unit: on one unit a seam that holds the compute while it waits
    for its neighbours would stop the run.
    """
    circuit = _tan_memory_circuit()
    memory = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(27)
    workload = workload_settings.WorkloadSettings(
        operations=[memory], rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(
        distance=3, round_period_microseconds=1.0, device=device
    )
    inner = decoders.PresetLatencyDecoder(5.0)
    decoder = mwpm.PyMatchingDecoder(inner)
    weak_decoder = decoder_settings.DecoderSettings(
        decoder=decoder, units=unit_count
    )
    scheme = windowing_schemes.TanSandwichScheme()
    windows = window_settings.WindowSettings(scheme=scheme)
    settings = machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        windows=windows,
    )
    machine = machine_module.Machine.build(settings, 3)
    result = machine.run()
    return machine, result


def test_a_tan_seam_waits_in_its_slot_and_never_holds_the_only_unit():
    machine, result = _tan_sandwich_run(unit_count=1)
    windows = machine.observation.windows.windows.values()
    undecoded = [window for window in windows if window.t_done is None]
    assert undecoded == []
    assert result.operation_results[0].logical_failure is False


def test_a_round_arriving_after_the_last_window_committed_is_refused():
    """The device is a plug-in, so a round past the plan is its mistake.

    An operation's syndrome RAM is freed when its last window commits,
    so a later round has nothing to land in and no window left to feed;
    the facade stops loudly rather than accounting a round nobody reads
    (window_manager.py, _refuse_unplanned_round).
    """
    machine = _weak_run()
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=0,
        patch_id=0,
        round_index=2,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(0, 2, (fragment,))
    window_manager = machine.window_manager
    with pytest.raises(
        RuntimeError, match="arrived after the op's last window committed"
    ):
        window_manager.accept_window_input(packet)
