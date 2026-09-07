"""The windows facade's laws through its public surface.

The status of a window lives on its record (gem5's CacheBlk carries
its own status bits, src/mem/cache/cache_blk.hh): the facade keeps no
set that mirrors it. The run is a six-round d=3 memory on the
timing-only device with a preset-latency weak decoder.
"""

import types

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.message as message
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.windows as window_records
import decsim.windows.built_window_models as built_window_models
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner
import decsim.windows.windowing_schemes as windowing_schemes


def _weak_run():
    memory = message.Operation(0, "M", (0,), clifford=True)
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
        op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=5, n_rounds=5
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
    manager.window_interaction = window_interactions.DefaultWindowInteraction()
    manager.planner = _stream_planner()
    manager.tracker = types.SimpleNamespace(is_sealed=lambda _stream_id: False)
    manager.courier = window_boundaries.BoundaryCourier(
        manager.planner, None, manager.window_interaction, None, _no_boundary
    )
    manager.retention = types.SimpleNamespace(
        register_window=lambda _key, _window: None
    )
    stream = message.Operation("stream", "stream", (0,))
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
