"""The boundary courier's laws through its deliveries.

Every send bumps the source version; a receiver accepts only the latest
(Skoric et al. 2209.08552: the artificial defects travel with the commit
that produced them; qLDPC carries the latest net_error). A version-1
boundary still in flight when the source is decoded again lands as a
no-op; version 2 releases the dependency exactly once, and a repeated
delivery of the current version releases nothing.
"""

import types

import decsim.engine as engine_module
import decsim.links.fabric as fabric
import decsim.links.link_profiles as link_profiles
import decsim.message as message
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager


def test_a_stale_delivery_is_ignored_and_the_edge_releases_once():
    engine = engine_module.Engine(verbose=False)
    operation = message.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,)
    )
    source = message.Window(
        op_id=1,
        k=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        n_rounds=5,
        dependents=[(1, 1)],
    )
    dependent = message.Window(
        op_id=1,
        k=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=8,
        n_rounds=8,
        deps=[(1, 0)],
        deps_remaining=1,
    )
    windows = {(1, 0): source, (1, 1): dependent}
    checks = []

    def check_window(key) -> None:
        checks.append((engine.now, key))

    def window_infos() -> dict:
        infos = {}
        for window_key, window in windows.items():
            infos[window_key] = message.WindowInfo.from_window(window)
        return infos

    planner = types.SimpleNamespace(
        windows_by_key=windows,
        model_by_window={},
        round_count_of=lambda _operation_id: 20,
    )
    profile = link_profiles.logical_reference_profile()
    links = fabric.LinkFabric(profile, engine)
    interaction = window_interactions.DefaultWindowInteraction()
    manager = types.SimpleNamespace(
        engine=engine,
        links=links,
        planner=planner,
        window_interaction=interaction,
        release_service=None,
        check_window=check_window,
        _window_attribution=window_manager.WindowManager._window_attribution,
    )
    manager._window_infos = window_infos
    courier = window_boundaries.BoundaryCourier(manager)
    key = message.DecoderRequestKey(1, 0, message.DecoderTier.WEAK, 0)
    boundary_v1 = {4: [1, 0, 0]}
    boundary_v2 = {4: [0, 1, 0]}

    def send_v2():
        courier.invalidate(source)
        courier.send(source, operation, boundary_v2, source_request_key=key)

    engine.schedule(
        0,
        lambda: courier.send(
            source, operation, boundary_v1, source_request_key=key
        ),
    )
    engine.schedule(1, send_v2)  # decoded again before version 1 could land
    engine.run()

    assert dependent.deps_remaining == 0
    assert dict(dependent.boundary_in) == {4: [0, 1, 0]}
    assert len(checks) == 2
    courier._receive_boundary((1, 1), 1, boundary_v2, (1, 0), 2, 2)
    assert dependent.deps_remaining == 0
    assert dict(dependent.boundary_in) == {4: [0, 1, 0]}
