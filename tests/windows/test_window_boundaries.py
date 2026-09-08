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
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_transfers as window_transfers


class _EagerPolicy:
    def on_commit(self, _window, *, final) -> bool:
        del final
        return True


_EAGER = _EagerPolicy()


def test_a_stale_delivery_is_ignored_and_the_edge_releases_once():
    engine = engine_module.Engine()
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,)
    )
    source = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=5,
        dependents=[(1, 1)],
    )
    dependent = window_records.Window(
        operation_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=8,
        round_count=8,
        deps=[(1, 0)],
        deps_remaining=1,
    )
    windows = {(1, 0): source, (1, 1): dependent}
    checks = []

    def on_boundary_received(key, _is_unblocked) -> None:
        checks.append((engine.now, key))

    planner = types.SimpleNamespace(
        windows_by_key=windows,
        model_by_window={},
        round_count_of=lambda _operation_id: 20,
    )
    profile = link_profiles.logical_reference_profile()
    links = fabric.LinkFabric(profile, engine)
    boundary_payload = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(
        0, boundary_payload
    )
    transfers = window_transfers.WindowTransfers(engine, links)
    courier = window_boundaries.BoundaryCourier(
        planner, transfers, interaction, _EAGER, on_boundary_received
    )
    key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
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
