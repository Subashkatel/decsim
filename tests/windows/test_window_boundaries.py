"""The boundary courier's laws through its deliveries.

Every send bumps the source version; a receiver accepts only the latest
(Skoric et al. 2209.08552: the artificial defects travel with the commit
that produced them; qLDPC carries the latest net_error). A version-1
boundary still in flight when the source is decoded again lands as a
no-op; version 2 releases the dependency exactly once, and a repeated
delivery of the current version releases nothing.
"""

import types

import pytest

import decsim.engine as engine_module
import decsim.links.fabric as fabric
import decsim.links.link_profiles as link_profiles
import decsim.links.window_transfers as window_transfers
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions


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

    no_models = types.SimpleNamespace(model_by_window={})
    planner = types.SimpleNamespace(
        windows_by_key=windows,
        models=no_models,
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


class _RecordingTransfers:
    """The window transfers, with every boundary send recorded."""

    def __init__(self, transfers) -> None:
        self.transfers = transfers
        self.boundary_sends = []

    def send_boundary(self, attribution, payload_bits, on_delivered) -> None:
        """Record the hand-off, then send it over the real fabric."""
        self.boundary_sends.append((attribution, payload_bits))
        self.transfers.send_boundary(attribution, payload_bits, on_delivered)


def _pinned_courier():
    """A courier whose window (1,0) has committed, and its strong reader.

    The strong window re-decodes rounds 4-6 with its past face pinned on
    window (1,0), so it reads from round 4 and its seam layer is round
    4: eight detectors, d*d-1 of a d=3 bulk layer.
    """
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
    no_models = types.SimpleNamespace(model_by_window={})
    planner = types.SimpleNamespace(
        windows_by_key=windows,
        models=no_models,
        round_count_of=lambda _operation_id: 20,
    )
    profile = link_profiles.logical_reference_profile()
    links = fabric.LinkFabric(profile, engine)
    boundary_payload = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(
        0, boundary_payload
    )
    transfers = window_transfers.WindowTransfers(engine, links)
    recording = _RecordingTransfers(transfers)
    courier = window_boundaries.BoundaryCourier(
        planner, recording, interaction, _EAGER, lambda _key, _ready: None
    )
    return engine, operation, source, courier, recording


def _strong_window_of_the_pin() -> window_records.Window:
    """The near-pinned strong redo of window (1,1): reads 4-9, commits 4-6."""
    return window_records.Window(
        operation_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=9,
        round_count=6,
        buffer_lo=4,
    )


def _strong_model():
    """Eight detectors on every round layer 1 to 9, positions 0 to 7.

    The window's own rows are the detectors of the rounds it reads, 4 to
    9; the earlier layers are placed because its faults flip them, which
    is what a model carries for a neighbour's input.
    """
    positions = {}
    rows = []
    for detector_id in range(72):
        round_index = detector_id // 8 + 1
        position = detector_id % 8
        positions[detector_id] = (round_index, position)
        if round_index >= 4:
            rows.append(detector_id)
    return types.SimpleNamespace(
        defect_positions=positions, detector_ids=tuple(rows)
    )


def test_a_pinned_face_carries_the_neighbours_committed_seam():
    """Bombin 2303.04846 lines 775-788, on the strong window's own model.

    The residual detectors that lie in the strong window's model land on
    the one layer where the two windows meet, its oldest read round
    (Tan 2209.09219 lines 943-946), and the message is priced against
    that layer of the strong window: eight detectors, one bit each.
    """
    engine, operation, source, courier, recording = _pinned_courier()
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    # detectors 24 and 26 sit on round 4 positions 0 and 2; detector 17
    # sits on round 3, which is not a row of the strong window
    residual = window_records.DependencyResidual(detector_ids=(17, 24, 26))
    courier.send(source, operation, residual, source_request_key=request_key)
    engine.run()
    strong_window = _strong_window_of_the_pin()
    model = _strong_model()
    strong_request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 3
    )
    courier.pin_strong_face(
        (1, 0), strong_window, model, operation, strong_request_key
    )
    engine.run()
    assert dict(strong_window.boundary_in) == {4: [1, 0, 1]}
    pinned_send = recording.boundary_sends[-1]
    attribution, payload_bits = pinned_send
    assert payload_bits == 8
    assert attribution.relation.source_window_key == (1, 0)
    assert attribution.relation.destination_window_key == (1, 1)
    assert attribution.relation.source_request_key == request_key
    # the transfer is the strong window's own: its index and its reads
    assert attribution.window_id == 1
    assert (attribution.first_round, attribution.last_round) == (4, 9)


def test_a_face_pinned_on_a_window_that_has_not_committed_is_refused():
    engine, operation, _source, courier, _recording = _pinned_courier()
    del engine
    strong_window = _strong_window_of_the_pin()
    model = _strong_model()
    strong_request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 3
    )
    with pytest.raises(RuntimeError) as refusal:
        courier.pin_strong_face(
            (1, 0), strong_window, model, operation, strong_request_key
        )
    assert "has not committed" in str(refusal.value)
