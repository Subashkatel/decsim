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
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.round_retention as round_retention
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
        # d*d-1 checks of one d=3 patch
        syndrome_bits_per_round_of=lambda _operation_id: 8,
    )
    profile = link_profiles.logical_reference_profile()
    links = fabric.LinkFabric(profile, engine)
    boundary_payload = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(
        0, boundary_payload
    )
    transfers = window_transfers.WindowTransfers(engine)
    transfers.link = links
    courier = window_boundaries.BoundaryCourier(window_records.DecoderTier.WEAK)
    courier.planner = planner
    courier.transfers = transfers
    courier.retention = _retention_of(20)
    courier.interaction = interaction
    courier.boundary_policy = _EAGER
    courier.windows = types.SimpleNamespace(
        accept_boundary=on_boundary_received
    )
    key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    boundary_v1 = {4: [1, 0, 0]}
    boundary_v2 = {4: [0, 1, 0]}

    engine.schedule(
        0,
        lambda: courier.send(
            source, operation, boundary_v1, source_request_key=key
        ),
    )
    engine.schedule(  # decoded again before version 1 could land
        1,
        lambda: courier.send(
            source, operation, boundary_v2, source_request_key=key
        ),
    )
    engine.run()

    assert dependent.deps_remaining == 0
    assert dict(dependent.boundary_in) == {4: [0, 1, 0]}
    assert len(checks) == 2
    courier._receive_boundary((1, 1), 1, boundary_v2, (1, 0), 2, 2)
    assert dependent.deps_remaining == 0
    assert dict(dependent.boundary_in) == {4: [0, 1, 0]}
    courier._receive_boundary((1, 1), 1, boundary_v1, (1, 0), 1, 1)
    assert dependent.deps_remaining == 0
    assert dict(dependent.boundary_in) == {4: [0, 1, 0]}


def _retention_of(round_count: int) -> round_retention.RoundRetention:
    """The retention of operation 1, round_count rounds, then operation 2."""
    retention = round_retention.RoundRetention(
        is_strong_context_retained=False,
        primary_tier=window_records.DecoderTier.WEAK,
    )
    retention.tracker = types.SimpleNamespace(
        effective_round_count_for_window=lambda _operation_id, _window: (
            round_count
        )
    )
    retention.planner = types.SimpleNamespace(successors_by_operation={1: [2]})
    return retention


class _RecordingTransfers:
    """The window transfers, with every boundary send recorded."""

    def __init__(self, transfers) -> None:
        self.transfers = transfers
        self.boundary_sends = []

    def send_boundary(
        self, path, attribution, payload_bits, on_delivered
    ) -> None:
        """Record the hand-off, then send it over the real fabric."""
        self.boundary_sends.append((path, attribution, payload_bits))
        self.transfers.send_boundary(
            path, attribution, payload_bits, on_delivered
        )


class _LandingTicks:
    """The facade, noting the tick each boundary lands."""

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.ticks = []

    def accept_boundary(self, _window_key: tuple, _is_unblocked: bool) -> None:
        """One boundary landed now."""
        self.ticks.append(self.engine.now)


def _reference_latency_ticks(path_name: str) -> int:
    """The reference card's latency on one hop."""
    profile = link_profiles.logical_reference_profile()
    path = getattr(profile, path_name)
    return path.channel.propagation_latency_ticks


def _pinned_courier():
    """A courier whose window (1,0) has committed, and its strong reader.

    The strong window re-decodes rounds 4-6 with its past face pinned on
    window (1,0), so it reads from round 4 and its seam layer is round
    4: eight detectors, d*d-1 of a d=3 bulk layer. Operation 1 has six
    rounds and operation 2 follows it, so the buffer past round 6 reads
    operation 2's first rounds.
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
        round_count_of=lambda _operation_id: 6,
        # d*d-1 checks of one d=3 patch
        syndrome_bits_per_round_of=lambda _operation_id: 8,
    )
    profile = link_profiles.logical_reference_profile()
    links = fabric.LinkFabric(profile, engine)
    boundary_payload = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(
        0, boundary_payload
    )
    transfers = window_transfers.WindowTransfers(engine)
    transfers.link = links
    recording = _RecordingTransfers(transfers)
    courier = window_boundaries.BoundaryCourier(window_records.DecoderTier.WEAK)
    courier.planner = planner
    courier.transfers = recording
    courier.retention = _retention_of(6)
    courier.interaction = interaction
    courier.boundary_policy = _EAGER
    courier.windows = _LandingTicks(engine)
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
    that layer of the strong window: eight detectors, one bit each,
    behind the request's name the cable to the host carries.
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
    _path, attribution, payload_bits = pinned_send
    assert payload_bits == 8 + window_records.REQUEST_KEY_WIRE_BITS
    assert attribution.relation.source_window_key == (1, 0)
    assert attribution.relation.destination_window_key == (1, 1)
    assert attribution.relation.source_request_key == request_key
    # the transfer is the strong window's own: its index and its reads,
    # the rounds past operation 1's sixth under operation 2
    assert attribution.window_id == 1
    assert (attribution.first_round, attribution.last_round) == (4, 9)
    assert attribution.round_keys == (
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 1),
        (2, 2),
        (2, 3),
    )


def test_a_weak_commit_reaches_a_strong_window_over_the_cable():
    """The chip commits the boundary and the host decodes the strong window.

    The weak hand-on stays on the chip, but the pinned face crosses to
    the host on the escalation's hop, so the strong decode, which starts
    once its boundary is determined (Toshio et al. 2510.25222 lines
    1248-1250), waits at least that hop's latency, not one chip cycle.
    """
    engine, operation, source, courier, recording = _pinned_courier()
    weak_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    residual = window_records.DependencyResidual(detector_ids=(24,))
    courier.send(source, operation, residual, source_request_key=weak_key)
    engine.run()
    pinned_ticks = engine.now
    strong_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 3
    )
    strong_window = _strong_window_of_the_pin()
    model = _strong_model()
    courier.pin_strong_face((1, 0), strong_window, model, operation, strong_key)
    engine.run()
    hand_on_path, _attribution, _bits = recording.boundary_sends[0]
    pin_path, _attribution, _bits = recording.boundary_sends[1]
    pin_ticks = courier.windows.ticks[1] - pinned_ticks
    cable_ticks = _reference_latency_ticks("weak_decoder_to_strong_decoder")
    assert pin_ticks >= cable_ticks
    assert hand_on_path is transfer_records.LinkPath.DECODER_TO_DECODER
    assert pin_path is transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER


def test_a_strong_commit_reaches_the_weak_window_after_it_over_the_cable():
    """A strong decode's boundary leaves the host for the chip.

    A held boundary ships once the strong answer is final, and the
    window after it decodes on the chip, so the message takes the strong
    answer's hop down and lands no sooner than its latency. The window
    after it has no model, so its seam is one round of the code card's
    checks, eight at d=3, behind the request's name.
    """
    engine, operation, source, courier, recording = _pinned_courier()
    strong_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 2
    )
    residual = window_records.DependencyResidual(detector_ids=(24,))
    courier.send(source, operation, residual, source_request_key=strong_key)
    engine.run()
    path, _attribution, payload_bits = recording.boundary_sends[0]
    cable_ticks = _reference_latency_ticks("strong_decoder_to_weak_decoder")
    assert courier.windows.ticks[0] >= cable_ticks
    assert path is transfer_records.LinkPath.STRONG_DECODER_TO_WEAK_DECODER
    assert payload_bits == 8 + window_records.REQUEST_KEY_WIRE_BITS


def test_a_strong_commit_reaches_a_strong_window_in_the_hosts_memory():
    """Both decodes run on the host, so the face crosses no cable.

    The strong window reads it out of the host's own memory, as it reads
    its rounds, and that read carries no request's name: the seam alone,
    eight detectors of one bit.
    """
    engine, operation, source, courier, recording = _pinned_courier()
    # the weak window after it has a model, so the cable prices the
    # message by its seam
    courier.planner.models.model_by_window[(1, 1)] = _strong_model()
    strong_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 2
    )
    residual = window_records.DependencyResidual(detector_ids=(24,))
    courier.send(source, operation, residual, source_request_key=strong_key)
    engine.run()
    next_strong_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 3
    )
    strong_window = _strong_window_of_the_pin()
    model = _strong_model()
    courier.pin_strong_face(
        (1, 0), strong_window, model, operation, next_strong_key
    )
    engine.run()
    path, _attribution, payload_bits = recording.boundary_sends[1]
    assert path is transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER
    assert payload_bits == 8


def test_a_face_pinned_on_its_own_window_folds_only_the_crossing_commit():
    """At a back-to-back seam the near face is the window's own commit.

    The strong redo decodes that window's rounds again, so the only part
    of its weak commit it may take is the faults crossing behind its
    first round; folding the whole commit would count the rounds it
    decodes again twice (Bombin et al. 2303.04846 lines 775-788, Toshio
    et al. 2510.25222 lines 1248-1250).
    """
    engine, operation, _source, courier, _recording = _pinned_courier()
    weak_window = courier.planner.windows_by_key[(1, 1)]
    request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.WEAK, 0
    )
    # the whole commit flips detectors on rounds 4, 5 and 6; of those,
    # detector 24 comes of a fault that also flips detector 17 on round
    # 3, the round before the window's first
    whole = window_records.DependencyResidual(detector_ids=(24, 33, 41))
    crossing_residual = window_records.DependencyResidual(detector_ids=(17, 24))
    weak_window.crossing_commit = window_records.CrossingCommit(
        crossing_residual, (1,)
    )
    courier.send(weak_window, operation, whole, source_request_key=request_key)
    engine.run()
    strong_window = _strong_window_of_the_pin()
    model = _strong_model()
    strong_request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 3
    )
    courier.pin_strong_face(
        (1, 1), strong_window, model, operation, strong_request_key
    )
    engine.run()
    assert dict(strong_window.boundary_in) == {4: [1]}


def test_a_face_pinned_on_a_window_that_has_not_committed_is_refused():
    engine, operation, _source, courier, _recording = _pinned_courier()
    del engine
    strong_window = _strong_window_of_the_pin()
    model = _strong_model()
    strong_request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 3
    )
    with pytest.raises(RuntimeError, match="which has not committed"):
        courier.pin_strong_face(
            (1, 0), strong_window, model, operation, strong_request_key
        )
