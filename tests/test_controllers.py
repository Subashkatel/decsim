#==================================================================
# TESTS FOR CONTROLLER-SIDE ROUND PACKAGING
# arXiv:2511.10633 Sec III.1: the controller aggregates per-qubit readout and
# forwards batched packets to the decoders -- so a round arriving in fragments
# (possibly at different times) must ship as ONE packet after its last fragment.
#==================================================================
from dataclasses import replace

import pytest
import numpy as np

from decsim.config import us
from decsim.controllers import ControllerIngressOverflow, ModularController
from decsim.decoders import PresetLatencyDecoder
from decsim.engine import Engine, SimulationFailed
from decsim.links import (
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkPath,
    LinkQuantityBasis,
    TrafficAttribution,
)
from decsim.message import (
    EndpointRole,
    EndpointState,
    Operation,
    RetainedSyndromeFragment,
    SyndromePacketRoute,
    SyndromePacketRouteKind,
    SyndromePayload,
    SyndromeRoundPacket,
    WINDOW_INPUT_ROUTE,
)
from decsim.payload_store import PayloadStore, SyndromeBufferingConfig
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec
from decsim.window_manager import WindowManager


def _frag(patch, n=2, index=None):
    return SyndromePayload(
        0,
        patch,
        1,
        n_fragments=n,
        fragment_index=patch if index is None else index,
    )


def _actual_edge(*, bandwidth=None):
    capacity = None if bandwidth is None else LinkCapacityConfig(
        float(bandwidth),
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "test capacity",
    )
    return LinkEdgeConfig(
        LinkConfig(0, capacity, "test channel"),
        None,
        "SyndromePayload.size_bits",
    )


def _links(**edges):
    return replace(
        LinkModelConfig.reference_fixed_latency_profile(),
        **edges,
    ).resolve()


def _round_attribution(round_index):
    return TrafficAttribution(0, (0,), None, round_index, round_index)


class _BlockedWindowInputReceiver:
    def accept_window_input(self, packet):
        return False


class _BlockedFeedbackMemoryReceiver:
    def accept_feedback_memory_round(self, source_operation_id):
        raise AssertionError("feedback route is not used by Trace A")


class _RecordingFeedbackMemoryReceiver:
    def __init__(self):
        self.source_operation_ids = []

    def accept_feedback_memory_round(self, source_operation_id):
        self.source_operation_ids.append(source_operation_id)


class _DeliveringWindowInputReceiver:
    def __init__(self, engine, links, deliver):
        self.engine = engine
        self.links = links
        self.deliver = deliver

    def accept_window_input(self, packet):
        sizes = tuple(fragment.size_bits for fragment in packet.fragments)
        bits = sum(sizes) if all(size is not None for size in sizes) else None
        reservation = self.links.reserve(
            LinkPath.CWD, payload_bits=bits, now_ticks=self.engine.now,
            attribution=TrafficAttribution(
                packet.operation_id,
                tuple(fragment.patch_id for fragment in packet.fragments),
                None, packet.round_index, packet.round_index))
        self.engine.schedule(
            reservation.total_delay_ticks,
            lambda: self.deliver(packet), label="typed test endpoint delivery")
        return True


def _controller(engine, deliver, *, links=None, **kwargs):
    resolved_links = links or _links()
    return ModularController(
        engine, links=resolved_links, controller_capacity=None,
        window_input_receiver=_DeliveringWindowInputReceiver(
            engine, resolved_links, deliver),
        feedback_memory_receiver=_BlockedFeedbackMemoryReceiver(), **kwargs)


def test_full_controller_staging_accepts_same_packet_continuation_in_place():
    engine = Engine(verbose=False)
    controller = ModularController(
        engine,
        links=_links(qc=_actual_edge(), cwd=_actual_edge()),
        controller_capacity=1,
        window_input_receiver=_BlockedWindowInputReceiver(),
        feedback_memory_receiver=_BlockedFeedbackMemoryReceiver(),
        log_syndromes=False,
    )
    route = WINDOW_INPUT_ROUTE

    controller.relay_syndrome(_frag("north", index=0), route)
    engine.run()
    partial = controller.staging_snapshot()
    allocated_slot_index = partial.identity_to_slot[0][1]

    assert partial.controller_occupancy == 1
    assert partial.free_slot_indices == ()
    assert partial.partial_identities == (partial.identity_to_slot[0][0],)
    assert partial.packed_wait_identities == ()
    assert partial.draining_identities == ()

    controller.relay_syndrome(_frag("south", index=1), route)
    engine.run()
    packed = controller.staging_snapshot()

    assert packed.controller_occupancy == 1
    assert packed.free_slot_indices == ()
    assert packed.partial_identities == ()
    assert packed.packed_wait_identities == (packed.identity_to_slot[0][0],)
    assert packed.draining_identities == ()
    assert packed.identity_to_slot[0][1] == allocated_slot_index


def test_controller_keeps_feedback_head_draining_until_cwd_delivery():
    engine = Engine(verbose=False)
    feedback_receiver = _RecordingFeedbackMemoryReceiver()
    controller = ModularController(
        engine,
        links=_links(qc=_actual_edge()),
        controller_capacity=1,
        window_input_receiver=_BlockedWindowInputReceiver(),
        feedback_memory_receiver=feedback_receiver,
        log_syndromes=False,
    )
    route = SyndromePacketRoute.feedback_memory_round("source")

    controller.relay_syndrome(_frag("patch", n=1, index=0), route)
    engine.run(until=0)

    draining = controller.staging_snapshot()
    assert draining.controller_occupancy == 1
    assert draining.free_slot_indices == ()
    assert draining.partial_identities == ()
    assert draining.packed_wait_identities == ()
    assert draining.draining_identities == (draining.identity_to_slot[0][0],)
    assert feedback_receiver.source_operation_ids == []

    engine.run()

    released = controller.staging_snapshot()
    assert released.controller_occupancy == 0
    assert released.free_slot_indices == (0,)
    assert released.identity_to_slot == ()
    assert feedback_receiver.source_operation_ids == ["source"]


def test_different_first_fragment_overflow_is_atomic_and_terminal(monkeypatch):
    captured_engines = []
    captured_controllers = []

    class CapturingEngine(Engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured_engines.append(self)

    class CapturingController(ModularController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.snapshots_before_feedback = []
            captured_controllers.append(self)

        def _receive_fragment(self, fragment, fragment_count, route):
            if route.kind is SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND:
                self.snapshots_before_feedback.append(self.staging_snapshot())
            return super()._receive_fragment(fragment, fragment_count, route)

    monkeypatch.setattr("decsim.engine.Engine", CapturingEngine)
    monkeypatch.setattr("decsim.controllers.ModularController", CapturingController)
    first = Operation(
        0, "partial window", (0,), has_successor=True,
        syndrome_fragment_index=0, syndrome_fragment_count=2,
    )
    blocked = Operation(
        1, "feedback successor", (0,), emits_detector_data=False,
        blocked_by=0, predecessors=(0,),
    )
    spec = RunSpec(
        ops=[first, blocked],
        rounds_policy=FixedRounds(1),
        decoder=PresetLatencyDecoder(0.0),
        syndrome_buffering=SyndromeBufferingConfig(
            controller_ingress_packet_slots=1,
            sb0_packet_slots=64,
            sb1_packet_slots=64,
        ),
    )

    with pytest.raises(ControllerIngressOverflow) as raised:
        spec.build(verbose=False)

    assert len(captured_controllers) == len(captured_engines) == 1
    controller = captured_controllers[0]
    engine = captured_engines[0]
    overflow = raised.value
    after = controller.staging_snapshot()
    assert overflow.status == "controller_ingress_overflow"
    assert overflow.route.kind is SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND
    assert overflow.controller_capacity == 1
    assert overflow.controller_occupancy == 1
    assert controller.snapshots_before_feedback == [after]
    assert after.controller_occupancy == 1
    assert len(after.partial_identities) == 1
    assert after.packed_wait_identities == after.draining_identities == ()
    assert spec._build_state == "invalid"
    assert engine._phase == "invalid"
    assert engine._failure_cause is overflow
    queued_event_count = len(engine._event_queue)
    assert queued_event_count > 0
    with pytest.raises(SimulationFailed) as later_run:
        engine.run()
    assert later_run.value.__cause__ is overflow
    assert len(engine._event_queue) == queued_event_count
    assert controller.staging_snapshot() == after


def test_tick_five_endpoint_release_coalesces_one_controller_wake_without_spin():
    engine = Engine(verbose=False)
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)

    class PairingReceiver:
        def __init__(self):
            self.attempt_ticks = []

        def accept_window_input(self, packet):
            self.attempt_ticks.append(engine.now)
            pair = store.prepare_pair(packet, completion_tick=engine.now)
            if pair is None:
                return False
            pair.commit_unpublished()
            pair.publish()
            return True

    class CountingController(ModularController):
        def __init__(self, *args, **kwargs):
            self.arbitration_ticks = []
            super().__init__(*args, **kwargs)

        def _arbitrate(self):
            self.arbitration_ticks.append(self.engine.now)
            return super()._arbitrate()

    receiver = PairingReceiver()
    controller = CountingController(
        engine, links=_links(qc=_actual_edge()), controller_capacity=1,
        window_input_receiver=receiver,
        feedback_memory_receiver=_BlockedFeedbackMemoryReceiver(),
        log_syndromes=False)
    store.connect_capacity_change_receiver(controller)
    owners = {}
    for operation_id in ("A", "B"):
        store.register_op(operation_id)
        key = (operation_id, 1)
        owners[operation_id] = ("consumer", operation_id)
        for role in EndpointRole:
            store.register_owner(role, owners[operation_id], (key,))

    packet_a = SyndromeRoundPacket(
        "A", 1, (RetainedSyndromeFragment.from_payload(
            SyndromePayload("A", "patch", 1)),))
    pair_a = store.prepare_pair(packet_a, completion_tick=0)
    pair_a.commit_unpublished()
    pair_a.publish()
    store.complete_cryo(("A", 1))
    controller.relay_syndrome(
        SyndromePayload(
            "B", "patch", 1, n_fragments=1, fragment_index=0),
        WINDOW_INPUT_ROUTE)
    engine.run(until=0)
    assert receiver.attempt_ticks == [0]
    assert controller.arbitration_ticks == [0]

    def release_a():
        for role in EndpointRole:
            store.release_owner(role, owners["A"])

    engine.schedule(5, release_a, label="release A endpoint owners")
    engine.run()

    assert receiver.attempt_ticks == [0, 5]
    assert controller.arbitration_ticks == [0, 5]
    assert controller.staging_snapshot().controller_occupancy == 0
    assert dict(store.endpoint_snapshot(
        EndpointRole.SB0).states_by_identity)[("B", 1)] is EndpointState.RESIDENT
    assert dict(store.endpoint_snapshot(EndpointRole.SB1).states_by_identity)[
        ("B", 1)] is EndpointState.CRYO_IN_FLIGHT
    assert not engine._event_queue


def test_run_spec_finite_endpoint_pressure_retries_after_window_release(
    monkeypatch,
):
    blocked_ticks = []
    accept = WindowManager.accept_window_input

    def record_pressure(runtime, packet):
        accepted = accept(runtime, packet)
        if not accepted:
            blocked_ticks.append(runtime.engine.now)
        return accepted

    monkeypatch.setattr(WindowManager, "accept_window_input", record_pressure)
    completed = RunSpec(
        ops=[Operation(0, "finite endpoint memory", (0,))],
        d=3,
        rounds_policy=FixedRounds(9),
        decoder=PresetLatencyDecoder(100.0),
        syndrome_buffering=SyndromeBufferingConfig(
            controller_ingress_packet_slots=9,
            sb0_packet_slots=6,
            sb1_packet_slots=1,
        ),
    ).build(verbose=False)

    store = completed.window_manager.store
    assert blocked_ticks
    assert completed.controller.staging_snapshot().controller_occupancy == 0
    assert store.backing_identities == ()
    assert all(not store.endpoint_snapshot(role).resident_identities
               for role in EndpointRole)
    assert not completed.engine._event_queue


def test_staggered_fragments_ship_as_one_packet_after_the_last():
    eng = Engine(verbose=False)
    arrivals = []
    deliver = lambda packet: arrivals.append((
        eng.now,
        tuple(fragment.patch_id for fragment in packet.fragments),
    ))
    ctrl = _controller(eng, deliver, log_syndromes=False)
    # the device emits the round in two chunks, 0.4 us apart (staggered readout)
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), WINDOW_INPUT_ROUTE))
    eng.schedule(us(0.4), lambda: ctrl.relay_syndrome(
        _frag(1), WINDOW_INPUT_ROUTE))
    eng.run()
    # both fragments arrive TOGETHER, one t_qc + t_cd after the LAST chunk left the chip
    expected = us(0.4) + us(0.15) + us(2.0)
    assert arrivals == [(expected, (0, 1))]


def test_packaging_cost_is_priced_per_packet():
    eng = Engine(verbose=False)
    arrivals = []
    deliver = lambda packet: arrivals.append(eng.now)
    ctrl = _controller(
        eng, deliver, log_syndromes=False, t_pack=us(0.3))
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), WINDOW_INPUT_ROUTE))
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(1), WINDOW_INPUT_ROUTE))
    eng.run()
    assert arrivals == [us(0.15) + us(0.3) + us(2.0)]


def test_same_patch_fragments_use_declared_order_not_arrival_order():
    engine = Engine(verbose=False)
    delivered = []
    deliver = delivered.append
    controller = _controller(engine, deliver, log_syndromes=False)
    terminal = SyndromePayload(
        7, "patch", 3, bits=[1], n_fragments=2,
        fragment_index=1, size_bits=3,
    )
    ordinary = SyndromePayload(
        7, "patch", 3, bits=[0, 1], n_fragments=2,
        fragment_index=0, size_bits=2,
    )

    controller.relay_syndrome(terminal, WINDOW_INPUT_ROUTE)
    controller.relay_syndrome(ordinary, WINDOW_INPUT_ROUTE)
    engine.run()

    assert len(delivered) == 1
    assert len(delivered[0].fragments) == 1
    fragment = delivered[0].fragments[0]
    assert fragment.patch_id == "patch"
    assert fragment.bits == (0, 1, 1)
    assert fragment.size_bits == 5
    assert fragment.fragment_index == 0


@pytest.mark.parametrize(
    ("count", "index"),
    [(True, 0), (2, True), (2, 2), (2, -1)],
)
def test_invalid_fragment_carrier_is_rejected_at_construction(count, index):
    with pytest.raises((TypeError, ValueError)):
        SyndromePayload(0, 0, 1, n_fragments=count, fragment_index=index)


def test_whole_round_payloads_take_the_original_path():
    # n_fragments=1 (every default device): no buffering, no t_pack -- two plain hops
    eng = Engine(verbose=False)
    arrivals = []
    ctrl = _controller(
        eng, lambda packet: arrivals.append(eng.now),
        log_syndromes=False, t_pack=us(9.9))
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0, n=1),
                                                WINDOW_INPUT_ROUTE))
    eng.run()
    assert arrivals == [us(0.15) + us(2.0)]


def test_controller_copies_mixed_identity_fragments_into_one_packet():
    eng = Engine(verbose=False)
    delivered = []
    deliver = lambda packet: delivered.append(packet)
    ctrl = _controller(eng, deliver, log_syndromes=False)
    identities = (2, "2", (), (2, "north"), ("gross", (5, "north")))
    mutable_bits = [[index & 1, (index + 1) & 1]
                    for index in range(len(identities))]

    for fragment_index, (patch_id, bits) in enumerate(
        zip(identities, mutable_bits)
    ):
        ctrl.relay_syndrome(
            SyndromePayload(
                operation_id=("operation", 3),
                patch_id=patch_id,
                round_index=1,
                    n_fragments=len(identities),
                    fragment_index=fragment_index,
                bits=bits,
                code="surface",
                size_bits=2,
            ),
            WINDOW_INPUT_ROUTE,
        )
    for bits in mutable_bits:
        bits[:] = [1, 1]

    eng.run()

    assert len(delivered) == 1
    packet = delivered[0]
    assert type(packet) is SyndromeRoundPacket
    assert packet.operation_id == ("operation", 3)
    assert packet.round_index == 1
    assert tuple(fragment.patch_id for fragment in packet.fragments) == identities
    assert tuple(fragment.bits for fragment in packet.fragments) == (
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )


def test_controller_freezes_fragment_count_before_transport():
    engine = Engine(verbose=False)
    delivered = []
    controller = _controller(engine, delivered.append, log_syndromes=False)
    payload = SyndromePayload(
        operation_id=0,
        patch_id="north",
        round_index=1,
        n_fragments=1,
    )

    controller.relay_syndrome(payload, WINDOW_INPUT_ROUTE)
    payload.n_fragments = 2
    engine.run()

    assert len(delivered) == 1
    assert tuple(
        fragment.patch_id for fragment in delivered[0].fragments
    ) == ("north",)
    assert controller.staging_snapshot().controller_occupancy == 0


def test_controller_rejects_invalid_fragment_before_packet_state_mutates():
    eng = Engine(verbose=False)
    ctrl = _controller(eng, lambda packet: None, log_syndromes=False)
    payload = SyndromePayload(
        operation_id=0,
        patch_id=0,
        round_index=True,
        n_fragments=1,
    )

    with pytest.raises(TypeError, match="round_index"):
        ctrl.relay_syndrome(payload, WINDOW_INPUT_ROUTE)

    assert ctrl.staging_snapshot().controller_occupancy == 0


class _IntegerSubclass(int):
    pass


class _StringSubclass(str):
    pass


class _HostileIdentity:
    def __eq__(self, other):
        raise AssertionError("hostile equality must not run")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_id", True),
        ("operation_id", "\ud800"),
        ("operation_id", (0, _HostileIdentity())),
        ("patch_id", False),
        ("patch_id", ("north", "\udfff")),
        ("round_index", True),
        ("round_index", 1.0),
        ("round_index", np.int64(1)),
        ("round_index", _IntegerSubclass(1)),
        ("round_index", 0),
        ("round_index", -1),
        ("n_fragments", True),
        ("n_fragments", 1.0),
        ("n_fragments", np.int64(1)),
        ("n_fragments", _IntegerSubclass(1)),
        ("n_fragments", 0),
        ("n_fragments", -1),
        ("size_bits", True),
        ("size_bits", 1.0),
        ("size_bits", np.int64(1)),
        ("size_bits", _IntegerSubclass(1)),
        ("size_bits", -1),
        ("code", ""),
        ("code", "\ud800"),
        ("code", 3),
        ("code", _StringSubclass("surface")),
        ("bits", np.array([[True]], dtype=bool)),
        ("bits", np.array([1], dtype=np.uint8)),
        ("bits", [0, 2]),
        ("bits", b"\x00"),
    ],
)
def test_controller_fragment_ingress_is_exact_and_non_mutating(field, value):
    engine = Engine(verbose=False)
    controller = _controller(
        engine, lambda packet: None, log_syndromes=False)
    payload = SyndromePayload(0, 0, 1)
    setattr(payload, field, value)

    with pytest.raises((TypeError, ValueError)):
        controller.relay_syndrome(payload, WINDOW_INPUT_ROUTE)

    assert controller.staging_snapshot().controller_occupancy == 0
    assert engine._event_queue == []


def test_controller_rejects_duplicate_count_sink_and_late_fragments():
    engine = Engine(verbose=False)
    delivered = []
    deliver = delivered.append
    controller = _controller(
        engine, deliver,
        links=_links(qc=_actual_edge(), cwd=_actual_edge()),
        log_syndromes=False)

    controller.relay_syndrome(_frag("north", index=0), WINDOW_INPUT_ROUTE)
    engine.run()
    partial = controller.staging_snapshot()
    assert len(partial.partial_identities) == 1
    assert partial.controller_occupancy == 1

    controller.relay_syndrome(_frag("north", index=0), WINDOW_INPUT_ROUTE)
    with pytest.raises(ValueError, match="duplicate"):
        engine.run()
    assert controller.staging_snapshot() == partial

    controller.relay_syndrome(
            SyndromePayload(0, "south", 1, n_fragments=3, fragment_index=1),
        WINDOW_INPUT_ROUTE,
    )
    with pytest.raises(ValueError, match="same count"):
        engine.run()
    assert controller.staging_snapshot() == partial

    controller.relay_syndrome(_frag("south", index=1), WINDOW_INPUT_ROUTE)
    engine.run()
    assert len(delivered) == 1
    assert controller.staging_snapshot().controller_occupancy == 0

    controller.relay_syndrome(_frag("late", index=0), WINDOW_INPUT_ROUTE)
    with pytest.raises(ValueError, match="already completed"):
        engine.run()
    assert len(delivered) == 1


#------------------------------------------------------------------
# BANDWIDTH AND SERIALIZATION: the controller must hand each hop the
# payload's size_bits (and, for shared buses, the current time) or the
# link prices every transfer at zero.
#------------------------------------------------------------------

def test_bandwidth_links_price_the_syndrome_bits():
    # 1000 bits over 1000 bits/us (qc) then 2000 bits/us (cd): 1.5 us total
    eng = Engine(verbose=False)
    links = _links(qc=_actual_edge(bandwidth=1000),
                   cwd=_actual_edge(bandwidth=2000))
    arrivals = []
    ctrl = _controller(
        eng, lambda packet: arrivals.append(eng.now),
        links=links, log_syndromes=False)
    payload = SyndromePayload(0, 0, 1, size_bits=1000)
    eng.schedule(0, lambda: ctrl.relay_syndrome(
        payload, WINDOW_INPUT_ROUTE))
    eng.run()
    assert arrivals == [us(1.5)]


def test_serialized_chip_link_queues_concurrent_rounds():
    # two 1000-bit rounds on a shared 1000 bits/us qc bus: 1 us then 2 us
    eng = Engine(verbose=False)
    links = _links(qc=_actual_edge(bandwidth=1000),
                   cwd=_actual_edge())
    arrivals = []
    ctrl = _controller(
        eng, lambda packet: arrivals.append(eng.now),
        links=links, log_syndromes=False)
    for round_index in (1, 2):
        payload = SyndromePayload(0, 0, round_index, size_bits=1000)
        eng.schedule(0, lambda p=payload: ctrl.relay_syndrome(
            p, WINDOW_INPUT_ROUTE))
    eng.run()
    assert arrivals == [us(1.0), us(2.0)]


def test_packed_fragments_are_priced_by_their_total_bits():
    # 400 + 600 bit fragments = one 1000-bit packet over 1000 bits/us (cd)
    eng = Engine(verbose=False)
    links = _links(qc=_actual_edge(),
                   cwd=_actual_edge(bandwidth=1000))
    arrivals = []
    deliver = lambda packet: arrivals.append(eng.now)
    ctrl = _controller(eng, deliver, links=links, log_syndromes=False)
    for patch, bits in ((0, 400), (1, 600)):
        payload = SyndromePayload(
            0, patch, 1, n_fragments=2, fragment_index=patch, size_bits=bits
        )
        eng.schedule(0, lambda p=payload: ctrl.relay_syndrome(
            p, WINDOW_INPUT_ROUTE))
    eng.run()
    assert arrivals == [us(1.0)]
