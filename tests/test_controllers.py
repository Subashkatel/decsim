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
from decsim.controllers import ModularController
from decsim.engine import Engine
from decsim.links import (
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkPath,
    LinkQuantityBasis,
    TrafficAttribution,
)
from decsim.message import SyndromePayload, SyndromeRoundPacket


def _frag(patch, n=2):
    return SyndromePayload(0, patch, 1, n_fragments=n)


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


def _window_attribution(round_index):
    return TrafficAttribution(0, (0,), 0, round_index, round_index)


def test_staggered_fragments_ship_as_one_packet_after_the_last():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False)
    arrivals = []
    deliver = lambda packet: arrivals.append((
        eng.now,
        tuple(fragment.patch_id for fragment in packet.fragments),
    ))
    # the device emits the round in two chunks, 0.4 us apart (staggered readout)
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), deliver))
    eng.schedule(us(0.4), lambda: ctrl.relay_syndrome(_frag(1), deliver))
    eng.run()
    # both fragments arrive TOGETHER, one t_qc + t_cd after the LAST chunk left the chip
    expected = us(0.4) + us(0.15) + us(2.0)
    assert arrivals == [(expected, (0, 1))]


def test_packaging_cost_is_priced_per_packet():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False, t_pack=us(0.3))
    arrivals = []
    deliver = lambda packet: arrivals.append(eng.now)
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), deliver))
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(1), deliver))
    eng.run()
    assert arrivals == [us(0.15) + us(0.3) + us(2.0)]


def test_whole_round_payloads_take_the_original_path():
    # n_fragments=1 (every default device): no buffering, no t_pack -- two plain hops
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False, t_pack=us(9.9))
    arrivals = []
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0, n=1),
                                                lambda packet: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(0.15) + us(2.0)]


def test_controller_copies_mixed_identity_fragments_into_one_packet():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False)
    delivered = []
    deliver = lambda packet: delivered.append(packet)
    identities = (2, "2", (), (2, "north"), ("gross", (5, "north")))
    mutable_bits = [[index & 1, (index + 1) & 1]
                    for index in range(len(identities))]

    for patch_id, bits in zip(identities, mutable_bits):
        ctrl.relay_syndrome(
            SyndromePayload(
                operation_id=("operation", 3),
                patch_id=patch_id,
                round_index=1,
                n_fragments=len(identities),
                bits=bits,
                code="surface",
                size_bits=2,
            ),
            deliver,
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
    controller = ModularController(engine, log_syndromes=False)
    delivered = []
    payload = SyndromePayload(
        operation_id=0,
        patch_id="north",
        round_index=1,
        n_fragments=1,
    )

    controller.relay_syndrome(payload, delivered.append)
    payload.n_fragments = 2
    engine.run()

    assert len(delivered) == 1
    assert tuple(
        fragment.patch_id for fragment in delivered[0].fragments
    ) == ("north",)
    assert controller._pending == {}
    assert controller._completed_rounds == {(0, 1)}


def test_controller_rejects_invalid_fragment_before_packet_state_mutates():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False)
    payload = SyndromePayload(
        operation_id=0,
        patch_id=0,
        round_index=True,
        n_fragments=1,
    )

    with pytest.raises(TypeError, match="round_index"):
        ctrl.relay_syndrome(payload, lambda packet: None)

    assert ctrl._pending == {}
    assert ctrl._completed_rounds == set()


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
    controller = ModularController(engine, log_syndromes=False)
    payload = SyndromePayload(0, 0, 1)
    setattr(payload, field, value)

    with pytest.raises((TypeError, ValueError)):
        controller.relay_syndrome(payload, lambda packet: None)

    assert controller._pending == {}
    assert controller._completed_rounds == set()
    assert engine._event_queue == []


def test_controller_rejects_duplicate_count_sink_and_late_fragments():
    engine = Engine(verbose=False)
    controller = ModularController(engine, links=_links(
                                       qc=_actual_edge(),
                                       cwd=_actual_edge()),
                                   log_syndromes=False)
    delivered = []
    deliver = delivered.append

    controller.relay_syndrome(_frag("north"), deliver)
    engine.run()
    assert len(controller._pending[(0, 1)].fragments) == 1

    controller.relay_syndrome(_frag("north"), deliver)
    with pytest.raises(ValueError, match="duplicate"):
        engine.run()
    assert len(controller._pending[(0, 1)].fragments) == 1

    controller.relay_syndrome(
        SyndromePayload(0, "south", 1, n_fragments=3),
        deliver,
    )
    with pytest.raises(ValueError, match="same count"):
        engine.run()
    assert len(controller._pending[(0, 1)].fragments) == 1

    controller.relay_syndrome(_frag("south"), lambda packet: None)
    with pytest.raises(ValueError, match="delivery sink"):
        engine.run()
    assert len(controller._pending[(0, 1)].fragments) == 1

    controller.relay_syndrome(_frag("south"), deliver)
    engine.run()
    assert len(delivered) == 1
    assert (0, 1) in controller._completed_rounds

    controller.relay_syndrome(_frag("late"), deliver)
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
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []
    payload = SyndromePayload(0, 0, 1, size_bits=1000)
    eng.schedule(0, lambda: ctrl.relay_syndrome(
        payload, lambda p: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(1.5)]


def test_serialized_chip_link_queues_concurrent_rounds():
    # two 1000-bit rounds on a shared 1000 bits/us qc bus: 1 us then 2 us
    eng = Engine(verbose=False)
    links = _links(qc=_actual_edge(bandwidth=1000),
                   cwd=_actual_edge())
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []
    for round_index in (1, 2):
        payload = SyndromePayload(0, 0, round_index, size_bits=1000)
        eng.schedule(0, lambda p=payload: ctrl.relay_syndrome(
            p, lambda q: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(1.0), us(2.0)]


def test_generic_send_prices_bits_and_queues_on_serialized_links():
    # the same two-message queueing through Transport.send on a dd bus
    eng = Engine(verbose=False)
    links = _links(dd=_actual_edge(bandwidth=1000))
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []

    def send_both():
        for round_index in (1, 2):
            ctrl.send(
                LinkPath.DD,
                SyndromePayload(0, 0, round_index, size_bits=1000),
                lambda p: arrivals.append(eng.now),
                now=eng.now,
                attribution=_window_attribution(round_index),
            )

    eng.schedule(0, send_both)
    eng.run()
    assert arrivals == [us(1.0), us(2.0)]


def test_packed_fragments_are_priced_by_their_total_bits():
    # 400 + 600 bit fragments = one 1000-bit packet over 1000 bits/us (cd)
    eng = Engine(verbose=False)
    links = _links(qc=_actual_edge(),
                   cwd=_actual_edge(bandwidth=1000))
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []
    deliver = lambda packet: arrivals.append(eng.now)
    for patch, bits in ((0, 400), (1, 600)):
        payload = SyndromePayload(0, patch, 1, n_fragments=2, size_bits=bits)
        eng.schedule(0, lambda p=payload: ctrl.relay_syndrome(
            p, deliver))
    eng.run()
    assert arrivals == [us(1.0)]
