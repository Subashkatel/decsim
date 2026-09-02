"""A link delivers when the point-to-point law says, and setups serialize.

Sources: ns-3 PointToPointNetDevice (one packet on the wire at a time; a
packet that arrives while the wire is busy waits, is serialized at the
data rate, and is received one propagation delay after its last bit
leaves; a fractional tick of serialization rounds up); gem5-Aladdin's DMA
setup (Shao et al., MICRO 2016: the processor programs each transfer
before the data mover starts, and successive setups serialize on that
processor while the wire keeps streaming); the row L1 harness
validation/component_matrix/rowL1_links/compare_point_to_point.py, which
runs the closed form over random traces. One microsecond is 1_000_000
ticks.
"""

import dataclasses
import math

import pytest

import decsim.links.links as links
from decsim.links.links import (
    Link,
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkPath,
    LinkPathRule,
    LinkQuantityBasis,
    LinkRelationRule,
    PayloadSizeConfig,
    RequestTransferRelation,
    TrafficAttribution,
    TransferOverheadConfig,
)
from decsim.message import DecoderRequestKey, DecoderTier


def bounded_channel(bits_per_microsecond, latency_ticks):
    capacity = LinkCapacityConfig(
        bits_per_microsecond, LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    return LinkConfig(latency_ticks, capacity, "test")


def unbounded_channel(latency_ticks):
    return LinkConfig(latency_ticks, None, "test")


def edge_on(channel, overhead_ticks=None):
    overhead = None
    if overhead_ticks is not None:
        overhead = TransferOverheadConfig(overhead_ticks, "test")
    return LinkEdgeConfig(channel, None, "test payload", overhead)


def card(**edges):
    """A card whose paths are free unless the test wires them itself."""
    free_channel = unbounded_channel(0)
    free_edge = edge_on(free_channel)
    wiring = dict.fromkeys(
        ("qc", "wbd", "wsd", "sbd", "wdo", "dd", "do", "oc", "cq"), free_edge
    )
    wiring.update(edges)
    return LinkModelConfig(profile_name="test", **wiring)


def round_attribution(round_index):
    return TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


def window_attribution(window_id, request_window_id=None):
    if request_window_id is None:
        request_window_id = window_id
    request_key = DecoderRequestKey(
        operation_id=1,
        window_id=request_window_id,
        tier=DecoderTier.WEAK,
        run_sequence=window_id,
    )
    relation = RequestTransferRelation(request_key)
    return TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=window_id,
        first_round=window_id,
        last_round=window_id,
        relation=relation,
    )


def test_delivery_on_an_unbounded_link_is_send_plus_latency():
    channel = unbounded_channel(7)
    link = Link(channel)
    reservation = link.reserve(payload_bits=1000, now_ticks=10)
    assert reservation.serialization_ticks == 0
    assert reservation.queue_wait_ticks == 0
    assert reservation.propagation_ticks == 7
    assert reservation.total_delay_ticks == 7
    delivery_ticks = 10 + reservation.total_delay_ticks
    assert delivery_ticks == 17


def test_delivery_on_a_bounded_link_is_send_plus_serialization_plus_latency():
    channel = bounded_channel(1000.0, 300)
    link = Link(channel)
    reservation = link.reserve(payload_bits=8, now_ticks=10)
    assert reservation.serialization_ticks == 8000
    assert reservation.serializer_start_ticks == 10
    assert reservation.serializer_end_ticks == 8010
    assert reservation.total_delay_ticks == 8300
    delivery_ticks = reservation.serializer_end_ticks + 300
    assert delivery_ticks == 10 + reservation.total_delay_ticks


def test_a_fractional_tick_of_serialization_rounds_up():
    channel = bounded_channel(3.0, 0)
    link = Link(channel)
    reservation = link.reserve(payload_bits=1, now_ticks=0)
    assert reservation.serialization_ticks == 333334


def test_a_per_channel_rate_serializes_at_the_aggregate_rate():
    capacity = LinkCapacityConfig(2.0, LinkQuantityBasis.PER_CHANNEL, 4, "t")
    channel = LinkConfig(0, capacity, "test")
    link = Link(channel)
    reservation = link.reserve(payload_bits=16, now_ticks=0)
    assert capacity.aggregate_bits_per_microsecond == 8.0
    assert reservation.serialization_ticks == 2_000_000


def test_two_transfers_on_one_channel_serialize_in_request_order():
    channel = bounded_channel(1000.0, 0)
    link = Link(channel)
    first = link.reserve(payload_bits=8, now_ticks=10)
    second = link.reserve(payload_bits=8, now_ticks=20)
    assert first.serializer_end_ticks == 8010
    assert second.serializer_start_ticks == 8010
    assert second.queue_wait_ticks == 7990
    assert second.total_delay_ticks == 15990
    assert (first.physical_sequence, second.physical_sequence) == (0, 1)


def test_a_send_before_the_previous_send_is_refused():
    channel = unbounded_channel(0)
    link = Link(channel)
    link.reserve(payload_bits=1, now_ticks=10)
    with pytest.raises(ValueError, match="precede"):
        link.reserve(payload_bits=1, now_ticks=5)


def test_a_negative_payload_is_refused():
    channel = unbounded_channel(0)
    link = Link(channel)
    with pytest.raises(ValueError, match="nonnegative"):
        link.reserve(payload_bits=-1, now_ticks=0)


def test_a_transfer_setup_queues_behind_the_previous_setup_on_the_channel():
    shared_channel = unbounded_channel(0)
    weak_edge = edge_on(shared_channel, overhead_ticks=5)
    strong_edge = edge_on(shared_channel, overhead_ticks=5)
    shared_card = card(wbd=weak_edge, sbd=strong_edge)
    model = shared_card.resolve()
    first_window = window_attribution(1)
    second_window = window_attribution(2)
    third_window = window_attribution(3)
    first = model.reserve(
        LinkPath.WBD, payload_bits=8, now_ticks=10, attribution=first_window
    )
    second = model.reserve(
        LinkPath.WBD, payload_bits=8, now_ticks=11, attribution=second_window
    )
    third = model.reserve(
        LinkPath.SBD, payload_bits=8, now_ticks=12, attribution=third_window
    )
    assert (first.setup_ticks, first.send_ticks) == (5, 15)
    assert (second.setup_ticks, second.send_ticks) == (9, 20)
    assert (third.setup_ticks, third.send_ticks) == (13, 25)
    assert third.total_delay_ticks == 13


def test_a_free_edge_on_a_busy_channel_waits_for_the_setup_in_progress():
    shared_channel = unbounded_channel(0)
    weak_edge = edge_on(shared_channel, overhead_ticks=5)
    strong_edge = edge_on(shared_channel)
    shared_card = card(wbd=weak_edge, sbd=strong_edge)
    model = shared_card.resolve()
    first_window = window_attribution(1)
    second_window = window_attribution(2)
    model.reserve(
        LinkPath.WBD, payload_bits=8, now_ticks=10, attribution=first_window
    )
    reservation = model.reserve(
        LinkPath.SBD, payload_bits=8, now_ticks=12, attribution=second_window
    )
    assert (reservation.setup_ticks, reservation.send_ticks) == (3, 15)


def test_a_free_edge_costs_nothing():
    complete_card = card()
    model = complete_card.resolve()
    attribution = round_attribution(1)
    reservation = model.reserve(
        LinkPath.QC, payload_bits=1000, now_ticks=42, attribution=attribution
    )
    assert reservation.send_ticks == 42
    assert reservation.setup_ticks == 0
    assert reservation.serialization_ticks == 0
    assert reservation.propagation_ticks == 0
    assert reservation.total_delay_ticks == 0


def test_the_snapshot_counts_every_transfer_per_path_and_per_channel():
    shared_channel = bounded_channel(1000.0, 300)
    qpu_edge = edge_on(shared_channel)
    buffer_edge = edge_on(shared_channel)
    shared_card = card(qc=qpu_edge, cwb=buffer_edge)
    model = shared_card.resolve()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    model.reserve(
        LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    model.reserve(
        LinkPath.CWB, payload_bits=4, now_ticks=0, attribution=second_round
    )
    snapshot = model.snapshot()
    counters_by_path = {edge.path: edge.counters for edge in snapshot.edges}
    channel_by_members = {
        channel.member_paths: channel for channel in snapshot.channels
    }
    shared = channel_by_members[(LinkPath.QC, LinkPath.CWB)]
    assert shared.counters.transfer_count == 2
    assert shared.counters.known_payload_bits == 12
    assert shared.counters.serialization_ticks == 12000
    assert shared.counters.propagation_ticks == 600
    assert shared.counters.queue_wait_ticks == 8000
    qpu_counters = counters_by_path[LinkPath.QC]
    buffer_counters = counters_by_path[LinkPath.CWB]
    assert qpu_counters.transfer_count == 1
    assert qpu_counters.serialization_ticks == 8000
    assert qpu_counters.propagation_ticks == 300
    assert qpu_counters.queue_wait_ticks == 0
    assert buffer_counters.transfer_count == 1
    assert buffer_counters.serialization_ticks == 4000
    assert buffer_counters.propagation_ticks == 300
    assert buffer_counters.queue_wait_ticks == 8000
    summed_over_paths = qpu_counters.plus(buffer_counters)
    assert summed_over_paths == shared.counters
    assert len(snapshot.transfers) == 2


def test_a_card_without_a_required_path_is_refused():
    complete_card = card()
    incomplete = dataclasses.replace(complete_card, oc=None)
    with pytest.raises(ValueError, match="oc is a required link path"):
        incomplete.wired_paths()


def test_a_transfer_on_an_unwired_path_is_refused():
    complete_card = card()
    model = complete_card.resolve()
    attribution = round_attribution(1)
    with pytest.raises(ValueError, match="not wired"):
        model.reserve(
            LinkPath.CWB, payload_bits=1, now_ticks=0, attribution=attribution
        )


def test_an_attribution_outside_the_paths_scope_is_refused():
    complete_card = card()
    model = complete_card.resolve()
    attribution = window_attribution(1)
    with pytest.raises(ValueError, match="requires round attribution"):
        model.reserve(
            LinkPath.QC, payload_bits=1, now_ticks=0, attribution=attribution
        )


def test_a_window_transfer_without_its_request_relation_is_refused():
    complete_card = card()
    model = complete_card.resolve()
    attribution = TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=3,
        first_round=3,
        last_round=3,
    )
    with pytest.raises(ValueError, match="requires a request relation"):
        model.reserve(
            LinkPath.SBD, payload_bits=1, now_ticks=0, attribution=attribution
        )


def test_a_relation_that_names_another_window_is_refused():
    complete_card = card()
    model = complete_card.resolve()
    attribution = window_attribution(3, request_window_id=4)
    with pytest.raises(ValueError, match="does not match"):
        model.reserve(
            LinkPath.SBD, payload_bits=1, now_ticks=0, attribution=attribution
        )


def test_an_actual_payload_on_an_edge_without_a_source_is_refused():
    channel = unbounded_channel(0)
    default_payload = PayloadSizeConfig(
        100, LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    default_only_edge = LinkEdgeConfig(channel, default_payload, None)
    default_card = card(qc=default_only_edge)
    model = default_card.resolve()
    attribution = round_attribution(1)
    with pytest.raises(ValueError, match="actual payload source"):
        model.reserve(
            LinkPath.QC, payload_bits=3, now_ticks=0, attribution=attribution
        )


def test_a_capacity_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="finite"):
        LinkCapacityConfig(
            math.inf, LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
        )


def test_a_default_payload_on_another_basis_than_the_capacity_is_refused():
    capacity = LinkCapacityConfig(2.0, LinkQuantityBasis.PER_CHANNEL, 4, "t")
    channel = LinkConfig(0, capacity, "test")
    aggregate_payload = PayloadSizeConfig(
        8, LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    with pytest.raises(ValueError, match="bases must match"):
        LinkEdgeConfig(channel, aggregate_payload, None)


def test_serialization_uses_the_decimal_rate_written_on_the_card():
    channel = bounded_channel(0.20846, 0)
    link = Link(channel)
    reservation = link.reserve(payload_bits=400_310_292, now_ticks=0)
    assert reservation.serialization_ticks == 1_920_321_845_917_683


def test_setup_serialization_and_propagation_add_up_on_one_bounded_channel():
    channel = bounded_channel(1000.0, 300)
    qpu_edge = edge_on(channel, overhead_ticks=5)
    bounded_card = card(qc=qpu_edge)
    model = bounded_card.resolve()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    first = model.reserve(
        LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    second = model.reserve(
        LinkPath.QC, payload_bits=8, now_ticks=0, attribution=second_round
    )
    assert first.payload_bits == 8
    assert first.setup_ticks == 5
    assert first.send_ticks == 5
    assert first.queue_wait_ticks == 0
    assert first.serializer_start_ticks == 5
    assert first.serialization_ticks == 8000
    assert first.serializer_end_ticks == 8005
    assert first.propagation_ticks == 300
    assert first.total_delay_ticks == 5 + 8000 + 300
    assert first.physical_sequence == 0
    assert second.payload_bits == 8
    assert second.setup_ticks == 10
    assert second.send_ticks == 10
    assert second.queue_wait_ticks == 7995
    assert second.serializer_start_ticks == 8005
    assert second.serialization_ticks == 8000
    assert second.serializer_end_ticks == 16005
    assert second.propagation_ticks == 300
    assert second.total_delay_ticks == 10 + 7995 + 8000 + 300
    assert second.physical_sequence == 1


def test_a_bounded_channel_does_not_queue_after_an_idle_gap():
    channel = bounded_channel(1000.0, 0)
    link = Link(channel)
    first = link.reserve(payload_bits=8, now_ticks=10)
    second = link.reserve(payload_bits=8, now_ticks=9000)
    assert first.serializer_end_ticks == 8010
    assert second.queue_wait_ticks == 0
    assert second.serializer_start_ticks == 9000


def test_a_request_earlier_than_the_channels_previous_request_is_refused():
    shared_channel = unbounded_channel(0)
    qpu_edge = edge_on(shared_channel)
    buffer_edge = edge_on(shared_channel)
    shared_card = card(qc=qpu_edge, cwb=buffer_edge)
    model = shared_card.resolve()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    third_round = round_attribution(3)
    model.reserve(
        LinkPath.QC, payload_bits=8, now_ticks=10, attribution=first_round
    )
    same_tick = model.reserve(
        LinkPath.CWB, payload_bits=8, now_ticks=10, attribution=second_round
    )
    assert same_tick.send_ticks == 10
    with pytest.raises(RuntimeError, match="never runs backwards"):
        model.reserve(
            LinkPath.CWB, payload_bits=8, now_ticks=9, attribution=third_round
        )


def test_a_rule_with_an_unknown_scope_is_a_loud_stop(monkeypatch):
    bogus_rule = LinkPathRule(
        "no such scope", LinkRelationRule.NONE, required=True
    )
    rules = dict(links._RULE_BY_PATH)
    rules[LinkPath.QC] = bogus_rule
    monkeypatch.setattr(links, "_RULE_BY_PATH", rules)
    complete_card = card()
    model = complete_card.resolve()
    attribution = round_attribution(1)
    with pytest.raises(RuntimeError, match="no such scope"):
        model.reserve(
            LinkPath.QC, payload_bits=8, now_ticks=0, attribution=attribution
        )


def test_a_bounded_channel_refuses_a_transfer_with_no_payload_size():
    channel = bounded_channel(1000.0, 0)
    buffer_edge = edge_on(channel)
    bounded_card = card(cwb=buffer_edge)
    model = bounded_card.resolve()
    attribution = round_attribution(1)
    with pytest.raises(RuntimeError, match="bounded"):
        model.reserve(
            LinkPath.CWB,
            payload_bits=None,
            now_ticks=0,
            attribution=attribution,
        )
    snapshot = model.snapshot()
    assert snapshot.transfers == ()


def test_an_unbounded_channel_carries_a_transfer_with_no_payload_size():
    complete_card = card()
    model = complete_card.resolve()
    attribution = round_attribution(1)
    reservation = model.reserve(
        LinkPath.QC, payload_bits=None, now_ticks=7, attribution=attribution
    )
    assert reservation.payload_bits is None
    assert reservation.total_delay_ticks == 0
