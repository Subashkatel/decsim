"""A link delivers when the point-to-point law says, and setups serialize.

Sources: ns-3 PointToPointNetDevice (one packet on the wire at a time; a
packet that arrives while the wire is busy waits, is serialized at the
data rate, and is received one propagation delay after its last bit
leaves; a fractional tick of serialization rounds up); gem5-Aladdin's DMA
setup (Shao et al., MICRO 2016: the processor programs each transfer
before the data mover starts, and successive setups serialize on that
processor while the wire keeps streaming; gem5's dma_device keeps one
transmit list per port, so a setup queue belongs to a channel); the row
L1 harness in the sandbox validation folder, which runs the closed form
over random traces. One microsecond is 1_000_000 ticks.
"""

import dataclasses
import math

import pytest

import decsim.links.links as links
import decsim.message as message


def bounded_channel(bits_per_microsecond, latency_ticks):
    capacity = links.LinkCapacityConfig(
        bits_per_microsecond,
        links.LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "test",
    )
    return links.LinkConfig(latency_ticks, capacity, "test")


def unbounded_channel(latency_ticks):
    return links.LinkConfig(latency_ticks, None, "test")


def free_edge_on(channel):
    return links.LinkEdgeConfig(channel, None, "test payload", None)


def edge_with_setup_on(channel, overhead_ticks):
    overhead = links.TransferOverheadConfig(overhead_ticks, "test")
    return links.LinkEdgeConfig(channel, None, "test payload", overhead)


def card(**edges):
    """A card whose paths are free unless the test wires them itself."""
    free_channel = unbounded_channel(0)
    free_edge = free_edge_on(free_channel)
    wiring = dict.fromkeys(
        ("qc", "wbd", "wsd", "sbd", "wdo", "dd", "do", "oc", "cq"), free_edge
    )
    wiring.update(edges)
    return links.LinkModelConfig(profile_name="test", **wiring)


def round_attribution(round_index):
    return links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


def request_relation_for(window_id):
    request_key = message.DecoderRequestKey(
        operation_id=1,
        window_id=window_id,
        tier=message.DecoderTier.WEAK,
        run_sequence=window_id,
    )
    return links.RequestTransferRelation(request_key)


def window_attribution(window_id):
    relation = request_relation_for(window_id)
    return links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=window_id,
        first_round=window_id,
        last_round=window_id,
        relation=relation,
    )


def window_attribution_with_request_for(window_id, request_window_id):
    relation = request_relation_for(request_window_id)
    return links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=window_id,
        first_round=window_id,
        last_round=window_id,
        relation=relation,
    )


def test_delivery_on_an_unbounded_link_is_send_plus_latency():
    channel = unbounded_channel(7)
    link = links.Link(channel)
    reservation = link.reserve(payload_bits=1000, now_ticks=10)
    assert reservation.serialization_ticks == 0
    assert reservation.queue_wait_ticks == 0
    assert reservation.propagation_ticks == 7
    assert reservation.total_delay_ticks == 7
    delivery_ticks = 10 + reservation.total_delay_ticks
    assert delivery_ticks == 17


def test_delivery_on_a_bounded_link_is_send_plus_serialization_plus_latency():
    channel = bounded_channel(1000.0, 300)
    link = links.Link(channel)
    reservation = link.reserve(payload_bits=8, now_ticks=10)
    assert reservation.serialization_ticks == 8000
    assert reservation.serializer_start_ticks == 10
    assert reservation.serializer_end_ticks == 8010
    assert reservation.total_delay_ticks == 8300
    delivery_ticks = reservation.serializer_end_ticks + 300
    assert delivery_ticks == 10 + reservation.total_delay_ticks


def test_a_fractional_tick_of_serialization_rounds_up():
    channel = bounded_channel(3.0, 0)
    link = links.Link(channel)
    reservation = link.reserve(payload_bits=1, now_ticks=0)
    assert reservation.serialization_ticks == 333334


# The old suite has the same case; the new suite carries every law on
# its own once the old tests are deleted.
def test_serialization_uses_the_decimal_rate_written_on_the_card():
    channel = bounded_channel(0.20846, 0)
    link = links.Link(channel)
    reservation = link.reserve(payload_bits=400_310_292, now_ticks=0)
    assert reservation.serialization_ticks == 1_920_321_845_917_683


def test_a_per_channel_rate_serializes_at_the_aggregate_rate():
    capacity = links.LinkCapacityConfig(
        2.0, links.LinkQuantityBasis.PER_CHANNEL, 4, "test"
    )
    channel = links.LinkConfig(0, capacity, "test")
    link = links.Link(channel)
    reservation = link.reserve(payload_bits=16, now_ticks=0)
    assert reservation.serialization_ticks == 2_000_000


def test_a_per_channel_rate_reports_the_input_times_the_count():
    capacity = links.LinkCapacityConfig(
        2.0, links.LinkQuantityBasis.PER_CHANNEL, 4, "test"
    )
    assert capacity.aggregate_bits_per_microsecond == 8.0


def test_a_per_channel_rate_is_multiplied_exactly():
    capacity = links.LinkCapacityConfig(
        0.7, links.LinkQuantityBasis.PER_CHANNEL, 3, "test"
    )
    channel = links.LinkConfig(0, capacity, "test")
    link = links.Link(channel)
    reservation = link.reserve(payload_bits=21, now_ticks=0)
    assert reservation.serialization_ticks == 10_000_000


def test_two_transfers_on_one_channel_serialize_in_request_order():
    channel = bounded_channel(1000.0, 0)
    link = links.Link(channel)
    first = link.reserve(payload_bits=8, now_ticks=10)
    second = link.reserve(payload_bits=8, now_ticks=20)
    assert first.serializer_end_ticks == 8010
    assert second.serializer_start_ticks == 8010
    assert second.queue_wait_ticks == 7990
    assert second.total_delay_ticks == 15990
    assert (first.physical_sequence, second.physical_sequence) == (0, 1)


def test_a_bounded_channel_does_not_queue_after_an_idle_gap():
    channel = bounded_channel(1000.0, 0)
    link = links.Link(channel)
    first = link.reserve(payload_bits=8, now_ticks=10)
    second = link.reserve(payload_bits=8, now_ticks=9000)
    assert first.serializer_end_ticks == 8010
    assert second.queue_wait_ticks == 0
    assert second.serializer_start_ticks == 9000


def test_a_send_before_the_previous_send_is_refused():
    channel = unbounded_channel(0)
    link = links.Link(channel)
    link.reserve(payload_bits=1, now_ticks=10)
    with pytest.raises(ValueError, match="precede"):
        link.reserve(payload_bits=1, now_ticks=5)


def test_a_negative_payload_is_refused():
    channel = unbounded_channel(0)
    link = links.Link(channel)
    with pytest.raises(ValueError, match="nonnegative"):
        link.reserve(payload_bits=-1, now_ticks=0)


def test_a_transfer_setup_queues_behind_the_previous_setup_on_the_channel():
    shared_channel = unbounded_channel(0)
    weak_edge = edge_with_setup_on(shared_channel, overhead_ticks=5)
    strong_edge = edge_with_setup_on(shared_channel, overhead_ticks=5)
    shared_card = card(wbd=weak_edge, sbd=strong_edge)
    model = shared_card.build()
    first_window = window_attribution(1)
    second_window = window_attribution(2)
    third_window = window_attribution(3)
    first = model.reserve(
        links.LinkPath.WBD,
        payload_bits=8,
        now_ticks=10,
        attribution=first_window,
    )
    second = model.reserve(
        links.LinkPath.WBD,
        payload_bits=8,
        now_ticks=11,
        attribution=second_window,
    )
    third = model.reserve(
        links.LinkPath.SBD,
        payload_bits=8,
        now_ticks=12,
        attribution=third_window,
    )
    assert (first.setup_ticks, first.send_ticks) == (5, 15)
    assert (second.setup_ticks, second.send_ticks) == (9, 20)
    assert (third.setup_ticks, third.send_ticks) == (13, 25)
    assert third.total_delay_ticks == 13


def test_a_free_edge_on_a_busy_channel_waits_for_the_setup_in_progress():
    shared_channel = unbounded_channel(0)
    weak_edge = edge_with_setup_on(shared_channel, overhead_ticks=5)
    strong_edge = free_edge_on(shared_channel)
    shared_card = card(wbd=weak_edge, sbd=strong_edge)
    model = shared_card.build()
    first_window = window_attribution(1)
    second_window = window_attribution(2)
    model.reserve(
        links.LinkPath.WBD,
        payload_bits=8,
        now_ticks=10,
        attribution=first_window,
    )
    reservation = model.reserve(
        links.LinkPath.SBD,
        payload_bits=8,
        now_ticks=12,
        attribution=second_window,
    )
    assert (reservation.setup_ticks, reservation.send_ticks) == (3, 15)


def test_setups_on_different_channels_do_not_wait_for_each_other():
    slow_channel = unbounded_channel(0)
    fast_channel = unbounded_channel(0)
    qpu_edge = edge_with_setup_on(slow_channel, overhead_ticks=50)
    buffer_edge = edge_with_setup_on(fast_channel, overhead_ticks=5)
    two_channel_card = card(qc=qpu_edge, cwb=buffer_edge)
    model = two_channel_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    slow = model.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    fast = model.reserve(
        links.LinkPath.CWB,
        payload_bits=8,
        now_ticks=0,
        attribution=second_round,
    )
    assert (slow.setup_ticks, slow.send_ticks) == (50, 50)
    assert (fast.setup_ticks, fast.send_ticks) == (5, 5)


def test_setup_serialization_and_propagation_add_up_on_one_bounded_channel():
    channel = bounded_channel(1000.0, 300)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    bounded_card = card(qc=qpu_edge)
    model = bounded_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    first = model.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    second = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=0,
        attribution=second_round,
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


def test_a_free_edge_costs_nothing():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    reservation = model.reserve(
        links.LinkPath.QC,
        payload_bits=1000,
        now_ticks=42,
        attribution=attribution,
    )
    assert reservation.send_ticks == 42
    assert reservation.setup_ticks == 0
    assert reservation.serialization_ticks == 0
    assert reservation.propagation_ticks == 0
    assert reservation.total_delay_ticks == 0


def test_an_unbounded_channel_carries_a_transfer_with_no_payload_size():
    channel = unbounded_channel(9)
    qpu_edge = free_edge_on(channel)
    latency_card = card(qc=qpu_edge)
    model = latency_card.build()
    attribution = round_attribution(1)
    reservation = model.reserve(
        links.LinkPath.QC,
        payload_bits=None,
        now_ticks=7,
        attribution=attribution,
    )
    assert reservation.payload_bits is None
    assert reservation.total_delay_ticks == 9


def test_a_bounded_channel_refuses_a_transfer_with_no_payload_size():
    channel = bounded_channel(1000.0, 0)
    buffer_edge = free_edge_on(channel)
    bounded_card = card(cwb=buffer_edge)
    model = bounded_card.build()
    attribution = round_attribution(1)
    with pytest.raises(RuntimeError, match="bounded"):
        model.reserve(
            links.LinkPath.CWB,
            payload_bits=None,
            now_ticks=0,
            attribution=attribution,
        )
    snapshot = model.snapshot()
    assert snapshot.transfers == ()


def test_a_request_earlier_than_the_channels_previous_request_is_refused():
    shared_channel = unbounded_channel(0)
    qpu_edge = edge_with_setup_on(shared_channel, overhead_ticks=5)
    buffer_edge = edge_with_setup_on(shared_channel, overhead_ticks=5)
    shared_card = card(qc=qpu_edge, cwb=buffer_edge)
    model = shared_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    third_round = round_attribution(3)
    fourth_round = round_attribution(4)
    model.reserve(
        links.LinkPath.CWB,
        payload_bits=8,
        now_ticks=10,
        attribution=first_round,
    )
    with pytest.raises(RuntimeError, match="nondecreasing order"):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=8,
            now_ticks=9,
            attribution=second_round,
        )
    with pytest.raises(RuntimeError, match="nondecreasing order"):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=8,
            now_ticks=9,
            attribution=third_round,
        )
    untouched = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=10,
        attribution=fourth_round,
    )
    assert (untouched.setup_ticks, untouched.send_ticks) == (10, 20)


def test_the_snapshot_counts_every_transfer_per_path():
    shared_channel = bounded_channel(1000.0, 300)
    qpu_edge = free_edge_on(shared_channel)
    buffer_edge = free_edge_on(shared_channel)
    shared_card = card(qc=qpu_edge, cwb=buffer_edge)
    model = shared_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    model.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    model.reserve(
        links.LinkPath.CWB,
        payload_bits=4,
        now_ticks=0,
        attribution=second_round,
    )
    snapshot = model.snapshot()
    counters_by_path = {edge.path: edge.counters for edge in snapshot.edges}
    qpu_counters = counters_by_path[links.LinkPath.QC]
    buffer_counters = counters_by_path[links.LinkPath.CWB]
    assert qpu_counters.transfer_count == 1
    assert qpu_counters.known_payload_bits == 8
    assert qpu_counters.serialization_ticks == 8000
    assert qpu_counters.propagation_ticks == 300
    assert qpu_counters.queue_wait_ticks == 0
    assert buffer_counters.transfer_count == 1
    assert buffer_counters.known_payload_bits == 4
    assert buffer_counters.serialization_ticks == 4000
    assert buffer_counters.propagation_ticks == 300
    assert buffer_counters.queue_wait_ticks == 8000
    assert len(snapshot.transfers) == 2


def test_a_channels_counters_are_the_sum_of_its_paths_counters():
    shared_channel = bounded_channel(1000.0, 300)
    qpu_edge = free_edge_on(shared_channel)
    buffer_edge = free_edge_on(shared_channel)
    shared_card = card(qc=qpu_edge, cwb=buffer_edge)
    model = shared_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    model.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    model.reserve(
        links.LinkPath.CWB,
        payload_bits=4,
        now_ticks=0,
        attribution=second_round,
    )
    snapshot = model.snapshot()
    counters_by_path = {edge.path: edge.counters for edge in snapshot.edges}
    channel_by_members = {
        channel.member_paths: channel for channel in snapshot.channels
    }
    shared = channel_by_members[(links.LinkPath.QC, links.LinkPath.CWB)]
    assert shared.counters.transfer_count == 2
    assert shared.counters.known_payload_bits == 12
    assert shared.counters.serialization_ticks == 12000
    assert shared.counters.propagation_ticks == 600
    assert shared.counters.queue_wait_ticks == 8000
    qpu_counters = counters_by_path[links.LinkPath.QC]
    buffer_counters = counters_by_path[links.LinkPath.CWB]
    summed_over_paths = qpu_counters.plus(buffer_counters)
    assert summed_over_paths == shared.counters


def test_a_card_without_a_required_path_is_refused():
    complete_card = card()
    incomplete = dataclasses.replace(complete_card, oc=None)
    with pytest.raises(ValueError, match="oc is a required link path"):
        incomplete.wired_paths()


def test_a_transfer_on_an_unwired_path_is_refused():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    with pytest.raises(ValueError, match="not wired"):
        model.reserve(
            links.LinkPath.CWB,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_an_attribution_outside_the_paths_scope_is_refused():
    complete_card = card()
    model = complete_card.build()
    attribution = window_attribution(1)
    with pytest.raises(ValueError, match="requires round attribution"):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_window_transfer_without_its_request_relation_is_refused():
    complete_card = card()
    model = complete_card.build()
    attribution = links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=3,
        first_round=3,
        last_round=3,
    )
    with pytest.raises(ValueError, match="requires a request relation"):
        model.reserve(
            links.LinkPath.SBD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_relation_that_names_another_window_is_refused():
    complete_card = card()
    model = complete_card.build()
    attribution = window_attribution_with_request_for(3, request_window_id=4)
    with pytest.raises(ValueError, match="does not match"):
        model.reserve(
            links.LinkPath.SBD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_an_actual_payload_on_an_edge_without_a_source_is_refused():
    channel = unbounded_channel(0)
    default_payload = links.PayloadSizeConfig(
        100, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    default_only_edge = links.LinkEdgeConfig(channel, default_payload, None)
    default_card = card(qc=default_only_edge)
    model = default_card.build()
    attribution = round_attribution(1)
    with pytest.raises(ValueError, match="actual payload source"):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=3,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_capacity_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="finite"):
        links.LinkCapacityConfig(
            math.inf, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
        )


def test_a_default_payload_on_another_basis_than_the_capacity_is_refused():
    capacity = links.LinkCapacityConfig(
        2.0, links.LinkQuantityBasis.PER_CHANNEL, 4, "test"
    )
    channel = links.LinkConfig(0, capacity, "test")
    aggregate_payload = links.PayloadSizeConfig(
        8, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    with pytest.raises(ValueError, match="bases must match"):
        links.LinkEdgeConfig(channel, aggregate_payload, None)


def test_a_negative_payload_leaves_the_channel_untouched():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    with pytest.raises(
        RuntimeError, match="a payload is a whole number of bits, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=-1,
            now_ticks=10,
            attribution=first_round,
        )
    accepted = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=10,
        attribution=second_round,
    )
    assert (accepted.setup_ticks, accepted.send_ticks) == (5, 15)
    assert accepted.physical_sequence == 0
    snapshot = model.snapshot()
    assert len(snapshot.transfers) == 1


def test_a_fractional_payload_leaves_the_channel_untouched():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    with pytest.raises(
        RuntimeError, match="a payload is a whole number of bits, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=1.5,
            now_ticks=10,
            attribution=first_round,
        )
    accepted = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=10,
        attribution=second_round,
    )
    assert (accepted.setup_ticks, accepted.send_ticks) == (5, 15)
    assert accepted.physical_sequence == 0
    snapshot = model.snapshot()
    assert len(snapshot.transfers) == 1


def test_a_fractional_request_tick_leaves_the_channel_untouched():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    with pytest.raises(
        RuntimeError, match="a request tick is a whole number, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=8,
            now_ticks=10.5,
            attribution=first_round,
        )
    accepted = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=10,
        attribution=second_round,
    )
    assert (accepted.setup_ticks, accepted.send_ticks) == (5, 15)
    assert accepted.physical_sequence == 0
    snapshot = model.snapshot()
    assert len(snapshot.transfers) == 1


def test_a_request_tick_that_is_not_a_number_leaves_the_channel_untouched():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    with pytest.raises(
        RuntimeError, match="a request tick is a whole number, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=8,
            now_ticks=math.nan,
            attribution=first_round,
        )
    accepted = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=10,
        attribution=second_round,
    )
    assert (accepted.setup_ticks, accepted.send_ticks) == (5, 15)
    assert accepted.physical_sequence == 0
    snapshot = model.snapshot()
    assert len(snapshot.transfers) == 1


def test_a_negative_first_request_tick_leaves_the_channel_untouched():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    with pytest.raises(
        RuntimeError, match="a request tick is a whole number, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=8,
            now_ticks=-5,
            attribution=first_round,
        )
    accepted = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=10,
        attribution=second_round,
    )
    assert (accepted.setup_ticks, accepted.send_ticks) == (5, 15)
    assert accepted.physical_sequence == 0
    snapshot = model.snapshot()
    assert len(snapshot.transfers) == 1


def operation_attribution():
    return links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=None,
        last_round=None,
    )


def test_a_request_exactly_at_the_serializers_free_tick_waits_zero():
    channel = bounded_channel(1000.0, 0)
    link = links.Link(channel)
    first = link.reserve(payload_bits=8, now_ticks=10)
    second = link.reserve(payload_bits=8, now_ticks=8010)
    assert first.serializer_end_ticks == 8010
    assert second.queue_wait_ticks == 0
    assert second.serializer_start_ticks == 8010


def test_an_unbounded_channel_never_queues_on_back_to_back_sends():
    channel = unbounded_channel(7)
    link = links.Link(channel)
    first = link.reserve(payload_bits=1000, now_ticks=10)
    second = link.reserve(payload_bits=1000, now_ticks=10)
    third = link.reserve(payload_bits=1000, now_ticks=10)
    assert (first.queue_wait_ticks, first.serializer_start_ticks) == (0, 10)
    assert (second.queue_wait_ticks, second.serializer_start_ticks) == (0, 10)
    assert (third.queue_wait_ticks, third.serializer_start_ticks) == (0, 10)
    assert third.total_delay_ticks == 7


def test_an_empty_payload_on_a_bounded_channel_pays_setup_only():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    attribution = round_attribution(1)
    reservation = model.reserve(
        links.LinkPath.QC, payload_bits=0, now_ticks=10, attribution=attribution
    )
    assert reservation.serialization_ticks == 0
    assert reservation.setup_ticks == 5
    assert reservation.total_delay_ticks == 5


def test_the_physical_sequence_counts_per_channel_across_paths():
    shared_channel = unbounded_channel(0)
    other_channel = unbounded_channel(0)
    qpu_edge = free_edge_on(shared_channel)
    buffer_edge = free_edge_on(shared_channel)
    weak_edge = free_edge_on(other_channel)
    two_channel_card = card(qc=qpu_edge, cwb=buffer_edge, wbd=weak_edge)
    model = two_channel_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    third_round = round_attribution(3)
    on_shared_first = model.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=0, attribution=first_round
    )
    on_shared_second = model.reserve(
        links.LinkPath.CWB,
        payload_bits=8,
        now_ticks=0,
        attribution=second_round,
    )
    on_other = model.reserve(
        links.LinkPath.WBD,
        payload_bits=8,
        now_ticks=0,
        attribution=third_round,
    )
    assert on_shared_first.physical_sequence == 0
    assert on_shared_second.physical_sequence == 1
    assert on_other.physical_sequence == 0


def test_a_setup_after_the_queue_went_idle_costs_only_its_own_overhead():
    channel = unbounded_channel(0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    first = model.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=10, attribution=first_round
    )
    later = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=100,
        attribution=second_round,
    )
    assert first.send_ticks == 15
    assert (later.setup_ticks, later.send_ticks) == (5, 105)


def test_a_transfer_with_no_payload_size_counts_as_unknown():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    model.reserve(
        links.LinkPath.QC,
        payload_bits=None,
        now_ticks=0,
        attribution=attribution,
    )
    snapshot = model.snapshot()
    counters_by_path = {edge.path: edge.counters for edge in snapshot.edges}
    qpu_counters = counters_by_path[links.LinkPath.QC]
    assert qpu_counters.unknown_payload_transfer_count == 1
    assert qpu_counters.known_payload_bits == 0
    assert qpu_counters.transfer_count == 1


def test_a_zero_capacity_is_refused():
    with pytest.raises(ValueError, match="positive"):
        links.LinkCapacityConfig(
            0.0, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
        )


def test_a_negative_capacity_is_refused():
    with pytest.raises(ValueError, match="positive"):
        links.LinkCapacityConfig(
            -1.0, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
        )


def test_a_round_or_window_path_refuses_an_attribution_without_rounds():
    complete_card = card()
    model = complete_card.build()
    attribution = operation_attribution()
    with pytest.raises(ValueError, match="requires round_or_window"):
        model.reserve(
            links.LinkPath.WBD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_windowed_transfer_on_the_weak_input_path_needs_its_request():
    complete_card = card()
    model = complete_card.build()
    attribution = links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=3,
        first_round=3,
        last_round=3,
    )
    with pytest.raises(ValueError, match="wbd requires a request relation"):
        model.reserve(
            links.LinkPath.WBD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_window_path_refuses_an_attribution_without_a_window():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    with pytest.raises(ValueError, match="sbd requires window attribution"):
        model.reserve(
            links.LinkPath.SBD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_the_decoder_to_decoder_path_needs_a_boundary_relation():
    complete_card = card()
    model = complete_card.build()
    attribution = window_attribution(3)
    with pytest.raises(ValueError, match="dd requires a boundary relation"):
        model.reserve(
            links.LinkPath.DD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_an_operation_only_path_refuses_an_attribution_with_rounds():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    with pytest.raises(ValueError, match="oc requires operation_only"):
        model.reserve(
            links.LinkPath.OC,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_an_operation_only_path_refuses_a_relation():
    complete_card = card()
    model = complete_card.build()
    relation = request_relation_for(3)
    attribution = links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=None,
        last_round=None,
        relation=relation,
    )
    with pytest.raises(ValueError, match="oc does not accept a relation"):
        model.reserve(
            links.LinkPath.OC,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_whole_float_tick_is_stored_as_the_same_python_int():
    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    float_card = card(qc=qpu_edge)
    int_card = card(qc=qpu_edge)
    from_float = float_card.build()
    from_int = int_card.build()
    attribution = round_attribution(1)
    float_run = from_float.reserve(
        links.LinkPath.QC,
        payload_bits=8.0,
        now_ticks=10.0,
        attribution=attribution,
    )
    int_run = from_int.reserve(
        links.LinkPath.QC, payload_bits=8, now_ticks=10, attribution=attribution
    )
    assert float_run == int_run
    assert type(float_run.setup_ticks) is int
    assert type(float_run.send_ticks) is int
    assert type(float_run.payload_bits) is int
    assert type(float_run.total_delay_ticks) is int
    assert (float_run.setup_ticks, float_run.total_delay_ticks) == (5, 8005)


def test_a_bool_is_not_a_request_tick():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    with pytest.raises(
        RuntimeError, match="a request tick is a whole number, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=8,
            now_ticks=True,
            attribution=attribution,
        )


def test_a_bool_is_not_a_payload_size():
    complete_card = card()
    model = complete_card.build()
    attribution = round_attribution(1)
    with pytest.raises(
        RuntimeError, match="a payload is a whole number of bits, not negative"
    ):
        model.reserve(
            links.LinkPath.QC,
            payload_bits=True,
            now_ticks=0,
            attribution=attribution,
        )


def test_a_numpy_tick_near_the_int64_limit_is_stored_as_a_python_int():
    import numpy

    channel = bounded_channel(1000.0, 0)
    qpu_edge = edge_with_setup_on(channel, overhead_ticks=5)
    setup_card = card(qc=qpu_edge)
    model = setup_card.build()
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    eight_below_the_limit = 2**63 - 8
    near_limit = numpy.int64(eight_below_the_limit)
    first = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=near_limit,
        attribution=first_round,
    )
    second = model.reserve(
        links.LinkPath.QC,
        payload_bits=8,
        now_ticks=near_limit,
        attribution=second_round,
    )
    assert type(first.send_ticks) is int
    assert type(first.setup_ticks) is int
    assert type(first.total_delay_ticks) is int
    assert (first.setup_ticks, first.send_ticks) == (5, 2**63 - 3)
    assert (second.setup_ticks, second.send_ticks) == (10, 2**63 + 2)


def test_capacities_built_from_the_same_numbers_compare_equal():
    first = links.LinkCapacityConfig(
        8.0, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    second = links.LinkCapacityConfig(
        8.0, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    assert first == second


def test_payload_sizes_built_from_the_same_numbers_compare_equal():
    first = links.PayloadSizeConfig(
        9, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    second = links.PayloadSizeConfig(
        9, links.LinkQuantityBasis.DIRECT_AGGREGATE, None, "test"
    )
    assert first == second


def test_transfer_overheads_built_from_the_same_numbers_compare_equal():
    first = links.TransferOverheadConfig(5, "test")
    second = links.TransferOverheadConfig(5, "test")
    assert first == second


def test_channels_built_from_the_same_numbers_compare_equal():
    first = bounded_channel(1000.0, 7)
    second = bounded_channel(1000.0, 7)
    assert first == second


def test_edges_built_from_the_same_numbers_compare_equal():
    channel = bounded_channel(1000.0, 7)
    first = links.LinkEdgeConfig(channel, None, "test payload", None)
    second = links.LinkEdgeConfig(channel, None, "test payload", None)
    assert first == second


def boundary_relation_from(source_operation_id, source_window_id, window_id):
    source_request_key = message.DecoderRequestKey(
        operation_id=source_operation_id,
        window_id=source_window_id,
        tier=message.DecoderTier.WEAK,
        run_sequence=source_window_id,
    )
    destination_window_id = window_id + 1
    return links.BoundaryTransferRelation(
        source_request_key=source_request_key,
        source_window_key=("stream", source_window_id),
        destination_window_key=("stream", destination_window_id),
        source_revision=1,
        delivery_revision=1,
    )


def window_attribution_with_boundary(window_id, relation):
    return links.TrafficAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=window_id,
        first_round=window_id,
        last_round=window_id,
        relation=relation,
    )


def test_the_decoder_to_decoder_path_accepts_a_matching_boundary():
    complete_card = card()
    model = complete_card.build()
    relation = boundary_relation_from(1, 3, 3)
    attribution = window_attribution_with_boundary(3, relation)
    reservation = model.reserve(
        links.LinkPath.DD,
        payload_bits=100,
        now_ticks=20,
        attribution=attribution,
    )
    assert reservation.send_ticks == 20
    assert reservation.payload_bits == 100
    snapshot = model.snapshot()
    record = snapshot.transfers[0]
    assert record.path is links.LinkPath.DD
    assert record.attribution.relation is relation


def test_a_boundary_from_another_window_is_refused():
    complete_card = card()
    model = complete_card.build()
    relation = boundary_relation_from(1, 4, 3)
    attribution = window_attribution_with_boundary(3, relation)
    with pytest.raises(ValueError, match="does not match"):
        model.reserve(
            links.LinkPath.DD,
            payload_bits=100,
            now_ticks=20,
            attribution=attribution,
        )


def test_a_boundary_from_another_operation_is_refused():
    complete_card = card()
    model = complete_card.build()
    relation = boundary_relation_from(2, 3, 3)
    attribution = window_attribution_with_boundary(3, relation)
    with pytest.raises(ValueError, match="does not match"):
        model.reserve(
            links.LinkPath.DD,
            payload_bits=100,
            now_ticks=20,
            attribution=attribution,
        )


def test_the_frame_to_controller_path_accepts_an_operation_only_attribution():
    complete_card = card()
    model = complete_card.build()
    attribution = operation_attribution()
    reservation = model.reserve(
        links.LinkPath.OC,
        payload_bits=32,
        now_ticks=30,
        attribution=attribution,
    )
    assert (reservation.send_ticks, reservation.payload_bits) == (30, 32)
    snapshot = model.snapshot()
    assert snapshot.transfers[0].path is links.LinkPath.OC


def test_the_controller_to_qpu_path_accepts_an_operation_only_attribution():
    complete_card = card()
    model = complete_card.build()
    attribution = operation_attribution()
    reservation = model.reserve(
        links.LinkPath.CQ,
        payload_bits=128,
        now_ticks=40,
        attribution=attribution,
    )
    assert (reservation.send_ticks, reservation.payload_bits) == (40, 128)
    snapshot = model.snapshot()
    assert snapshot.transfers[0].path is links.LinkPath.CQ


def test_the_weak_to_strong_path_accepts_a_window_with_its_request():
    complete_card = card()
    model = complete_card.build()
    attribution = window_attribution(5)
    reservation = model.reserve(
        links.LinkPath.WSD,
        payload_bits=16,
        now_ticks=50,
        attribution=attribution,
    )
    assert (reservation.send_ticks, reservation.payload_bits) == (50, 16)
    snapshot = model.snapshot()
    assert snapshot.transfers[0].path is links.LinkPath.WSD


def test_the_weak_decoder_to_frame_path_accepts_a_window_with_its_request():
    complete_card = card()
    model = complete_card.build()
    attribution = window_attribution(5)
    reservation = model.reserve(
        links.LinkPath.WDO,
        payload_bits=2,
        now_ticks=60,
        attribution=attribution,
    )
    assert (reservation.send_ticks, reservation.payload_bits) == (60, 2)
    snapshot = model.snapshot()
    assert snapshot.transfers[0].path is links.LinkPath.WDO


def test_the_strong_decoder_to_frame_path_accepts_a_window_with_its_request():
    complete_card = card()
    model = complete_card.build()
    attribution = window_attribution(5)
    reservation = model.reserve(
        links.LinkPath.DO, payload_bits=2, now_ticks=70, attribution=attribution
    )
    assert (reservation.send_ticks, reservation.payload_bits) == (70, 2)
    snapshot = model.snapshot()
    assert snapshot.transfers[0].path is links.LinkPath.DO


def test_the_controller_to_strong_buffer_path_accepts_a_round():
    channel = unbounded_channel(3)
    store_edge = free_edge_on(channel)
    stored_card = card(csb=store_edge)
    model = stored_card.build()
    attribution = round_attribution(6)
    reservation = model.reserve(
        links.LinkPath.CSB,
        payload_bits=24,
        now_ticks=80,
        attribution=attribution,
    )
    assert (reservation.send_ticks, reservation.total_delay_ticks) == (80, 3)
    snapshot = model.snapshot()
    assert snapshot.transfers[0].path is links.LinkPath.CSB
