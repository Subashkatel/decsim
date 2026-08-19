"""Blind Q-062(d/f) contracts for the priced controller-to-Buffer-0 hop."""

from types import SimpleNamespace

import pytest

from decsim.config import us
from decsim.links.link_profiles import logical_reference_profile, with_controller_to_buffer_edge
from decsim.links.links import LinkPath, TrafficAttribution
from decsim.controller.syndrome_ingress import SyndromePacketRouteKind, SyndromeIngress, _IngressSlotState
from decsim.links.link_traffic_report import topology_json_value, traffic_json_value


class _Engine:
    def __init__(self, now=0):
        self.now = now
        self.events = []

    def schedule(self, delay, callback, *, label):
        self.events.append((self.now + delay, callback, label))

    def run_next(self):
        tick, callback, _label = min(self.events, key=lambda event: event[0])
        self.events.remove((tick, callback, _label))
        self.now = tick
        callback()


class _Receiver:
    def __init__(self, answers):
        self.answers = list(answers)
        self.received = []

    def accept_window_input(self, packet):
        self.received.append(packet)
        return self.answers.pop(0)


class _PublicationSpy:
    def __init__(self):
        self.calls = []

    def mark_publication_tick(self, round_identity, publication_tick):
        self.calls.append((round_identity, publication_tick))


def _wired_profile(*, latency_us=0.25, bandwidth=100.0, source="Q-062 test card"):
    return with_controller_to_buffer_edge(
        logical_reference_profile(),
        latency_us=latency_us,
        aggregate_bits_per_us=bandwidth,
        source=source,
    )


def test_legacy_cards_leave_optional_c2b_absent_without_changing_edge_identity():
    legacy = logical_reference_profile()
    legacy_edges = {path.value: getattr(legacy, path.value) for path in legacy.wired_paths()}

    extended = with_controller_to_buffer_edge(
        legacy, latency_us=0.25, aggregate_bits_per_us=100.0, source="Q-062 test card")

    assert legacy.c2b is None
    assert LinkPath.C2B not in legacy.wired_paths()
    assert LinkPath.C2B not in legacy.resolve().paths
    assert extended.c2b is not None
    assert LinkPath.C2B in extended.wired_paths()
    for path_name, edge in legacy_edges.items():
        assert getattr(extended, path_name) is edge


def test_wired_c2b_card_preserves_positive_numbers_source_and_physical_topology():
    source = "explicit Q-062 PROJECT_DESIGN latency and bandwidth"
    model = _wired_profile(latency_us=0.25, bandwidth=120.0, source=source).resolve()
    topology = topology_json_value(model.snapshot())
    c2b_edge = next(edge for edge in topology["edges"] if edge["path"] == "c2b")
    channel = next(
        row for row in topology["physical_channels"]
        if row["physical_alias"] == c2b_edge["physical_alias"]
    )

    assert topology["path_order"].count("c2b") == 1
    assert channel["member_paths"] == ["c2b"]
    assert channel["propagation_latency_ticks"] == us(0.25)
    assert channel["capacity"]["aggregate_bits_per_us"] == 120.0
    assert channel["configuration_source"] == source
    assert c2b_edge["actual_payload_source"] == "SyndromeRoundPacket.fragment_size_sum"


def test_c2b_traffic_uses_exact_round_attribution_payload_and_fifo_delays():
    model = _wired_profile(latency_us=0.25, bandwidth=100.0).resolve()
    attribution = TrafficAttribution(
        operation_id=7, patch_ids=(2, 9), window_id=None, round_lo=11, round_hi=11)

    first = model.reserve(
        LinkPath.C2B, payload_bits=300, now_ticks=10,
        attribution=attribution,
    )
    second = model.reserve(
        LinkPath.C2B, payload_bits=200, now_ticks=10,
        attribution=TrafficAttribution(
            operation_id=7, patch_ids=(2, 9), window_id=None,
            round_lo=12, round_hi=12),
    )
    traffic = traffic_json_value(model.snapshot())
    rows = [row for row in traffic["transfers"] if row["path"] == "c2b"]

    assert first.serialization_ticks == us(3.0)
    assert first.propagation_ticks == us(0.25)
    assert first.total_delay_ticks == us(3.25)
    assert second.queue_wait_ticks == first.serialization_ticks
    assert [row["payload_bits"] for row in rows] == [300, 200]
    assert rows[0]["payload_source"] == "SyndromeRoundPacket.fragment_size_sum"
    attribution_row = rows[0]["attribution"]
    assert attribution_row["operation_id"]["value"] == "7"
    assert [p["value"] for p in attribution_row["patch_ids"]] == ["2", "9"]
    assert (attribution_row["window_id"], attribution_row["round_lo"],
            attribution_row["round_hi"], attribution_row["relation"]) == (None, 11, 11, None)
    assert all(row["reconciles"] for row in traffic["reconciliation"])


def test_finite_c2b_bandwidth_charges_serialization_plus_propagation():
    model = _wired_profile(latency_us=0.10, bandwidth=1000.0).resolve()

    reservation = model.reserve(
        LinkPath.C2B, payload_bits=500, now_ticks=0,
        attribution=TrafficAttribution(
            operation_id=1, patch_ids=(0,), window_id=None, round_lo=1, round_hi=1))

    assert reservation.serialization_ticks == us(0.5)   # 500 bits at 1000 bits/us
    assert reservation.propagation_ticks == us(0.10)
    assert reservation.total_delay_ticks == us(0.60)


def test_unbounded_c2b_bandwidth_charges_propagation_only():
    model = _wired_profile(latency_us=0.10, bandwidth=None).resolve()

    first = model.reserve(
        LinkPath.C2B, payload_bits=500, now_ticks=0,
        attribution=TrafficAttribution(
            operation_id=1, patch_ids=(0,), window_id=None, round_lo=1, round_hi=1))
    second = model.reserve(
        LinkPath.C2B, payload_bits=500, now_ticks=0,
        attribution=TrafficAttribution(
            operation_id=1, patch_ids=(0,), window_id=None, round_lo=2, round_hi=2))

    assert first.serialization_ticks == 0
    assert first.total_delay_ticks == us(0.10)
    assert second.queue_wait_ticks == 0                 # no serialization, so no FIFO wait
    assert second.total_delay_ticks == us(0.10)


def test_ingress_reserves_c2b_exactly_once_and_publishes_at_arrival_before_retry():
    engine = _Engine(now=1_000)
    links = _wired_profile(latency_us=0.25, bandwidth=100.0).resolve()
    receiver = _Receiver([False, True])
    publication = _PublicationSpy()
    packet = SimpleNamespace(
        operation_id=17,
        round_index=4,
        fragments=(
            SimpleNamespace(patch_id=2, size_bits=120),
            SimpleNamespace(patch_id=9, size_bits=180),
        ),
    )
    slot = SimpleNamespace(
        identity="ctx", round_key=(17, 4),
        packet=packet,
        packet_bits=300,
        c2b_reserved=False,
        c2b_delivered=False,
        state=_IngressSlotState.PACKED_WAIT,
    )
    ingress = object.__new__(SyndromeIngress)
    ingress.engine = engine
    ingress.links = links
    ingress.window_input_receiver = receiver
    ingress.syndrome_buffer = publication
    slot.route = SimpleNamespace(kind=SyndromePacketRouteKind.WINDOW_INPUT)
    ingress._contexts = {"ctx": slot}
    ingress._route_queues = {SyndromePacketRouteKind.WINDOW_INPUT: ["ctx"]}

    assert ingress._transmit_window_input_round(slot) is True
    assert receiver.received == []
    assert publication.calls == []
    assert len(traffic_json_value(links.snapshot())["transfers"]) == 1

    engine.run_next()
    arrival_tick = 1_000 + us(0.25) + us(3.0)
    assert engine.now == arrival_tick
    assert receiver.received == [packet]
    assert publication.calls == [((17, 4), arrival_tick)]
    assert slot.state is _IngressSlotState.PACKED_WAIT

    assert ingress._transmit_window_input_round(slot) is True
    assert receiver.received == [packet, packet]
    assert publication.calls == [((17, 4), arrival_tick)]
    assert len(traffic_json_value(links.snapshot())["transfers"]) == 1


def test_unwired_legacy_ingress_delivery_is_immediate_and_has_no_c2b_publication():
    engine = _Engine(now=321)
    links = logical_reference_profile().resolve()
    receiver = _Receiver([True])
    publication = _PublicationSpy()
    packet = SimpleNamespace(
        operation_id=3, round_index=8,
        fragments=(SimpleNamespace(patch_id=1, size_bits=16),),
    )
    slot = SimpleNamespace(
        identity="ctx", round_key=(3, 8),
        packet=packet, packet_bits=16, c2b_reserved=False,
        c2b_delivered=False, state=_IngressSlotState.PACKED_WAIT,
    )
    ingress = object.__new__(SyndromeIngress)
    ingress.engine = engine
    ingress.links = links
    ingress.window_input_receiver = receiver
    ingress.syndrome_buffer = publication
    slot.route = SimpleNamespace(kind=SyndromePacketRouteKind.WINDOW_INPUT)
    ingress._contexts = {"ctx": slot}
    ingress._route_queues = {SyndromePacketRouteKind.WINDOW_INPUT: ["ctx"]}

    assert ingress._transmit_window_input_round(slot) is True
    assert receiver.received == [packet]
    assert publication.calls == []
    assert "c2b" not in traffic_json_value(links.snapshot())["path_order"]


def test_rounds_pipeline_on_c2b_instead_of_stop_and_wait():
    """Fast rounds arrive at Buffer 0 spaced by the round period, not by the C2B
    latency: the link serializes them FIFO and propagation is pipelined."""
    stim = pytest.importorskip("stim")
    from decsim.qpu.stim_device import StimDevice
    from decsim.config import TimingConfig, TICKS_PER_US
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=8, distance=3,
        after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001, before_round_data_depolarization=0.001)
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    completed = RunSpec(
        ops=[op], d=3, rounds_policy=FixedRounds(8), device=StimDevice(),
        decoder=PresetLatencyDecoder(0.01),
        timing=TimingConfig(round_us=0.02),
        links=with_controller_to_buffer_edge(
            logical_reference_profile(), latency_us=0.25,
            aggregate_bits_per_us=10_000.0, source="test"),
        seed=0).build()
    delivered = sorted(row["delivery_ticks"] for row in completed.result.link_traffic["transfers"]
                       if row["path"] == "c2b")
    assert len(delivered) == 8
    gaps = [(b - a) / TICKS_PER_US for a, b in zip(delivered, delivered[1:])]
    assert all(gap == pytest.approx(0.02, abs=1e-3) for gap in gaps), gaps   # first gap adds serialization
