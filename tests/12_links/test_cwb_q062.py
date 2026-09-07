"""The priced controller-to-buffer-0 hop (cwb) and its card.

Sources: the cwb card in decsim/links/link_profiles.py; the rounds
pipeline onto it from the transmitter (decsim/controller/round_transmission.py).
"""

import pytest

from decsim.config import microseconds_to_ticks
from decsim.engine import Engine
from decsim.links.fabric import LinkFabric
from decsim.links.link_profiles import (
    logical_reference_profile,
    with_controller_to_weak_buffer_path,
)
from decsim.message import LinkPath, TransferAttribution
from decsim.observe.link_traffic import TrafficLedger


class _Engine:
    def __init__(self, now=0):
        self.now = now
        self.events = []

    def schedule(self, delay, callback, label=""):
        self.events.append((self.now + delay, callback, label))

    def log_io(self, who, message):
        """The I/O trace is off in these tests; packing still narrates."""

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


def _wired_profile(
    *, latency_us=0.25, bandwidth=100.0, source="Q-062 test card"
):
    return with_controller_to_weak_buffer_path(
        logical_reference_profile(),
        latency_microseconds=latency_us,
        aggregate_bits_per_microsecond=bandwidth,
        source=source,
    )


def _fabric_and_ledger(settings, engine):
    ledger = TrafficLedger(settings)
    fabric = LinkFabric(settings, engine)
    fabric.transfer_delivered.connect(ledger.on_transfer)
    return fabric, ledger


def _send(engine, fabric, path, payload_bits, attribution):
    """Send now and run the engine to the delivery; the transfer comes back."""
    delivered = []
    fabric.send(path, payload_bits, engine.now, attribution, delivered.append)
    engine.run()
    return delivered[0]


def test_legacy_cards_leave_optional_cwb_absent_without_changing_edge_identity():
    legacy = logical_reference_profile()
    legacy_edges = {
        path.value: getattr(legacy, path.value) for path in legacy.wired_paths()
    }

    extended = with_controller_to_weak_buffer_path(
        legacy,
        latency_microseconds=0.25,
        aggregate_bits_per_microsecond=100.0,
        source="Q-062 test card",
    )

    assert legacy.controller_to_weak_buffer is None
    assert LinkPath.CONTROLLER_TO_WEAK_BUFFER not in legacy.wired_paths()
    assert not LinkFabric(legacy, Engine()).is_wired(
        LinkPath.CONTROLLER_TO_WEAK_BUFFER
    )
    assert extended.controller_to_weak_buffer is not None
    assert LinkPath.CONTROLLER_TO_WEAK_BUFFER in extended.wired_paths()
    for path_name, edge in legacy_edges.items():
        assert getattr(extended, path_name) is edge


def test_wired_cwb_card_preserves_positive_numbers_source_and_physical_topology():
    source = "explicit Q-062 PROJECT_DESIGN latency and bandwidth"
    settings = _wired_profile(latency_us=0.25, bandwidth=120.0, source=source)
    _fabric, ledger = _fabric_and_ledger(settings, Engine())
    traffic = ledger.traffic_json_value()
    cwb_edge = next(
        edge
        for edge in traffic["semantic_edges"]
        if edge["path"] == "controller_to_weak_buffer"
    )
    channel = next(
        row
        for row in traffic["physical_channels"]
        if row["physical_alias"] == cwb_edge["physical_alias"]
    )

    assert traffic["path_order"].count("controller_to_weak_buffer") == 1
    assert channel["member_paths"] == ["controller_to_weak_buffer"]
    assert (
        settings.controller_to_weak_buffer.channel.propagation_latency_ticks
        == microseconds_to_ticks(0.25)
    )
    assert (
        settings.controller_to_weak_buffer.channel.capacity.aggregate_bits_per_microsecond
        == 120.0
    )
    assert (
        settings.controller_to_weak_buffer.channel.configuration_source
        == source
    )
    assert (
        settings.controller_to_weak_buffer.actual_payload_source
        == "SyndromeRoundPacket.fragment_size_sum"
    )


def test_cwb_traffic_uses_exact_round_attribution_payload_and_fifo_delays():
    engine = Engine()
    model, ledger = _fabric_and_ledger(
        _wired_profile(latency_us=0.25, bandwidth=100.0), engine
    )
    attribution = TransferAttribution(
        operation_id=7,
        patch_ids=(2, 9),
        window_id=None,
        first_round=11,
        last_round=11,
    )

    delivered = []
    model.send(
        LinkPath.CONTROLLER_TO_WEAK_BUFFER,
        300,
        10,
        attribution,
        delivered.append,
    )
    model.send(
        LinkPath.CONTROLLER_TO_WEAK_BUFFER,
        200,
        10,
        TransferAttribution(
            operation_id=7,
            patch_ids=(2, 9),
            window_id=None,
            first_round=12,
            last_round=12,
        ),
        delivered.append,
    )
    engine.run()
    first, second = delivered
    traffic = ledger.traffic_json_value()
    rows = [
        row
        for row in traffic["transfers"]
        if row["path"] == "controller_to_weak_buffer"
    ]

    assert first.serialization_ticks == microseconds_to_ticks(3.0)
    assert first.propagation_ticks == microseconds_to_ticks(0.25)
    assert first.total_delay_ticks == microseconds_to_ticks(3.25)
    assert second.queue_wait_ticks == first.serialization_ticks
    assert [row["payload_bits"] for row in rows] == [300, 200]
    assert rows[0]["payload_source"] == "SyndromeRoundPacket.fragment_size_sum"
    attribution_row = rows[0]["attribution"]
    assert attribution_row["operation_id"]["value"] == "7"
    assert [p["value"] for p in attribution_row["patch_ids"]] == ["2", "9"]
    assert (
        attribution_row["window_id"],
        attribution_row["round_lo"],
        attribution_row["round_hi"],
        attribution_row["relation"],
    ) == (None, 11, 11, None)
    assert all(row["reconciles"] for row in traffic["reconciliation"])


def test_finite_cwb_bandwidth_charges_serialization_plus_propagation():
    engine = Engine()
    model = LinkFabric(
        _wired_profile(latency_us=0.10, bandwidth=1000.0), engine
    )

    reservation = _send(
        engine,
        model,
        LinkPath.CONTROLLER_TO_WEAK_BUFFER,
        500,
        TransferAttribution(
            operation_id=1,
            patch_ids=(0,),
            window_id=None,
            first_round=1,
            last_round=1,
        ),
    )

    assert reservation.serialization_ticks == microseconds_to_ticks(
        0.5
    )  # 500 bits at 1000 bits/us
    assert reservation.propagation_ticks == microseconds_to_ticks(0.10)
    assert reservation.total_delay_ticks == microseconds_to_ticks(0.60)


def test_unbounded_cwb_bandwidth_charges_propagation_only():
    engine = Engine()
    model = LinkFabric(_wired_profile(latency_us=0.10, bandwidth=None), engine)

    delivered = []
    model.send(
        LinkPath.CONTROLLER_TO_WEAK_BUFFER,
        500,
        0,
        TransferAttribution(
            operation_id=1,
            patch_ids=(0,),
            window_id=None,
            first_round=1,
            last_round=1,
        ),
        delivered.append,
    )
    model.send(
        LinkPath.CONTROLLER_TO_WEAK_BUFFER,
        500,
        0,
        TransferAttribution(
            operation_id=1,
            patch_ids=(0,),
            window_id=None,
            first_round=2,
            last_round=2,
        ),
        delivered.append,
    )
    engine.run()
    first, second = delivered

    assert first.serialization_ticks == 0
    assert first.total_delay_ticks == microseconds_to_ticks(0.10)
    assert second.queue_wait_ticks == 0  # no serialization, so no FIFO wait
    assert second.total_delay_ticks == microseconds_to_ticks(0.10)


def test_rounds_pipeline_on_cwb_instead_of_stop_and_wait():
    """Fast rounds arrive at Buffer 0 spaced by the round period, not by the CWB
    latency: the link serializes them FIFO and propagation is pipelined."""
    stim = pytest.importorskip("stim")
    from decsim.config import TICKS_PER_MICROSECOND
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.decoders.settings import DecoderSettings
    from decsim.frontends.settings import WorkloadSettings
    from decsim.machine import Machine, MachineSettings
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.qpu.settings import QpuSettings
    from decsim.qpu.stim_device import StimDevice

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=8,
        distance=3,
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
        before_round_data_depolarization=0.001,
    )
    op = Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    settings = MachineSettings(
        workload=WorkloadSettings(
            operations=[op], rounds_policy=FixedRounds(8)
        ),
        qpu=QpuSettings(
            distance=3, round_period_microseconds=0.02, device=StimDevice()
        ),
        weak_decoder=DecoderSettings(decoder=PresetLatencyDecoder(0.01)),
        links=with_controller_to_weak_buffer_path(
            logical_reference_profile(),
            latency_microseconds=0.25,
            aggregate_bits_per_microsecond=10_000.0,
            source="test",
        ),
    )
    completed = Machine.build(settings, 0)
    result = completed.run()
    delivered = sorted(
        row["delivery_ticks"]
        for row in result.link_traffic["transfers"]
        if row["path"] == "controller_to_weak_buffer"
    )
    assert len(delivered) == 8
    gaps = [
        (b - a) / TICKS_PER_MICROSECOND
        for a, b in zip(delivered, delivered[1:])
    ]
    assert all(gap == pytest.approx(0.02, abs=1e-3) for gap in gaps), (
        gaps
    )  # first gap adds serialization
