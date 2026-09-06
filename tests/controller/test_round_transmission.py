"""The transmitter: a stored round leaves at the write and lands on its route.

A window-input round on a priced controller_to_weak_buffer hop is
published at delivery, where the store is stamped; on a free hop the
publication is the storage. A feedback-memory round tells the windows at
delivery and frees its slot. The sender never waits for a landing:
two memory rounds one QEC cycle apart on a 5 us weak_buffer_to_weak_decoder
land one cycle apart (Yang et al. 2605.04892 and Google 2408.13687 stream
every round; gem5 src/dev/dma_device.cc transmitList; ns-3
point-to-point-net-device.cc TransmitComplete). Rounds sent at one tick
leave in completion order, each at its own write, with no arbitration
between the routes: the memory round of the reviewed bounded
weak_buffer_to_weak_decoder shape serializes ahead of the decode input
the windows send at its publication. The link law itself is the
channel's (tests/links/test_channel.py).
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.controller.round_transmission as round_transmission
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.message as message
import decsim.observe.link_traffic as link_traffic
import decsim.observe.round_events as round_events
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings

CWB_TICKS = config.microseconds_to_ticks(0.25)
WBD_TICKS = config.microseconds_to_ticks(5.0)
CYCLE_TICKS = config.microseconds_to_ticks(1.0)
MEMORY_ROUTE = message.SyndromePacketRoute.feedback_memory_round(7)
WBD_PATH = message.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
# 64 bits at 1000 bits per microsecond, one serialization on the wire
ROUND_BITS = 64
SERIALIZATION_TICKS = config.microseconds_to_ticks(0.064)


class RecordingWindows:
    def __init__(self, engine):
        self.engine = engine
        self.published = []
        self.memory_rounds = []

    def accept_window_input(self, packet):
        self.published.append((self.engine.now, packet.round_index))

    def accept_feedback_memory_round(self, source_operation_id):
        self.memory_rounds.append((self.engine.now, source_operation_id))


class DispatchingWindows:
    """Windows that send a decode input on the wire at a publication."""

    def __init__(self, engine, link):
        self.engine = engine
        self.link = link
        self.published = []
        self.memory_rounds = []
        self.decode_inputs_delivered = []

    def accept_feedback_memory_round(self, source_operation_id):
        self.memory_rounds.append((self.engine.now, source_operation_id))

    def accept_window_input(self, packet):
        self.published.append((self.engine.now, packet.round_index))
        attribution = message.TransferAttribution.for_round(1, (0,), 9)
        self.link.send(
            WBD_PATH,
            ROUND_BITS,
            self.engine.now,
            attribution,
            self.decode_inputs_delivered.append,
        )


def packed(round_index, route=message.WINDOW_INPUT_ROUTE, wire_bits=2):
    fragment = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = message.SyndromeRoundPacket(1, round_index, (fragment,))
    return message.PackedRound(packet, route, wire_bits)


def priced_cwb_profile():
    reference = link_profiles.logical_reference_profile()
    return link_profiles.with_controller_to_weak_buffer_path(
        reference,
        latency_microseconds=0.25,
        aggregate_bits_per_microsecond=None,
        source="test",
    )


def five_microsecond_wbd_profile(bits_per_microsecond=None):
    """The reference card with a 5 us weak_buffer_to_weak_decoder.

    With a rate the channel is bandwidth bounded (reviewer A's bounded
    wbd shape); without one it is a pure delay.
    """
    reference = link_profiles.logical_reference_profile()
    edge = reference.weak_buffer_to_weak_decoder
    capacity = None
    if bits_per_microsecond is not None:
        capacity = link_settings.CapacitySettings(
            bits_per_microsecond,
            link_settings.QuantityBasis.AGGREGATE,
            None,
            "test",
        )
    channel = link_settings.ChannelSettings(
        edge.channel.name, WBD_TICKS, capacity, "test"
    )
    path = link_settings.PathSettings(
        channel, edge.default_payload, edge.actual_payload_source
    )
    return dataclasses.replace(reference, weak_buffer_to_weak_decoder=path)


def transmitter_with(engine, profile, windows=None):
    ledger = link_traffic.TrafficLedger(profile)
    links = fabric_module.LinkFabric(profile, engine)
    links.transfer_delivered.connect(ledger.on_transfer)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings)
    if windows is None:
        windows = RecordingWindows(engine)
    else:
        windows = windows(engine, links)
    recorder = round_events.RoundEventRecorder(engine)
    transmitter = round_transmission.RoundTransmitter(
        engine, links, store, windows
    )
    transmitter.round_event.connect(recorder.record)
    return transmitter, store, windows, recorder, ledger


def test_a_priced_hop_publishes_at_delivery_and_stamps_the_store():
    engine = engine_module.Engine()
    profile = priced_cwb_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1)
    stored_tick = transmitter.publication_tick_at_storage(first.route)
    store.accept_packed_round(first.packet, publication_tick=stored_tick)

    transmitter.send(first)
    engine.run()

    assert stored_tick is None
    assert store.publication_tick((1, 1)) == CWB_TICKS
    assert windows.published == [(CWB_TICKS, 1)]
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("CWB_SENT", 0), ("PUBLISHED", CWB_TICKS)]
    assert transmitter.in_flight == 0


def test_a_free_hop_publishes_as_the_round_is_stored():
    engine = engine_module.Engine()
    profile = link_profiles.logical_reference_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1)
    stored_tick = transmitter.publication_tick_at_storage(first.route)
    store.accept_packed_round(first.packet, publication_tick=stored_tick)

    transmitter.send(first)
    engine.run()

    assert stored_tick == 0
    assert windows.published == [(0, 1)]
    assert recorder.events == []


def test_a_memory_round_tells_the_windows_at_delivery_and_frees_its_slot():
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    memory_round = packed(1, route=MEMORY_ROUTE)
    stored_tick = transmitter.publication_tick_at_storage(memory_round.route)
    store.accept_packed_round(memory_round.packet, publication_tick=stored_tick)

    transmitter.send(memory_round)
    engine.run()

    assert stored_tick is None
    assert windows.memory_rounds == [(WBD_TICKS, 7)]
    assert store.occupancy == 0
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("FEEDBACK_MEMORY_DELIVERED", WBD_TICKS)]


def test_memory_rounds_pipeline_onto_the_link_without_a_landing_wait():
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, _recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1, route=MEMORY_ROUTE)
    second = packed(2, route=MEMORY_ROUTE)

    def send(memory_round):
        store.accept_packed_round(memory_round.packet, publication_tick=None)
        transmitter.send(memory_round)

    engine.schedule(0, lambda: send(first))
    engine.schedule(CYCLE_TICKS, lambda: send(second))
    engine.run()

    assert windows.memory_rounds == [
        (WBD_TICKS, 7),
        (CYCLE_TICKS + WBD_TICKS, 7),
    ]


@pytest.mark.parametrize(
    "first_route, memory_delivery, input_delivery, wire_order",
    [
        (MEMORY_ROUTE, 1, 2, [1, 9]),
        (message.WINDOW_INPUT_ROUTE, 2, 1, [9, 1]),
    ],
)
def test_rounds_sent_at_one_tick_leave_in_completion_order(
    first_route, memory_delivery, input_delivery, wire_order
):
    """The first of two rounds completing at one tick is on the wire first.

    A memory round and a window round complete at one tick on a
    bandwidth-bounded weak_buffer_to_weak_decoder (5 us, 1000 bits per
    microsecond, 64-bit rounds). The decode input the windows send at
    the window round's publication serializes behind the memory round
    when the memory round completed first, ahead of it otherwise. gem5's
    DmaPort queues at the request (src/dev/dma_device.cc transmitList)
    and ns-3's device starts the next packet at TransmitComplete
    (point-to-point-net-device.cc): no arbitration event between the
    routes.
    """
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile(bits_per_microsecond=1000.0)
    transmitter, store, windows, _recorder, ledger = transmitter_with(
        engine, profile, windows=DispatchingWindows
    )
    memory_round = packed(1, route=MEMORY_ROUTE, wire_bits=ROUND_BITS)
    window_round = packed(2, wire_bits=ROUND_BITS)
    completion_order = [window_round, memory_round]
    if first_route is MEMORY_ROUTE:
        completion_order = [memory_round, window_round]

    def complete_both():
        for finished in completion_order:
            store.accept_packed_round(finished.packet, publication_tick=None)
            transmitter.send(finished)

    engine.schedule(0, complete_both)
    engine.run()

    memory_tick = WBD_TICKS + memory_delivery * SERIALIZATION_TICKS
    input_tick = WBD_TICKS + input_delivery * SERIALIZATION_TICKS
    assert windows.memory_rounds == [(memory_tick, 7)]
    (decode_input,) = windows.decode_inputs_delivered
    assert decode_input.delivery_ticks == input_tick
    traffic = ledger.traffic_json_value()
    rounds_on_the_wire = [
        transfer["attribution"]["round_lo"] for transfer in traffic["transfers"]
    ]
    assert rounds_on_the_wire == wire_order
