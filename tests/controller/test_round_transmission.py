"""The transmitter: a stored round leaves at the write and lands on its route.

A window-input round rides controller_to_weak_buffer and reaches Buffer
0's incoming port at delivery, which publishes it. A feedback-memory
round tells the windows at delivery and frees its slot. The sender never
waits for a landing:
two memory rounds one QEC cycle apart on a 5 us weak_buffer_to_weak_decoder
land one cycle apart (Yang et al. 2605.04892 and Google 2408.13687 stream
every round; gem5 src/dev/dma_device.cc transmitList; ns-3
point-to-point-net-device.cc TransmitComplete). Rounds sent at one tick
leave in completion order, each at its own write, with no arbitration
between the routes: the memory round of the reviewed bounded
weak_buffer_to_weak_decoder shape serializes ahead of the decode input
the windows send one store hop later. The link law itself is the
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
import decsim.links.window_transfers as window_transfers
import decsim.observe.link_traffic as link_traffic
import decsim.observe.round_events as round_events
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.syndrome_buffer.round_input as round_input
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings

CWB_TICKS = config.microseconds_to_ticks(0.25)
WBD_TICKS = config.microseconds_to_ticks(5.0)
CYCLE_TICKS = config.microseconds_to_ticks(1.0)
MEMORY_ROUTE = round_records.SyndromePacketRoute.feedback_memory_round(7)
WBD_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
# 64 bits at 1000 bits per microsecond, one serialization on the wire
ROUND_BITS = 64
SERIALIZATION_TICKS = config.microseconds_to_ticks(0.064)
# the reference card's controller_to_weak_buffer, Caune's 40 ns stage
REFERENCE_CWB_TICKS = config.microseconds_to_ticks(0.04)


class RecordingWindows:
    """The window side and the decoders' memory end, in one recorder."""

    def __init__(self, engine):
        self.engine = engine
        self.published = []
        self.memory_rounds = []

    def accept_window_input(self, packet):
        self.published.append((self.engine.now, packet.round_index))

    def receive_memory_round(self, source_operation_id):
        self.memory_rounds.append((self.engine.now, source_operation_id))


class DispatchingWindows:
    """Windows that send a decode input on the wire at a publication."""

    def __init__(self, engine, link):
        self.engine = engine
        self.link = link
        self.published = []
        self.memory_rounds = []
        self.decode_inputs_delivered = []

    def receive_memory_round(self, source_operation_id):
        self.memory_rounds.append((self.engine.now, source_operation_id))

    def accept_window_input(self, packet):
        self.published.append((self.engine.now, packet.round_index))
        attribution = transfer_records.TransferAttribution.for_round(1, (0,), 9)
        self.link.send(
            WBD_PATH,
            ROUND_BITS,
            self.engine.now,
            attribution,
            self.decode_inputs_delivered.append,
        )


def packed(round_index, route=round_records.WINDOW_INPUT_ROUTE, wire_bits=2):
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(packet, route, wire_bits)


def priced_cwb_profile():
    reference = link_profiles.logical_reference_profile()
    base = reference.controller_to_weak_buffer
    channel = link_settings.ChannelSettings(
        base.channel.name, CWB_TICKS, None, "test"
    )
    path = link_settings.PathSettings(
        channel, base.default_payload, base.actual_payload_source
    )
    return dataclasses.replace(reference, controller_to_weak_buffer=path)


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
    links.trace.transfer_delivered.connect(ledger.on_transfer)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings)
    if windows is None:
        windows = RecordingWindows(engine)
    else:
        windows = windows(engine, links)
    recorder = round_events.RoundEventRecorder(engine)
    transfers = window_transfers.WindowTransfers(engine, links)
    store_output = round_output.RoundStoreOutput(
        transfers,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "Buffer 0",
        store,
    )
    store_input = round_input.RoundStoreInput(
        engine, store, store_output, windows
    )
    transmitter = round_transmission.RoundTransmitter(
        engine, links, windows, store_input
    )
    transmitter.trace.round_event.connect(recorder.record)
    store_input.trace.round_event.connect(recorder.record)
    return transmitter, store, windows, recorder, ledger


def test_a_priced_hop_publishes_at_delivery_and_stamps_the_store():
    engine = engine_module.Engine()
    profile = priced_cwb_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1)
    store.accept_packed_round(first.packet, publication_tick=None)

    transmitter.send(first)
    engine.run()

    assert store.publication_tick((1, 1)) == CWB_TICKS
    assert windows.published == [(CWB_TICKS, 1)]
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("CWB_SENT", 0), ("PUBLISHED", CWB_TICKS)]
    assert transmitter.in_flight == 0


def test_a_memory_round_tells_the_windows_at_delivery_and_frees_its_slot():
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    memory_round = packed(1, route=MEMORY_ROUTE)
    store.accept_packed_round(memory_round.packet, publication_tick=None)

    transmitter.send(memory_round)
    engine.run()

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
    "memory_send_ticks, memory_delivery, input_delivery, wire_order",
    [
        (0, 5_064_000, 5_128_000, [1, 9]),
        (REFERENCE_CWB_TICKS, 5_168_000, 5_104_000, [9, 1]),
    ],
)
def test_two_routes_take_one_wire_in_the_order_they_reach_it(
    memory_send_ticks, memory_delivery, input_delivery, wire_order
):
    """The round that reaches the shared wire first serializes first.

    A memory round and a decode input share a bandwidth-bounded
    weak_buffer_to_weak_decoder (5 us, 1000 bits per microsecond,
    64-bit rounds, so 0.064 us on the wire). The window round crosses
    controller_to_weak_buffer into Buffer 0 first, so the decode input
    the windows send at its publication reaches the shared wire one
    store hop after the round was sent: a memory round sent with the
    window round is ahead of it, and one sent at the publication tick
    is behind it. gem5's DmaPort queues at the request
    (src/dev/dma_device.cc transmitList) and ns-3's device starts the
    next packet at TransmitComplete (point-to-point-net-device.cc): no
    arbitration event between the routes.
    """
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile(bits_per_microsecond=1000.0)
    transmitter, store, windows, _recorder, ledger = transmitter_with(
        engine, profile, windows=DispatchingWindows
    )
    memory_round = packed(1, route=MEMORY_ROUTE, wire_bits=ROUND_BITS)
    window_round = packed(2, wire_bits=ROUND_BITS)

    def send(finished):
        store.accept_packed_round(finished.packet, publication_tick=None)
        transmitter.send(finished)

    engine.schedule(0, lambda: send(window_round))
    engine.schedule(memory_send_ticks, lambda: send(memory_round))
    engine.run()

    assert windows.memory_rounds == [(memory_delivery, 7)]
    (decode_input,) = windows.decode_inputs_delivered
    assert decode_input.delivery_ticks == input_delivery
    traffic = ledger.traffic_json_value()
    rounds_on_the_wire = []
    for transfer in traffic["transfers"]:
        if transfer["path"] != "weak_buffer_to_weak_decoder":
            continue
        round_lo = transfer["attribution"]["round_lo"]
        rounds_on_the_wire.append(round_lo)
    assert rounds_on_the_wire == wire_order
