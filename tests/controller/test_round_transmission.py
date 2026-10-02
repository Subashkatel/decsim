"""The transmitter: a stored round leaves at the write and lands on its route.

A window-input round rides controller_to_weak_buffer and reaches Buffer
0's incoming port at delivery, which publishes it. A feedback-memory
round rides the same hop into the store, as the strong route's rounds
ride controller_to_strong_buffer, then tells the windows at its delivery
to the decoder and frees its slot. The sender never waits for a landing:
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
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.links.window_transfers as window_transfers
import decsim.observe.link_traffic as link_traffic
import decsim.observe.round_events as round_events
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_syndrome_round_receiver,
)

CWB_TICKS = config.microseconds_to_ticks(0.25)
WBD_TICKS = config.microseconds_to_ticks(5.0)
CYCLE_TICKS = config.microseconds_to_ticks(1.0)
MEMORY_ROUTE = round_records.SyndromePacketRoute.feedback_memory_round(7)
WBD_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
# 64 bits at 1000 bits per microsecond, one serialization on the wire
ROUND_BITS = 64
SERIALIZATION_TICKS = config.microseconds_to_ticks(0.064)
# a pure-delay controller_to_weak_buffer, so a window round publishes
# one fixed store hop after it is sent
STORE_HOP_TICKS = config.microseconds_to_ticks(0.04)


class RecordingPackingLine:
    """The line in front of the packing stage, keeping each retry's count."""

    def __init__(self, transmitter=None) -> None:
        self.transmitter = transmitter
        self.in_flight_at_retry = []

    def retry(self) -> None:
        self.in_flight_at_retry.append(self.transmitter.in_flight)


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
        patch_ids=(0,),
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
        channel,
        base.default_payload,
        base.actual_payload_source,
        excludes_receiver_processing=base.excludes_receiver_processing,
    )
    return dataclasses.replace(reference, controller_to_weak_buffer=path)


def five_microsecond_wbd_profile(bits_per_microsecond=None):
    """The reference card with a 5 us weak_buffer_to_weak_decoder.

    With a rate the channel is bandwidth bounded (reviewer A's bounded
    wbd shape); without one it is a pure delay. The store hop in front
    of it is a pure delay of STORE_HOP_TICKS.
    """
    reference = link_profiles.logical_reference_profile()
    edge = reference.weak_buffer_to_weak_decoder
    capacity = None
    if bits_per_microsecond is not None:
        capacity = link_settings.CapacitySettings(bits_per_microsecond, "test")
    channel = link_settings.ChannelSettings(
        edge.channel.name, WBD_TICKS, capacity, "test"
    )
    path = link_settings.PathSettings(
        channel,
        edge.default_payload,
        edge.actual_payload_source,
        excludes_receiver_processing=edge.excludes_receiver_processing,
    )
    store_hop = reference.controller_to_weak_buffer
    store_channel = link_settings.ChannelSettings(
        store_hop.channel.name, STORE_HOP_TICKS, None, "test"
    )
    store_path = dataclasses.replace(store_hop, channel=store_channel)
    return dataclasses.replace(
        reference,
        weak_buffer_to_weak_decoder=path,
        controller_to_weak_buffer=store_path,
    )


def transmitter_with(engine, profile, windows=None, settings=None):
    ledger = link_traffic.TrafficLedger(profile)
    links = fabric_module.LinkFabric(profile, engine)
    links.trace.transfer_delivered.connect(ledger.on_transfer)
    if settings is None:
        settings = syndrome_buffer_module.SyndromeBufferSettings()
    store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    if windows is None:
        windows = RecordingWindows(engine)
    else:
        windows = windows(engine, links)
    recorder = round_events.RoundEventRecorder(engine)
    transfers = window_transfers.WindowTransfers(engine)
    transfers.link = links
    store_output = round_output.SyndromeBufferOutput(
        engine,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "weak syndrome buffer",
    )
    store_output.transfers = transfers
    store_output.store = store
    store_output.link = links
    weak_receiver = weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(
        engine
    )
    weak_receiver.store = store
    weak_receiver.output = store_output
    weak_receiver.windows = windows
    at_the_controller = event_settings.DetectionEventSettings()
    weak_receiver.detection_events = formation.SeatedFormation(
        None, at_the_controller
    )
    transmitter = round_transmission.RoundTransmitter(engine)
    transmitter.link = links
    transmitter.memory_arrivals = windows
    transmitter.weak_receiver = weak_receiver
    transmitter.packing_line = RecordingPackingLine(transmitter)
    transmitter.trace.round_event.connect(recorder.record)
    weak_receiver.trace.round_event.connect(recorder.record)
    return transmitter, store, windows, recorder, ledger


def reserve_and_send(transmitter, packed) -> None:
    """The sender's two steps: the room is reserved, then the round leaves."""
    transmitter.weak_receiver.reserve_write(packed)
    transmitter.send(packed)


def test_a_priced_hop_publishes_at_delivery_and_stamps_the_store():
    engine = engine_module.Engine()
    profile = priced_cwb_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1)

    reserve_and_send(transmitter, first)
    engine.run()

    assert store.publication_tick((1, 1)) == CWB_TICKS
    assert windows.published == [(CWB_TICKS, 1)]
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("CWB_SENT", 0), ("PUBLISHED", CWB_TICKS)]
    assert transmitter.in_flight == 0


def test_a_round_that_leaves_its_route_lets_a_waiting_round_enter():
    """The stage's room frees when the windows hear of the round.

    gem5's responder calls sendRetryReq once it can take the request it
    refused (src/mem/port.hh:244-262), so the line is retried after the
    count drops, once per round.
    """
    engine = engine_module.Engine()
    profile = priced_cwb_profile()
    transmitter, _store, _windows, _recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1)

    reserve_and_send(transmitter, first)
    engine.run()

    assert transmitter.packing_line.in_flight_at_retry == [0]


def test_a_window_round_is_in_flight_until_its_write_publishes_it():
    """The packing stage's bound holds a round until the windows hear of it.

    A five-cycle write on a 1 MHz clock publishes the round microseconds
    after its 0.25 us landing, and the count holds it until then.
    """
    engine = engine_module.Engine()
    profile = priced_cwb_profile()
    clock = config.Clock(CYCLE_TICKS)
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        clock=clock, write_cycles=5
    )
    transmitter, _store, windows, _recorder, _ledger = transmitter_with(
        engine, profile, settings=settings
    )
    counts_after_the_landing = []
    first = packed(1)
    probe_ticks = CWB_TICKS + 1

    reserve_and_send(transmitter, first)
    engine.schedule(
        probe_ticks,
        lambda: counts_after_the_landing.append(transmitter.in_flight),
    )
    engine.run()
    publication_tick, _round_index = windows.published[0]

    assert counts_after_the_landing == [1]
    assert publication_tick > probe_ticks
    assert transmitter.in_flight == 0


def test_a_memory_round_crosses_the_store_hop_then_tells_the_windows():
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, recorder, _ledger = transmitter_with(
        engine, profile
    )
    memory_round = packed(1, route=MEMORY_ROUTE)

    reserve_and_send(transmitter, memory_round)
    engine.run()

    delivery_ticks = STORE_HOP_TICKS + WBD_TICKS
    assert windows.memory_rounds == [(delivery_ticks, 7)]
    assert store.occupancy == 0
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("FEEDBACK_MEMORY_DELIVERED", delivery_ticks)]


def test_memory_rounds_pipeline_onto_the_link_without_a_landing_wait():
    engine = engine_module.Engine()
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, _recorder, _ledger = transmitter_with(
        engine, profile
    )
    first = packed(1, route=MEMORY_ROUTE)
    second = packed(2, route=MEMORY_ROUTE)

    def send(memory_round):
        reserve_and_send(transmitter, memory_round)

    engine.schedule(0, lambda: send(first))
    engine.schedule(CYCLE_TICKS, lambda: send(second))
    engine.run()

    delivery_ticks = STORE_HOP_TICKS + WBD_TICKS
    assert windows.memory_rounds == [
        (delivery_ticks, 7),
        (CYCLE_TICKS + delivery_ticks, 7),
    ]


@pytest.mark.parametrize(
    "memory_send_ticks, memory_delivery, input_delivery, wire_order",
    [
        (0, 5_104_000, 5_168_000, [1, 9]),
        (STORE_HOP_TICKS, 5_168_000, 5_104_000, [9, 1]),
    ],
)
def test_two_routes_take_one_wire_in_the_order_they_reach_it(
    memory_send_ticks, memory_delivery, input_delivery, wire_order
):
    """The round that reaches the shared wire first serializes first.

    A memory round and a decode input share a bandwidth-bounded
    weak_buffer_to_weak_decoder (5 us, 1000 bits per microsecond,
    64-bit rounds, so 0.064 us on the wire). Both rounds cross
    controller_to_weak_buffer into the weak syndrome buffer first, and the
    decode input the windows send at the window round's publication
    reaches the shared wire one store hop after that round was sent: a
    memory round sent just before the window round is ahead of it, and
    one sent at the publication tick is behind it. gem5's DmaPort
    queues at the request (src/dev/dma_device.cc transmitList) and
    ns-3's device starts the next packet at TransmitComplete
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

    def send(finished):
        reserve_and_send(transmitter, finished)

    engine.schedule(memory_send_ticks, lambda: send(memory_round))
    engine.schedule(0, lambda: send(window_round))
    engine.run()

    assert windows.memory_rounds == [(memory_delivery, 7)]
    (decode_input,) = windows.decode_inputs_delivered
    assert decode_input.delivery_ticks == input_delivery
    traffic = ledger.traffic_json_value()
    rounds_on_the_wire = []
    for transfer in traffic["transfers"]:
        if transfer["path"] != "weak_buffer_to_weak_decoder":
            continue
        (rounds,) = transfer["attribution"]["rounds_by_operation"]
        rounds_on_the_wire.append(rounds["round_lo"])
    assert rounds_on_the_wire == wire_order
