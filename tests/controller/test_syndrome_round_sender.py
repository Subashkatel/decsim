"""The sender: a finished round leaves for every store it reaches, or waits.

The backpressure law: each store's own end answers has_room before any
round leaves for it, counting the rounds it holds and the writes it has
in flight, and the room is reserved before the wire is used (gem5's
queue counts its reserved entries as taken,
tmp/resources/gem5/src/mem/cache/queue.hh:150-153 isFull with the
reserve at :87-93; Ruby MessageBuffer.cc:181 sums the same two counts).
A round that finds no room waits in HeldRounds and enters in completion
order when a slot frees (gem5 src/mem/cache/base.cc:255-257 setBlocked,
:266-271 clearBlocked and the retry; Ciw
tmp/resources/l5_buffers/Ciw/ciw/node.py:470-473
release_blocked_individual), or is dropped under the drop knob (ns-3
point-to-point-net-device.cc Send: Enqueue false, packet dropped). A
strong-primary plan takes one hop, into the strong store; a
feedback-memory round still takes a weak syndrome buffer slot. A
weak-primary plan's round goes into the weak syndrome buffer only:
what the strong tier needs of it rides the escalation (Battistel
2303.00054 lines 342 to 347, a cold first stage keeps the rounds off
the cryostat I/O).

The controller is the end a strong-primary run's round leaves by, so it
executes the controller_to_strong_buffer send and the room side handles
the landing (OMNeT++ csimplemodule.cc:333-334, a module sends only what
it owns; gem5 packet.hh:424-431, a transfer is billed to the port it
left by). The crossing's own delay is the card's, on the reference
profile.
"""

import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.observe.round_events as round_events
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver,
)
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_syndrome_round_receiver,
)

STALL = controller_settings.PackingOverflowPolicy.STALL
DROP = controller_settings.PackingOverflowPolicy.DROP_ROUND
MEMORY_ROUTE = round_records.SyndromePacketRoute.feedback_memory_round(9)


def packed(round_index, route=round_records.WINDOW_INPUT_ROUTE):
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(packet, route, 2)


class RecordingTransmitter:
    def __init__(self, engine):
        self.engine = engine
        self.sent = []

    def send(self, round):
        self.sent.append(round.packet.round_index)


class RecordingWindows:
    """The window side hears each round the weak syndrome buffer publishes."""

    def __init__(self):
        self.published = []

    def accept_window_input(self, packet):
        self.published.append(packet.round_index)


class RecordingStrongReceiver:
    """The room side of the crossing, recording what reaches it."""

    def __init__(self, room=True):
        self.room = room
        self.reserved = 0
        self.written = []

    def has_room(self):
        return self.room

    def reserve_write(self):
        self.reserved += 1

    def receive_round(self, packet, packet_bits):
        del packet_bits
        self.written.append(packet.round_index)


def sender_with(
    engine,
    weak_rounds=None,
    strong_receiver=None,
    publishes_from_strong_store=False,
    on_full=STALL,
):
    recorder = round_events.RoundEventRecorder(engine)
    held = syndrome_round_sender.HeldRounds(engine, on_full)
    held.trace.round_event.connect(recorder.record)
    settings = syndrome_buffer_settings.SyndromeBufferSettings(
        rounds=weak_rounds
    )
    weak_store = syndrome_buffer_module.SyndromeBuffer(settings)
    weak_store.held_rounds = held
    transmitter = RecordingTransmitter(engine)
    profile = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(profile, engine)
    windows = RecordingWindows()
    weak_receiver = weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(
        engine, settings
    )
    weak_receiver.store = weak_store
    weak_receiver.windows = windows
    sender = syndrome_round_sender.SyndromeRoundSender(engine)
    sender.link = links
    sender.weak_receiver = weak_receiver
    sender.weak_store = weak_store
    if strong_receiver is not None:
        sender.strong_receiver = strong_receiver
    sender.held_rounds = held
    sender.transmitter = transmitter
    sender.publishes_from_strong_store = publishes_from_strong_store
    return sender, weak_receiver, transmitter, recorder


def test_a_round_with_no_room_is_held_and_written_in_order_when_a_slot_frees():
    """One slot, taken by a round in flight: the next waits for it to free.

    The room a crossing round will need is spent the moment it leaves,
    so a weak syndrome buffer of one slot refuses the second round while the
    first is still on controller_to_weak_buffer, and still refuses it once the
    first lands and holds the slot. The slot frees, and the head of the
    waiting line enters.
    """
    engine = engine_module.Engine()
    sender, weak_receiver, transmitter, recorder = sender_with(
        engine, weak_rounds=1
    )
    first = packed(1)
    second = packed(2)

    first_admitted = sender.admit(first)
    second_admitted = sender.admit(second)
    held_while_in_flight = sender.held_rounds.count
    weak_receiver.receive_round(first)
    held_after_the_landing = sender.held_rounds.count
    weak_receiver.store.release_round((1, 1))

    assert first_admitted is True
    assert second_admitted is False
    assert held_while_in_flight == 1
    assert held_after_the_landing == 1
    assert transmitter.sent == [1, 2]
    assert sender.held_rounds.count == 0
    stalled = [
        event.round_index
        for event in recorder.events
        if event.kind == "STALLED"
    ]
    assert stalled == [2]


def test_a_strong_primary_window_round_takes_one_hop_into_the_strong_store():
    engine = engine_module.Engine()
    strong_receiver = RecordingStrongReceiver()
    sender, weak_receiver, transmitter, _recorder = sender_with(
        engine,
        strong_receiver=strong_receiver,
        publishes_from_strong_store=True,
    )
    window_round = packed(1)
    memory_round = packed(2, route=MEMORY_ROUTE)

    sender.admit(window_round)
    sender.admit(memory_round)
    engine.run()

    assert strong_receiver.written == [1, 2]
    assert weak_receiver.writes_in_flight == 1
    assert transmitter.sent == [2]


def test_the_controller_carries_the_round_to_the_room_side_and_lands_it():
    """The send is the controller's; the landing is the room side's.

    The reference card prices controller_to_strong_buffer at 0.26
    microseconds (Caune 2410.05202 Fig. 1a stage F), so a strong-primary
    run's round is reserved and sent at the write and stored one card
    delay later, by the room side's own method.
    """
    engine = engine_module.Engine()
    store_settings = syndrome_buffer_settings.SyndromeBufferSettings()
    strong_store = syndrome_buffer_module.SyndromeBuffer(store_settings)
    reads = decoding_records.WindowReads((1, 0))
    strong_store.register_hold(reads, [(1, 1)])
    room_side = strong_syndrome_round_receiver.StrongSyndromeRoundReceiver(
        engine
    )
    room_side.store = strong_store
    sender, _weak_store, _transmitter, _recorder = sender_with(
        engine, strong_receiver=room_side, publishes_from_strong_store=True
    )
    first = packed(1)

    sender.admit(first)
    reserved_while_crossing = room_side.writes_in_flight
    stored_at_the_write = strong_store.occupancy
    engine.run()

    crossing_ticks = config.microseconds_to_ticks(0.26)
    assert reserved_while_crossing == 1
    assert stored_at_the_write == 0
    assert strong_store.occupancy == 1
    assert strong_store.publication_tick((1, 1)) == crossing_ticks
    assert room_side.writes_in_flight == 0


def test_a_weak_primary_round_never_leaves_for_the_strong_store():
    """The strong side gets an escalated window's rounds, not every round."""
    engine = engine_module.Engine()
    strong_receiver = RecordingStrongReceiver(room=False)
    sender, weak_receiver, transmitter, _recorder = sender_with(
        engine, strong_receiver=strong_receiver
    )
    first = packed(1)

    admitted = sender.admit(first)
    engine.run()

    assert admitted is True
    assert strong_receiver.written == []
    assert strong_receiver.reserved == 0
    assert weak_receiver.writes_in_flight == 1
    assert transmitter.sent == [1]


def test_a_strong_primary_round_with_no_room_on_the_room_side_is_held():
    engine = engine_module.Engine()
    strong_receiver = RecordingStrongReceiver(room=False)
    sender, weak_receiver, transmitter, _recorder = sender_with(
        engine,
        strong_receiver=strong_receiver,
        publishes_from_strong_store=True,
    )
    first = packed(1)

    admitted = sender.admit(first)

    assert admitted is False
    assert weak_receiver.store.occupancy == 0
    assert weak_receiver.writes_in_flight == 0
    assert transmitter.sent == []
    assert sender.held_rounds.count == 1


def test_the_drop_knob_drops_a_round_that_found_no_room():
    engine = engine_module.Engine()
    sender, _weak_input, transmitter, recorder = sender_with(
        engine, weak_rounds=1, on_full=DROP
    )
    first = packed(1)
    second = packed(2)

    sender.admit(first)
    sender.admit(second)

    assert transmitter.sent == [1]
    assert recorder.packing_drops == 1
    assert sender.held_rounds.count == 0
