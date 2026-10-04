"""The sender: a finished round leaves for every store it reaches, or waits.

The backpressure law: each store's own end answers has_room with the
round's bits before any round leaves for it, counting the bits it holds
and the bits reserved for the writes in flight, and the room is
reserved before the wire is used (gem5's packet store answers
`avail() = _maxsize - _size - _reserved` against the packet's own
length, gem5 src/dev/net/pktfifo.hh, and reserves it with
`reserve(len)`).
A round that finds no room waits in HeldRounds and enters in completion
order when a slot frees (gem5 src/mem/cache/base.cc:255-257 setBlocked,
:266-271 clearBlocked and the retry; Ciw
Ciw ciw/node.py:470-473
release_blocked_individual); nothing is dropped. A
strong-primary plan takes one hop into the strong store for both window
input and feedback-memory rounds. A
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

import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.observe.round_events as round_events
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver,
)
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_syndrome_round_receiver,
)

# every round this file sends carries one fragment of two bits
BITS_PER_ROUND = 2


def formed_at_the_controller():
    """The run's former seated at the controller, forming nothing here."""
    at_the_controller = event_settings.DetectionEventSettings()
    return formation.SeatedFormation(None, at_the_controller)


def packed(round_index, route=round_records.WINDOW_INPUT_ROUTE):
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0),
        size_bits=BITS_PER_ROUND,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(packet, route, BITS_PER_ROUND)


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
        self.reserved_bits = 0
        self.written = []

    def has_room(self, packed):
        del packed
        return self.room

    def reserve_write(self, packed):
        self.reserved += 1
        self.reserved_bits += packed.wire_bits

    def receive_round(self, packed: round_records.PackedRound) -> None:
        self.written.append(packed.packet.round_index)


def no_sender_counts_it() -> None:
    """The publication callback of a round no transmitter counts here."""


def sender_with(
    engine,
    weak_bits=None,
    strong_receiver=None,
    publishes_from_strong_store=False,
):
    recorder = round_events.RoundEventRecorder(engine)
    held = syndrome_round_sender.HeldRounds(engine)
    held.trace.round_event.connect(recorder.record)
    settings = syndrome_buffer_module.SyndromeBufferSettings(bits=weak_bits)
    weak_store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    weak_store.held_rounds = held
    transmitter = RecordingTransmitter(engine)
    profile = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(profile, engine)
    windows = RecordingWindows()
    weak_receiver = weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(
        engine
    )
    weak_receiver.store = weak_store
    weak_receiver.windows = windows
    weak_receiver.detection_events = formed_at_the_controller()
    sender = syndrome_round_sender.SyndromeRoundSender(engine)
    sender.link = links
    sender.weak_receiver = weak_receiver
    sender.weak_store = weak_store
    if strong_receiver is not None:
        sender.strong_receiver = strong_receiver
    sender.held_rounds = held
    sender.packing_line = syndrome_round_sender.HeldRounds(engine)
    sender.transmitter = transmitter
    sender.publishes_from_strong_store = publishes_from_strong_store
    return sender, weak_receiver, transmitter, recorder


def test_a_narrow_round_waits_behind_a_held_round_it_would_fit_beside():
    """A round that finds the line non-empty joins it, room or not.

    gem5's refused requester waits for the retry before it sends again
    (src/mem/port.hh:244-255) and its packet queue keeps later packets
    behind the refused front (src/mem/packet_queue.cc:155-162), so the
    one-bit round 3, which fits beside the in-flight round 1, still
    waits for the two-bit round 2 held ahead of it.
    """
    engine = engine_module.Engine()
    sender, weak_receiver, transmitter, _recorder = sender_with(
        engine, weak_bits=3
    )
    first = packed(1)
    second = packed(2)
    narrow = _one_bit_round(3)

    sender.admit(first)
    sender.admit(second)
    narrow_admitted = sender.admit(narrow)
    sent_while_held = list(transmitter.sent)
    weak_receiver.receive_round(first, no_sender_counts_it)
    weak_receiver.store.release_round((1, 1))

    assert narrow_admitted is False
    assert sent_while_held == [1]
    assert transmitter.sent == [1, 2, 3]


def _one_bit_round(round_index):
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1,),
        size_bits=1,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(
        packet, round_records.WINDOW_INPUT_ROUTE, 1
    )


def test_a_strong_primary_round_is_in_flight_until_it_lands():
    """The packing stage's bound holds the round until the windows hear of it.

    The strong syndrome buffer publishes a strong-primary round at its
    landing, the card's 0.26 us after the send, so the sender counts it
    across the crossing and lets it go in the event after the landing.
    """
    engine = engine_module.Engine()
    strong_receiver = RecordingStrongReceiver()
    sender, _weak_receiver, _transmitter, _recorder = sender_with(
        engine,
        strong_receiver=strong_receiver,
        publishes_from_strong_store=True,
    )
    first = packed(1)

    retried_counts = []
    sender.packing_line.retry = lambda: retried_counts.append(
        sender.strong_crossing_count
    )

    sender.admit(first)
    count_while_crossing = sender.strong_crossing_count
    engine.run()

    assert count_while_crossing == 1
    assert strong_receiver.written == [1]
    assert sender.strong_crossing_count == 0
    assert retried_counts == [0]


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
    assert weak_receiver.reserved_bits_by_round == {}
    assert transmitter.sent == []
    assert sender.held_rounds.count == 1
