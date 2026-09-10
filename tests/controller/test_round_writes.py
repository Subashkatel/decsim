"""The writer: a finished round enters every store it must reach, or waits.

The backpressure law: a store answers has_room before any write; a round
that finds no room waits in HeldRounds and enters in completion order
when a slot frees (gem5 src/mem/cache/base.cc clearBlocked and the
retry; Ciw ciw/node.py release_blocked_individual), or is dropped under
the drop knob (ns-3 point-to-point-net-device.cc Send: Enqueue false,
packet dropped). A strong-primary plan takes one hop, into the strong
store; a feedback-memory round still crosses Buffer 0.

The controller is also the end the room-side round leaves by, so it
executes the controller_to_strong_buffer send and the room side handles
the landing (OMNeT++ csimplemodule.cc:333-334, a module sends only what
it owns; gem5 packet.hh:424-431, a transfer is billed to the port it
left by). The crossing's own delay is the card's, on the reference
profile.
"""

import decsim.config as config
import decsim.controller.round_writes as round_writes
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.observe.round_events as round_events
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer

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


class RecordingStrongWriter:
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


def writer_with(
    engine,
    weak_rounds=None,
    strong_writer=None,
    publishes_from_strong_store=False,
    on_full=STALL,
):
    recorder = round_events.RoundEventRecorder(engine)
    held = round_writes.HeldRounds(engine, on_full)
    held.trace.round_event.connect(recorder.record)
    settings = round_store_settings.RoundStoreSettings(rounds=weak_rounds)
    weak_store = round_store_module.RoundStore(
        settings, on_slot_freed=held.retry
    )
    transmitter = RecordingTransmitter(engine)
    profile = link_profiles.logical_reference_profile()
    links = fabric_module.LinkFabric(profile, engine)
    writer = round_writes.RoundWriter(
        engine,
        links,
        weak_store,
        strong_writer,
        publishes_from_strong_store=publishes_from_strong_store,
        held_rounds=held,
        transmitter=transmitter,
    )
    return writer, weak_store, transmitter, recorder


def test_a_round_with_no_room_is_held_and_written_in_order_when_a_slot_frees():
    engine = engine_module.Engine()
    writer, weak_store, transmitter, recorder = writer_with(
        engine, weak_rounds=1
    )
    first = packed(1)
    second = packed(2)

    first_admitted = writer.admit(first)
    second_admitted = writer.admit(second)
    held_before_release = writer.held_rounds.count
    weak_store.release_round((1, 1))

    assert first_admitted is True
    assert second_admitted is False
    assert held_before_release == 1
    assert transmitter.sent == [1, 2]
    assert writer.held_rounds.count == 0
    stalled = [
        event.round_index
        for event in recorder.events
        if event.kind == "STALLED"
    ]
    assert stalled == [2]


def test_a_strong_primary_window_round_takes_one_hop_into_the_strong_store():
    engine = engine_module.Engine()
    strong_writer = RecordingStrongWriter()
    writer, weak_store, transmitter, _recorder = writer_with(
        engine, strong_writer=strong_writer, publishes_from_strong_store=True
    )
    window_round = packed(1)
    memory_round = packed(2, route=MEMORY_ROUTE)

    writer.admit(window_round)
    writer.admit(memory_round)
    engine.run()

    assert strong_writer.written == [1, 2]
    assert weak_store.retained_fragments((1, 1)) is None
    assert weak_store.retained_fragments((1, 2)) is not None
    assert transmitter.sent == [2]


def test_the_controller_carries_the_round_to_the_room_side_and_lands_it():
    """The send is the controller's; the landing is the room side's.

    The reference card prices controller_to_strong_buffer at 0.26
    microseconds (Caune 2410.05202 Fig. 1a stage F), so the round is
    reserved and sent at the write and stored one card delay later, by
    the room side's own method.
    """
    engine = engine_module.Engine()
    store_settings = round_store_settings.RoundStoreSettings()
    strong_store = round_store_module.RoundStore(store_settings)
    reads = decoding_records.WindowReads((1, 0))
    strong_store.register_hold(reads, [(1, 1)])
    room_side = strong_round_writer.StrongRoundWriter(engine, strong_store)
    writer, _weak_store, _transmitter, _recorder = writer_with(
        engine, strong_writer=room_side
    )
    first = packed(1)

    writer.admit(first)
    reserved_while_crossing = room_side.writes_in_flight
    stored_at_the_write = strong_store.occupancy
    engine.run()

    crossing_ticks = config.microseconds_to_ticks(0.26)
    assert reserved_while_crossing == 1
    assert stored_at_the_write == 0
    assert strong_store.occupancy == 1
    assert strong_store.publication_tick((1, 1)) == crossing_ticks
    assert room_side.writes_in_flight == 0


def test_a_full_strong_store_holds_the_round_too():
    engine = engine_module.Engine()
    strong_writer = RecordingStrongWriter(room=False)
    writer, weak_store, transmitter, _recorder = writer_with(
        engine, strong_writer=strong_writer
    )
    first = packed(1)

    admitted = writer.admit(first)

    assert admitted is False
    assert weak_store.occupancy == 0
    assert transmitter.sent == []
    assert writer.held_rounds.count == 1


def test_the_drop_knob_drops_a_round_that_found_no_room():
    engine = engine_module.Engine()
    writer, _weak_store, transmitter, recorder = writer_with(
        engine, weak_rounds=1, on_full=DROP
    )
    first = packed(1)
    second = packed(2)

    writer.admit(first)
    writer.admit(second)

    assert transmitter.sent == [1]
    assert recorder.packing_drops == 1
    assert writer.held_rounds.count == 0


def test_the_buffer_0_line_is_narrated_only_on_the_io_line():
    import decsim.observe.log_writers as log_writers

    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.io_line.connect(log.write)
    writer, _weak_store, _transmitter, _recorder = writer_with(engine)
    silent_engine = engine_module.Engine()
    silent_log = log_writers.LogWriter()
    silent_engine.line.connect(silent_log.write)
    silent_writer, _store, _sent, _events = writer_with(silent_engine)
    first = packed(1)

    writer.admit(first)
    silent_writer.admit(first)

    assert silent_log.lines == []
    (line,) = log.lines
    assert line.endswith(
        "Buffer 0: received round 1 of op 1 from packing; "
        "defects {0}; holds op 1 rounds 1 (1)"
    )
