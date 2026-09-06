"""The writer: a finished round enters every store it must reach, or waits.

The backpressure law: a store answers has_room before any write; a round
that finds no room waits in HeldRounds and enters in completion order
when a slot frees (gem5 src/mem/cache/base.cc clearBlocked and the
retry; Ciw ciw/node.py release_blocked_individual), or is dropped under
the drop knob (ns-3 point-to-point-net-device.cc Send: Enqueue false,
packet dropped). A strong-primary plan takes one hop, into the strong
store; a feedback-memory round still crosses Buffer 0.
"""

import decsim.controller.round_writes as round_writes
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.message as message
import decsim.observe.round_events as round_events
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings

STALL = controller_settings.PackingOverflowPolicy.STALL
DROP = controller_settings.PackingOverflowPolicy.DROP_ROUND
MEMORY_ROUTE = message.SyndromePacketRoute.feedback_memory_round(9)


def packed(round_index, route=message.WINDOW_INPUT_ROUTE):
    fragment = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = message.SyndromeRoundPacket(1, round_index, (fragment,))
    return message.PackedRound(packet, route, 2)


class RecordingTransmitter:
    def __init__(self, engine):
        self.engine = engine
        self.sent = []

    def publication_tick_at_storage(self, route):
        if route.kind is message.SyndromePacketRouteKind.WINDOW_INPUT:
            return self.engine.now
        return None

    def send(self, round):
        self.sent.append(round.packet.round_index)


class RecordingStrongWriter:
    def __init__(self, room=True):
        self.room = room
        self.written = []

    def has_room(self):
        return self.room

    def write(self, packet, *, packet_bits, attribution):
        del packet_bits, attribution
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
    held.round_event.connect(recorder.record)
    settings = round_store_settings.RoundStoreSettings(rounds=weak_rounds)
    weak_store = round_store_module.RoundStore(
        settings, on_slot_freed=held.retry
    )
    transmitter = RecordingTransmitter(engine)
    writer = round_writes.RoundWriter(
        engine,
        weak_store,
        strong_writer,
        publishes_from_strong_store=publishes_from_strong_store,
        held_rounds=held,
        transmitter=transmitter,
    )
    writer.round_event.connect(recorder.record)
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


def test_a_published_round_is_recorded_at_its_storage_when_the_hop_is_free():
    engine = engine_module.Engine()
    writer, weak_store, _transmitter, recorder = writer_with(engine)
    first = packed(1)

    writer.admit(first)

    assert weak_store.publication_tick((1, 1)) == 0
    published = [
        event for event in recorder.events if event.kind == "PUBLISHED"
    ]
    assert [(event.round_index, event.tick) for event in published] == [(1, 0)]


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

    assert strong_writer.written == [1, 2]
    assert weak_store.retained_fragments((1, 1)) is None
    assert weak_store.retained_fragments((1, 2)) is not None
    assert transmitter.sent == [2]


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
