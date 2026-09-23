"""The room-side end: room counts the bits in flight, the landing stores.

Referent: gem5's packet store counts its reserved bytes as taken before
the data lands (gem5 src/dev/net/pktfifo.hh, `avail() = _maxsize -
_size - _reserved` with `reserve(len)`); this end counts the bits of a
round crossing toward it the same way, reserved by the sender before
the round leaves. The crossing
itself is the sender's and is tested where it is executed: the
controller's write in tests/controller/test_syndrome_round_sender.py,
the escalated region's send in tests/escalation/test_strong_redecode.py.
An escalated region lands whole (Toshio 2510.25222 lines 1247 to 1250
assign the region's rounds to the strong decoder at the switch) and
each of its rounds takes its slot and wakes the window side.

The whole-run law at the end of the file places that landing in the
pipeline: under a strong-primary policy the landing is what makes a
window ready, on the declared card of tests/declared_run.py.
"""

import pytest

import decsim.assembly as assembly
import decsim.config as config
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver,
)

LANDING_TICKS = config.microseconds_to_ticks(0.5)
# every round this file sends carries one fragment of three bits
BITS_PER_ROUND = 3


def packet(round_index: int) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0, 1),
        size_bits=BITS_PER_ROUND,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(1, round_index, (fragment,))


class RecordingWindows:
    """The window side, keeping every room-side round it hears of."""

    def __init__(self):
        self.room_rounds = []

    def accept_room_round(self, operation_id, round_index):
        self.room_rounds.append((operation_id, round_index))


class RecordingListener:
    def __init__(self):
        self.stored = []
        self.released = []
        self.copies = []

    def round_stored(self, round_key, _packet):
        self.stored.append(round_key)

    def round_released(self, round_key):
        self.released.append(round_key)

    def copy_made(self, round_key, bits, source_name, target_name):
        self.copies.append((round_key, bits, source_name, target_name))


def room_side(engine, bits=None, listener=None, windows=None):
    store_settings = syndrome_buffer_settings.SyndromeBufferSettings(bits=bits)
    store = syndrome_buffer_module.SyndromeBuffer(store_settings)
    if listener is not None:
        store.trace.round_stored.connect(listener.round_stored)
        store.trace.round_released.connect(listener.round_released)
    receiver = strong_syndrome_round_receiver.StrongSyndromeRoundReceiver(
        engine
    )
    receiver.store = store
    if listener is not None:
        receiver.trace.copy_made.connect(listener.copy_made)
    if windows is not None:
        receiver.windows = windows
    return receiver


class StoreWithoutSettlement:
    """A store row with every SyndromeBuffer method but check_settled."""

    occupied_bits = 0

    def has_room(self, bits, reserved_bits=0):
        del bits, reserved_bits
        return True

    def accept_packed_round(self, packet, *, publication_tick):
        del packet, publication_tick

    def release_round(self, round_key):
        del round_key

    def capacity_bits(self):
        return None

    def held_rounds_description(self):
        return "empty"


def region(*round_indices) -> round_records.EscalatedRegion:
    """The escalated region of these rounds, in the strong request's name."""
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 1
    )
    packets = tuple(packet(round_index) for round_index in round_indices)
    return round_records.EscalatedRegion.of(request_key, packets)


def cross(
    engine: engine_module.Engine,
    receiver: strong_syndrome_round_receiver.StrongSyndromeRoundReceiver,
    round_index: int,
) -> None:
    """One round on its way over the crossing, landing after 0.5 us.

    The controller reserves the room and sends; the send itself is the
    controller's and is tested at
    tests/controller/test_syndrome_round_sender.py, so this file stands
    the round up at the landing tick instead.
    """
    receiver.reserve_write(BITS_PER_ROUND)
    landing = packet(round_index)
    packed = round_records.PackedRound(
        landing, round_records.WINDOW_INPUT_ROUTE, BITS_PER_ROUND
    )
    engine.schedule(LANDING_TICKS, lambda: receiver.receive_round(packed))


def test_a_write_lands_after_the_crossing_and_the_listener_hears_it_once():
    engine = engine_module.Engine()
    listener = RecordingListener()
    windows = RecordingWindows()
    receiver = room_side(engine, listener=listener, windows=windows)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])

    cross(engine, receiver, 1)
    in_flight = receiver.store.retained_fragments((1, 1))
    engine.run()

    assert in_flight is None
    assert receiver.store.publication_tick((1, 1)) == LANDING_TICKS
    assert listener.stored == [(1, 1)]
    assert windows.room_rounds == [(1, 1)]


def test_the_reserved_bits_count_against_the_room_until_the_round_lands():
    engine = engine_module.Engine()
    receiver = room_side(engine, bits=BITS_PER_ROUND)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])

    cross(engine, receiver, 1)
    room_while_crossing = receiver.has_room(BITS_PER_ROUND)
    reserved_while_crossing = receiver.reserved_bits
    engine.run()

    assert room_while_crossing is False
    assert reserved_while_crossing == BITS_PER_ROUND
    assert receiver.writes_in_flight == 0
    assert receiver.reserved_bits == 0
    assert receiver.has_room(BITS_PER_ROUND) is False
    assert receiver.store.occupied_bits == BITS_PER_ROUND
    assert receiver.store.occupancy == 1


def test_a_round_whose_readers_resolved_while_crossing_is_dropped_at_landing():
    engine = engine_module.Engine()
    receiver = room_side(engine)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])

    cross(engine, receiver, 1)
    receiver.store.release_hold(reads)
    engine.run()

    assert receiver.store.retained_fragments((1, 1)) is None
    receiver.check_settled()


def test_a_round_that_lands_after_its_operation_closed_is_dropped():
    engine = engine_module.Engine()
    receiver = room_side(engine, bits=BITS_PER_ROUND)
    receiver.store.open_operation(1)
    room_before = receiver.has_room(BITS_PER_ROUND)

    cross(engine, receiver, 1)
    receiver.store.close_operation(1)
    engine.run()

    assert room_before is True
    assert receiver.store.retained_fragments((1, 1)) is None
    assert receiver.store.occupancy == 0
    assert receiver.store.has_operation(1) is False
    assert receiver.writes_in_flight == 0
    assert receiver.has_room(BITS_PER_ROUND) is True
    receiver.check_settled()


def test_settlement_reports_a_write_still_in_flight():
    engine = engine_module.Engine()
    receiver = room_side(engine)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])
    cross(engine, receiver, 1)

    with pytest.raises(RuntimeError, match="1 writes in flight"):
        receiver.check_settled()


def test_an_escalated_region_lands_whole_and_each_round_wakes_the_windows():
    engine = engine_module.Engine()
    listener = RecordingListener()
    windows = RecordingWindows()
    three_rounds_bits = 3 * BITS_PER_ROUND
    two_rounds_bits = 2 * BITS_PER_ROUND
    receiver = room_side(
        engine, bits=three_rounds_bits, listener=listener, windows=windows
    )
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1), (1, 2)])
    carried = region(1, 2)

    receiver.reserve_region(2, carried.wire_bits)
    room_while_crossing = receiver.has_room(BITS_PER_ROUND)
    engine.schedule(LANDING_TICKS, lambda: receiver.receive_region(carried))
    engine.run()

    assert carried.wire_bits == two_rounds_bits
    assert room_while_crossing is True
    assert receiver.writes_in_flight == 0
    assert receiver.reserved_bits == 0
    assert receiver.store.occupancy == 2
    assert receiver.store.occupied_bits == two_rounds_bits
    assert receiver.store.publication_tick((1, 2)) == LANDING_TICKS
    assert windows.room_rounds == [(1, 1), (1, 2)]
    assert listener.copies == [
        (
            (1, 1),
            BITS_PER_ROUND,
            "weak syndrome buffer",
            "strong syndrome buffer",
        ),
        (
            (1, 2),
            BITS_PER_ROUND,
            "weak syndrome buffer",
            "strong syndrome buffer",
        ),
    ]


def test_a_regions_reservation_is_its_bits_and_the_refusal_names_them():
    """An escalation cannot wait, so the refusal names the yaml key."""
    engine = engine_module.Engine()
    two_rounds_bits = 2 * BITS_PER_ROUND
    receiver = room_side(engine, bits=BITS_PER_ROUND)
    carried = region(1, 2)

    with pytest.raises(RuntimeError) as refusal:
        receiver.reserve_region(2, carried.wire_bits)

    sentence = str(refusal.value)
    assert f"no room for the {two_rounds_bits} bits" in sentence
    assert "of an escalated region's 2 rounds" in sentence
    assert "0 bits stored and 0 reserved" in sentence
    assert f"strong_syndrome_buffer.bits {BITS_PER_ROUND}" in sentence
    assert receiver.writes_in_flight == 0
    assert receiver.reserved_bits == 0


# ---- the landing in the whole pipeline


def test_a_strong_primary_window_is_ready_on_the_room_side_landing():
    """With the strong tier alone, this landing drives readiness.

    Under StrongOnly (decsim/escalation/policies.py) a round travels
    once, over controller_to_strong_buffer, so round r is ready at r
    plus qpu_to_controller 2 plus readout_to_bits 3 plus the room-side
    hop 7; the window then pays strong_buffer_to_strong_decoder 6 and
    the 30 us strong decode, and its correction rides
    strong_decoder_to_frame 4 home before the 1 us frame write. Every
    latency is the declared card's (tests/declared_run.py).
    """
    machine = declared_run.strong_only_run(rounds=6)
    windows = machine.observation.windows.windows
    window = windows[(1, 0)]
    snapshot = machine.pauli_frame.snapshot()
    (record,) = snapshot.records
    expected_first_round = config.microseconds_to_ticks(13.0)
    expected_data_complete = config.microseconds_to_ticks(18.0)
    expected_done = config.microseconds_to_ticks(54.0)
    expected_accepted = config.microseconds_to_ticks(58.0)
    expected_committed = config.microseconds_to_ticks(59.0)

    assert window.t_first_round == expected_first_round
    assert window.t_data_complete == expected_data_complete
    assert window.t_done == expected_done
    assert record.tier == "strong"
    assert record.accepted_ticks == expected_accepted
    assert record.committed_ticks == expected_committed


def test_arrival_can_create_a_windows_first_round_hold() -> None:
    """gem5 base.cc services response targets before freeing their entry."""
    engine = engine_module.Engine()
    receiver = room_side(engine, bits=BITS_PER_ROUND)
    receiver.store.open_operation(1)
    receiver.windows = _ArrivalConsumer(receiver.store)

    cross(engine, receiver, 1)
    engine.run()

    fragments = receiver.store.retained_fragments((1, 1))
    assert fragments is not None
    assert fragments[0].bits == (1, 0, 1)
    assert receiver.store.publication_tick((1, 1)) == LANDING_TICKS
    assert receiver.has_room(BITS_PER_ROUND) is False


class _ArrivalConsumer:
    """A window consumer whose first claim is created by the arrival."""

    def __init__(self, store: syndrome_buffer_module.SyndromeBuffer) -> None:
        self.store = store

    def accept_room_round(self, operation_id: int, round_index: int) -> None:
        reads = decoding_records.WindowReads((operation_id, 0))
        round_key = (operation_id, round_index)
        self.store.register_hold(reads, [round_key])


def test_a_store_row_without_check_settled_does_not_bind_to_the_strong_end():
    """The strong end calls check_settled, so the port it binds says so."""
    engine = engine_module.Engine()
    receiver = strong_syndrome_round_receiver.StrongSyndromeRoundReceiver(
        engine
    )
    seats = {
        "strong_syndrome_round_receiver": receiver,
        "strong_syndrome_buffer": StoreWithoutSettlement(),
    }
    wires = (("strong_syndrome_round_receiver.store", "strong_syndrome_buffer"),)

    with pytest.raises(ValueError, match="does not answer"):
        assembly.bind(wires, seats)
