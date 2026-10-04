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
"""

import pytest

import decsim.config as config
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver,
)

LANDING_TICKS = config.microseconds_to_ticks(0.5)
# every round this file sends carries one fragment of three bits
BITS_PER_ROUND = 3
FORMING_CYCLES = 5


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


def packed_round(round_index: int) -> round_records.PackedRound:
    """The round as the controller writes it toward this end."""
    landing = packet(round_index)
    return round_records.PackedRound(
        landing, round_records.WINDOW_INPUT_ROUTE, BITS_PER_ROUND
    )


class RecordingFormer:
    """The placement port at this seat, keeping what it is asked."""

    clock = None

    def __init__(self):
        self.formed = []
        self.cycles_asked = []

    def cycles_at(self, seat, round_count):
        self.cycles_asked.append((seat, round_count))
        return 0

    def form_at(self, seat, fragments, rounds_before=()):
        held = tuple(fragment.round_index for fragment in rounds_before)
        self.formed.append((seat, fragments[0].round_index, held))
        return fragments

    def width_at(self, seat, fragments):
        del seat
        return round_records.fragment_wire_bits(fragments)


class FormingInCycles(RecordingFormer):
    """A seat that takes FORMING_CYCLES of a 10-tick clock to form."""

    clock = config.Clock(10)

    def cycles_at(self, seat, round_count):
        self.cycles_asked.append((seat, round_count))
        return FORMING_CYCLES


class FormingInAPipeline(RecordingFormer):
    """A pipelined seat: 3 cycles for the first round, 2 for each after."""

    clock = config.Clock(10)
    settings = event_settings.DetectionEventSettings(
        clock=clock, latency_cycles=3, cycles_per_round=2
    )

    def cycles_at(self, seat: str, round_count: int) -> int:
        """The pipelined cost of forming round_count rounds together."""
        self.cycles_asked.append((seat, round_count))
        return self.settings.cycles_for(round_count)


def room_side(
    engine, bits=None, listener=None, windows=None, detection_events=None
):
    store_settings = syndrome_buffer_module.SyndromeBufferSettings(bits=bits)
    store = syndrome_buffer_module.SyndromeBuffer(store_settings, engine)
    if listener is not None:
        store.trace.round_stored.connect(listener.round_stored)
        store.trace.round_released.connect(listener.round_released)
    receiver = strong_syndrome_round_receiver.StrongSyndromeRoundReceiver(
        engine
    )
    receiver.store = store
    if detection_events is None:
        at_the_controller = event_settings.DetectionEventSettings()
        detection_events = formation.SeatedFormation(None, at_the_controller)
    receiver.detection_events = detection_events
    if listener is not None:
        receiver.trace.copy_made.connect(listener.copy_made)
    if windows is not None:
        receiver.windows = windows
    return receiver


def region(*round_indices, first_round=1) -> round_records.EscalatedRegion:
    """The escalated region of these rounds, in the strong request's name."""
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 1
    )
    packets = tuple(packet(round_index) for round_index in round_indices)
    return round_records.EscalatedRegion.of(request_key, packets, first_round)


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
    packed = packed_round(round_index)
    receiver.reserve_write(packed)
    engine.schedule(LANDING_TICKS, lambda: receiver.receive_round(packed))


def test_the_reserved_bits_count_against_the_room_until_the_round_lands():
    engine = engine_module.Engine()
    receiver = room_side(engine, bits=BITS_PER_ROUND)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])

    asked = packed_round(2)

    cross(engine, receiver, 1)
    room_while_crossing = receiver.has_room(asked)
    reserved_while_crossing = dict(receiver.reserved_bits_by_round)
    engine.run()

    assert room_while_crossing is False
    assert reserved_while_crossing == {(1, 1): BITS_PER_ROUND}
    assert receiver.reserved_bits_by_round == {}
    assert receiver.has_room(asked) is False
    assert receiver.store.occupied_bits == BITS_PER_ROUND
    assert receiver.store.occupancy == 1


def test_a_round_being_formed_here_keeps_its_room_until_it_is_stored():
    """gem5's packet store clears a reserve only in the push that fills it.

    src/dev/net/pktfifo.hh: reserve(len) adds to _reserved and push()
    moves the length into _size, so avail() never counts a slot twice nor
    frees one early. Here the round lands at 0.5 us and forms for five
    cycles of a 10-tick clock; between the two its bits are still taken.
    """
    engine = engine_module.Engine()
    former = FormingInCycles()
    receiver = room_side(engine, bits=BITS_PER_ROUND, detection_events=former)
    receiver.store.open_operation(1)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])
    asked = packed_round(2)
    rooms_asked = []

    def ask_room() -> None:
        has_room = receiver.has_room(asked)
        rooms_asked.append(has_room)

    while_forming = LANDING_TICKS + 1
    cross(engine, receiver, 1)
    engine.schedule(while_forming, ask_room)
    engine.run()

    assert rooms_asked == [False]
    assert receiver.reserved_bits_by_round == {}
    assert receiver.store.occupied_bits == BITS_PER_ROUND


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
    asked = packed_round(2)
    room_before = receiver.has_room(asked)

    cross(engine, receiver, 1)
    receiver.store.close_operation(1)
    engine.run()

    assert room_before is True
    assert receiver.store.retained_fragments((1, 1)) is None
    assert receiver.store.occupancy == 0
    assert receiver.store.has_operation(1) is False
    assert receiver.reserved_bits_by_round == {}
    assert receiver.has_room(asked) is True
    receiver.check_settled()


def test_settlement_reports_a_write_still_in_flight():
    engine = engine_module.Engine()
    receiver = room_side(engine)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])
    cross(engine, receiver, 1)

    with pytest.raises(RuntimeError, match="strong syndrome buffer ended"):
        receiver.check_settled()


def test_the_rounds_before_a_region_land_raw_and_cost_no_formation():
    """The seat holds rounds 1 and 2 for round 3 and forms neither.

    The pipelined seat forms two rounds, 3 cycles then 2 of a 10-tick
    clock, so the region is stored at 50; four formed rounds would take
    until 90.
    """
    engine = engine_module.Engine()
    former = FormingInAPipeline()
    receiver = room_side(engine, detection_events=former)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1), (1, 2), (1, 3), (1, 4)])
    carried = region(1, 2, 3, 4, first_round=3)
    stored_at = []

    def record_stored() -> None:
        stored_at.append(engine.now)

    receiver.reserve_region(carried)
    receiver.receive_region(carried, record_stored)
    engine.run()

    seat = "strong_syndrome_buffer"
    assert carried.wire_bits == 4 * BITS_PER_ROUND
    assert stored_at == [50]
    assert receiver.detection_events.formed == [
        (seat, 3, (1, 2)),
        (seat, 4, ()),
    ]
    assert receiver.store.occupancy == 4


def test_a_region_formed_here_is_reported_stored_once_every_round_is():
    """Five cycles of a 10-tick clock: the store holds both rounds at 50.

    The sender counts the region's rounds as carried until then, so a
    wake-up at the landing does not send them again.
    """
    engine = engine_module.Engine()
    former = FormingInCycles()
    receiver = room_side(engine, detection_events=former)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1), (1, 2)])
    carried = region(1, 2)
    stored_at = []

    def record_stored() -> None:
        stored_at.append((engine.now, receiver.store.occupancy))

    receiver.reserve_region(carried)
    receiver.receive_region(carried, record_stored)
    engine.run()

    assert stored_at == [(50, 2)]


def test_a_later_region_forms_after_an_earlier_one_landing_at_once():
    """Rounds 1 to 6 and 7 to 10 land together; round 7 reads round 6.

    The former takes a region's rounds one every 2 cycles, so the second
    region's first round enters at cycle 12 and leaves 3 cycles after
    its entry, 6 cycles after that for its last: the store holds it at
    210, after the first region at 130, never before.
    """
    engine = engine_module.Engine()
    former = FormingInAPipeline()
    receiver = room_side(engine, detection_events=former)
    first = region(1, 2, 3, 4, 5, 6)
    second = region(7, 8, 9, 10)
    stored_at = []

    def first_stored() -> None:
        stored_at.append((engine.now, "rounds 1 to 6"))

    def second_stored() -> None:
        stored_at.append((engine.now, "rounds 7 to 10"))

    receiver.reserve_region(first)
    receiver.reserve_region(second)
    receiver.receive_region(first, first_stored)
    receiver.receive_region(second, second_stored)
    engine.run()

    assert stored_at == [(130, "rounds 1 to 6"), (210, "rounds 7 to 10")]
    formed_rounds = [round_index for _, round_index, _ in former.formed]
    assert formed_rounds == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def stated_event_round(round_index: int) -> round_records.PackedRound:
    """A round whose source states its events take one bit."""
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0, 1),
        size_bits=BITS_PER_ROUND,
        fragment_index=0,
        event_bits=1,
    )
    landing = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(
        landing, round_records.WINDOW_INPUT_ROUTE, BITS_PER_ROUND
    )


def test_the_room_is_weighed_at_the_width_this_seat_stores():
    """Rounds of three raw bits formed here into one event take one bit each.

    gem5 makes room for a block at the size it will be stored at after
    its own compressor, not the packet's (src/mem/cache/base.cc:1678-1698).
    """
    engine = engine_module.Engine()
    settings = event_settings.DetectionEventSettings(
        formed_at=("strong_syndrome_buffer",)
    )
    here = formation.SeatedFormation(None, settings)
    receiver = room_side(engine, bits=2, detection_events=here)
    crossing = stated_event_round(1)
    asked = stated_event_round(2)
    receiver.reserve_write(crossing)

    has_room = receiver.has_room(asked)

    assert receiver.reserved_bits_by_round == {(1, 1): 1}
    assert has_room is True


def test_a_region_reserves_its_formed_rounds_and_its_raw_rounds_before():
    """The rounds before land raw; the rounds formed here take one bit."""
    engine = engine_module.Engine()
    settings = event_settings.DetectionEventSettings(
        formed_at=("strong_syndrome_buffer",)
    )
    here = formation.SeatedFormation(None, settings)
    room_bits = 2 * BITS_PER_ROUND + 2
    receiver = room_side(engine, bits=room_bits, detection_events=here)
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 1
    )
    first = stated_event_round(1)
    second = stated_event_round(2)
    third = stated_event_round(3)
    fourth = stated_event_round(4)
    packets = (first.packet, second.packet, third.packet, fourth.packet)
    carried = round_records.EscalatedRegion.of(request_key, packets, 3)

    receiver.reserve_region(carried)

    assert receiver.reserved_bits_by_round == {
        (1, 1): BITS_PER_ROUND,
        (1, 2): BITS_PER_ROUND,
        (1, 3): 1,
        (1, 4): 1,
    }


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
    asked = packed_round(2)
    assert receiver.store.publication_tick((1, 1)) == LANDING_TICKS
    assert receiver.has_room(asked) is False


class _ArrivalConsumer:
    """A window consumer whose first claim is created by the arrival."""

    def __init__(self, store: syndrome_buffer_module.SyndromeBuffer) -> None:
        self.store = store

    def accept_room_round(self, operation_id: int, round_index: int) -> None:
        reads = decoding_records.WindowReads((operation_id, 0))
        round_key = (operation_id, round_index)
        self.store.register_hold(reads, [round_key])
