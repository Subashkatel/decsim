"""A store's receiving end: room counts the bits in flight, the landing stores.

A round occupies a slot when its bits are in the store, at the landing
of the hop that carried them, and it is readable at that same instant:
ns-3 schedules the destination device's own Receive after the
propagation (the ns3-point-to-point copy of
point-to-point-channel.cc:88-92 into point-to-point-net-device.cc:324),
OMNeT++ takes ownership into the destination module and inserts inside
its handler (omnetpp src/sim/csimplemodule.cc:782-783,
:799, with queueinglib/Queue.cc:84-94), Ciw counts the individual in the
destination's own accept (Ciw/ciw/node.py:602 into :102-103), and Caune
2410.05202 lines 1243-1247 store the outcomes in the decoder sequencer's
memory only after the propagation. The sender still refuses before it
sends, so the room counts the bits of the writes in flight as taken
(gem5 src/dev/net/pktfifo.hh, `avail() = _maxsize - _size - _reserved`
with `reserve(len)`).

The first half drives the written intake, the weak store's
(weak_buffer); the second the landed intake, the strong store's
(strong_buffer). The
crossing itself is the sender's and is tested where it is executed: the
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
import decsim.records.formation as formation_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module
import tests.declared_run as declared_run

LANDING_TICKS = 40_000
MEMORY_ROUTE = round_records.SyndromePacketRoute.feedback_memory_round(9)
# every written round carries one fragment of two bits
BITS_PER_ROUND = 2
# the landed intake's rounds land after 0.5 us
LANDED_TICKS = config.microseconds_to_ticks(0.5)
# every landed round carries one fragment of three bits
LANDED_BITS_PER_ROUND = 3
FORMING_CYCLES = 5
# the controller writes the weak store; an escalated region lands in the
# strong one, as the layout makes each store's intake from the graph
_CONTROLLER_LABEL = "controller assembler"
_WEAK_WRITE = receiver_module.Hop(
    "controller_to_weak_buffer", _CONTROLLER_LABEL
)
_STRONG_WRITE = receiver_module.Hop(
    "controller_to_strong_buffer", _CONTROLLER_LABEL
)
_ESCALATION = receiver_module.Hop("weak_to_strong", "weak")
WEAK_STORE_INTAKE = receiver_module.StoreIntake(
    name="weak_buffer",
    seat="weak_buffer",
    shape=receiver_module.WRITTEN,
    controller_write=_WEAK_WRITE,
)
STRONG_STORE_INTAKE = receiver_module.StoreIntake(
    name="strong_buffer",
    seat="strong_buffer",
    shape=receiver_module.LANDED,
    controller_write=_STRONG_WRITE,
    escalation=_ESCALATION,
)


class _Windows:
    """The WindowInput port, reading the store as the round arrives."""

    def __init__(self, engine, store) -> None:
        self.engine = engine
        self.store = store
        self.published = []

    def accept_window_input(self, packet) -> None:
        round_key = (packet.operation_id, packet.round_index)
        publication_tick = self.store.publication_tick(round_key)
        self.published.append((self.engine.now, round_key, publication_tick))


class _Output:
    """The store's outgoing port, recording what it is asked to send."""

    def __init__(self) -> None:
        self.sent = []

    def send_memory_round(self, packed, on_delivered) -> None:
        self.sent.append(packed.round_key)
        on_delivered()


def _packed(
    round_index: int, route=round_records.WINDOW_INPUT_ROUTE
) -> round_records.PackedRound:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(packet, route, BITS_PER_ROUND)


def _store(engine, bits=None) -> syndrome_buffer_module.SyndromeBuffer:
    settings = syndrome_buffer_module.SyndromeBufferSettings(bits=bits)
    return syndrome_buffer_module.SyndromeBuffer(settings, engine)


def _receiver_with(engine, store, output=None, detection_events=None):
    """A receiver whose rounds arrive formed, unless a placement is given."""
    windows = _Windows(engine, store)
    receiver = _written_receiver(engine)
    receiver.store = store
    receiver.windows = windows
    if detection_events is None:
        at_the_controller = event_settings.DetectionEventSettings()
        detection_events = formation.SeatedFormation(None, at_the_controller)
    receiver.detection_events = detection_events
    if output is not None:
        receiver.output = output
    return receiver, windows


def _no_sender_counts_it() -> None:
    """The publication callback of a round no transmitter counts here."""


def _cross(receiver, packed, landing_ticks=LANDING_TICKS):
    """One crossing: the room is taken at the send, the slot at the landing."""
    receiver.reserve_write(packed)
    receiver.engine.schedule(
        landing_ticks,
        lambda: receiver.receive_written(packed, _no_sender_counts_it),
    )


def test_the_intake_copy_is_made_at_the_landing_by_this_end():
    engine = engine_module.Engine()
    store = _store(engine)
    receiver, _windows = _receiver_with(engine, store)
    copies = []

    def copy_made(round_key, bits, source, destination) -> None:
        made_at = engine.now
        copies.append((made_at, round_key, bits, source, destination))

    receiver.trace.copy_made.connect(copy_made)
    packed = _packed(1)

    _cross(receiver, packed)
    engine.run()

    assert copies == [
        (
            LANDING_TICKS,
            (1, 1),
            2,
            "controller assembler",
            "weak_buffer",
        )
    ]


def test_a_timing_only_round_takes_its_slot_here_and_is_never_published():
    """The round leaves the store, so the store's own port sends it."""
    engine = engine_module.Engine()
    store = _store(engine)
    output = _Output()
    receiver, windows = _receiver_with(engine, store, output)
    packed = _packed(2, route=MEMORY_ROUTE)
    delivered = []

    receiver.reserve_write(packed)
    receiver.send_memory_round(packed, lambda: delivered.append(True))

    assert output.sent == [(1, 2)]
    assert delivered == [True]
    assert store.occupancy == 1
    assert store.publication_tick((1, 2)) is None
    assert windows.published == []


def test_a_timing_only_round_takes_its_slot_once_its_write_completes():
    """Three write cycles of a 10-tick clock from tick 1 end at 40."""
    engine = engine_module.Engine()
    engine.now = 1
    clock = config.Clock(10)
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        clock=clock, write_cycles=3
    )
    store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    output = _Output()
    receiver, _windows = _receiver_with(engine, store, output)
    packed = _packed(2, route=MEMORY_ROUTE)
    delivered = []

    receiver.reserve_write(packed)
    receiver.send_memory_round(packed, lambda: delivered.append(engine.now))
    sent_before_the_write_ends = list(output.sent)
    engine.run()

    assert sent_before_the_write_ends == []
    assert delivered == [40]


def test_a_write_still_in_flight_at_the_end_of_a_run_is_a_failure():
    engine = engine_module.Engine()
    store = _store(engine)
    receiver, _windows = _receiver_with(engine, store)

    crossing = _packed(1)
    receiver.reserve_write(crossing)

    with pytest.raises(RuntimeError, match="weak_buffer ended with 1"):
        receiver.check_settled()


def test_a_weak_store_too_small_for_a_window_stops_the_run_at_its_hold():
    """The first round forms 4 events, the rest 8; the window reads six."""
    settings = syndrome_buffer_module.SyndromeBufferSettings(bits=16)

    with pytest.raises(RuntimeError, match="no round leaving it makes room"):
        declared_run.weak_only_run(weak_syndrome_buffer=settings)


def test_write_cycles_move_every_reaction_point_by_the_store_periods():
    storage = config.Clock.from_megahertz(1.0)
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        clock=storage, write_cycles=3
    )
    free = declared_run.weak_only_run()
    charged = declared_run.weak_only_run(weak_syndrome_buffer=settings)
    free_ticks = declared_run.reaction_ticks(free)
    charged_ticks = declared_run.reaction_ticks(charged)
    paired = zip(charged_ticks, free_ticks, strict=True)
    shifts = [charged_tick - free_tick for charged_tick, free_tick in paired]
    expected = 3 * settings.clock.period_ticks
    assert shifts == [expected] * 6


def test_the_sender_hears_the_publication_after_the_windows_do():
    """on_published runs at the write edge, once the windows have the round.

    Three cycles of a 10-tick clock from tick 1 end at the edge 40.
    """
    engine = engine_module.Engine()
    engine.now = 1
    clock = config.Clock(10)
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        bits=BITS_PER_ROUND, clock=clock, write_cycles=3
    )
    store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    receiver, windows = _receiver_with(engine, store)
    packed = _packed(1)
    heard = []

    def sender_hears() -> None:
        published_count = len(windows.published)
        heard.append((engine.now, published_count))

    receiver.reserve_write(packed)
    receiver.receive_written(packed, sender_hears)
    engine.run()

    assert heard == [(40, 1)]


def test_a_priced_write_keeps_its_reservation_until_the_write_edge():
    engine = engine_module.Engine()
    engine.now = 1
    clock = config.Clock(10)
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        bits=BITS_PER_ROUND, clock=clock, write_cycles=3
    )
    store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    receiver, windows = _receiver_with(engine, store)
    packed = _packed(1)
    receiver.reserve_write(packed)
    receiver.receive_written(packed, _no_sender_counts_it)
    next_round = _packed(2)
    assert store.occupancy == 0
    assert receiver.reserved_bits_by_round == {(1, 1): BITS_PER_ROUND}
    assert receiver.has_room(next_round) is False
    assert windows.published == []

    engine.run()

    assert receiver.reserved_bits_by_round == {}
    assert store.occupancy == 1
    assert windows.published == [(40, (1, 1), 40)]


class _ChipSource:
    """Recipes of a one-round operation: its one event is its first outcome."""

    def formation_table(self, operation_id):
        """The one table."""
        del operation_id
        recipe = formation_records.DetectorRecipe(
            detector_index=0,
            round_index=1,
            kind=formation_records.LayerKind.PREPARATION,
            records=((1, 0),),
            reference_parity=0,
            coordinates=(),
        )
        return formation_records.FormationTable(
            round_count=1,
            packet_width_by_round={1: 2},
            readout_slot_start=None,
            detectors=(recipe,),
            observables=(),
        )


def _on_the_chip(clock=None, latency_cycles=0):
    """The former seated at this store's receiving end."""
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_buffer",),
        clock=clock,
        latency_cycles=latency_cycles,
    )
    source = _ChipSource()
    return formation.SeatedFormation(source, settings)


def test_the_store_holds_the_events_when_the_chip_forms_them():
    """The hop's copy is the raw round; the store and the windows get events."""
    engine = engine_module.Engine()
    store = _store(engine)
    on_the_chip = _on_the_chip()
    receiver, _windows = _receiver_with(
        engine, store, detection_events=on_the_chip
    )
    copied_bits = []

    def copy_made(round_key, bits, source, destination) -> None:
        del round_key, source, destination
        copied_bits.append(bits)

    receiver.trace.copy_made.connect(copy_made)
    packed = _packed(1)

    _cross(receiver, packed)
    engine.run()

    (stored,) = store.retained_fragments((1, 1))
    assert stored.bits == (1,)
    assert stored.size_bits == 1
    assert copied_bits == [2]


def test_the_room_is_weighed_at_the_width_the_chip_stores():
    """A round that crosses at two bits and is stored as one event takes one.

    gem5 makes room for a block at the size it will be stored at after
    its own compressor, not the packet's (src/mem/cache/base.cc:1678-1698).
    """
    engine = engine_module.Engine()
    store = _store(engine, bits=2)
    on_the_chip = _on_the_chip()
    receiver, _windows = _receiver_with(
        engine, store, detection_events=on_the_chip
    )
    crossing = _packed(1)
    asked = _packed(2)
    receiver.reserve_write(crossing)

    has_room = receiver.has_room(asked)

    assert receiver.reserved_bits_by_round == {(1, 1): 1}
    assert has_room is True


def test_the_chips_formation_cycles_are_added_to_the_write_cycles():
    engine = engine_module.Engine()
    clock = config.Clock(10)
    formation_cycles = 5
    write_cycles = 3
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        clock=clock, write_cycles=write_cycles
    )
    store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    on_the_chip = _on_the_chip(clock, formation_cycles)
    receiver, windows = _receiver_with(
        engine, store, detection_events=on_the_chip
    )
    packed = _packed(1)

    receiver.reserve_write(packed)
    receiver.receive_written(packed, _no_sender_counts_it)
    engine.run()

    charged_cycles = formation_cycles + write_cycles
    expected = charged_cycles * clock.period_ticks
    assert windows.published == [(expected, (1, 1), expected)]


def test_the_store_is_asked_for_the_round_beside_the_rounds_in_flight():
    """gem5 asks the responder with the packet (src/mem/port.hh:268)."""
    engine = engine_module.Engine()
    store = _RoomAskingStore(engine)
    receiver, _windows = _receiver_with(engine, store)
    crossing = _packed(1)
    asked = _packed(2)
    receiver.reserve_write(crossing)

    receiver.has_room(asked)

    assert store.asked[-1] == ((1, 2), BITS_PER_ROUND, {(1, 1): BITS_PER_ROUND})


class _RoomAskingStore(syndrome_buffer_module.SyndromeBuffer):
    """An unbounded store recording every room question it is asked."""

    def __init__(self, engine) -> None:
        settings = syndrome_buffer_module.SyndromeBufferSettings()
        syndrome_buffer_module.SyndromeBuffer.__init__(self, settings, engine)
        self.asked = []

    def has_room(self, round_key, bits, reserved_bits_by_round):
        reserved = dict(reserved_bits_by_round)
        self.asked.append((round_key, bits, reserved))
        return True


def packet(round_index: int) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0, 1),
        size_bits=LANDED_BITS_PER_ROUND,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(1, round_index, (fragment,))


def packed_round(round_index: int) -> round_records.PackedRound:
    """The round as the controller writes it toward this end."""
    landing = packet(round_index)
    return round_records.PackedRound(
        landing, round_records.WINDOW_INPUT_ROUTE, LANDED_BITS_PER_ROUND
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
    receiver = _landed_receiver(engine)
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
        1, 0, window_records.STRONG_POOL_ID, 1
    )
    packets = tuple(packet(round_index) for round_index in round_indices)
    return round_records.EscalatedRegion.of(request_key, packets, first_round)


def cross(
    engine: engine_module.Engine,
    receiver: receiver_module.SyndromeRoundReceiver,
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
    engine.schedule(LANDED_TICKS, lambda: receiver.receive_landed(packed))


def test_the_reserved_bits_count_against_the_room_until_the_round_lands():
    engine = engine_module.Engine()
    receiver = room_side(engine, bits=LANDED_BITS_PER_ROUND)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])

    asked = packed_round(2)

    cross(engine, receiver, 1)
    room_while_crossing = receiver.has_room(asked)
    reserved_while_crossing = dict(receiver.reserved_bits_by_round)
    engine.run()

    assert room_while_crossing is False
    assert reserved_while_crossing == {(1, 1): LANDED_BITS_PER_ROUND}
    assert receiver.reserved_bits_by_round == {}
    assert receiver.has_room(asked) is False
    assert receiver.store.occupied_bits == LANDED_BITS_PER_ROUND
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
    receiver = room_side(
        engine, bits=LANDED_BITS_PER_ROUND, detection_events=former
    )
    receiver.store.open_operation(1)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])
    asked = packed_round(2)
    rooms_asked = []

    def ask_room() -> None:
        has_room = receiver.has_room(asked)
        rooms_asked.append(has_room)

    while_forming = LANDED_TICKS + 1
    cross(engine, receiver, 1)
    engine.schedule(while_forming, ask_room)
    engine.run()

    assert rooms_asked == [False]
    assert receiver.reserved_bits_by_round == {}
    assert receiver.store.occupied_bits == LANDED_BITS_PER_ROUND


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
    receiver = room_side(engine, bits=LANDED_BITS_PER_ROUND)
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

    with pytest.raises(RuntimeError, match="strong_buffer ended"):
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

    seat = "strong_buffer"
    assert carried.wire_bits == 4 * LANDED_BITS_PER_ROUND
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
        size_bits=LANDED_BITS_PER_ROUND,
        fragment_index=0,
        event_bits=1,
    )
    landing = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(
        landing, round_records.WINDOW_INPUT_ROUTE, LANDED_BITS_PER_ROUND
    )


def test_the_room_is_weighed_at_the_width_this_seat_stores():
    """Rounds of three raw bits formed here into one event take one bit each.

    gem5 makes room for a block at the size it will be stored at after
    its own compressor, not the packet's (src/mem/cache/base.cc:1678-1698).
    """
    engine = engine_module.Engine()
    settings = event_settings.DetectionEventSettings(
        formed_at=("strong_buffer",)
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
        formed_at=("strong_buffer",)
    )
    here = formation.SeatedFormation(None, settings)
    room_bits = 2 * LANDED_BITS_PER_ROUND + 2
    receiver = room_side(engine, bits=room_bits, detection_events=here)
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.STRONG_POOL_ID, 1
    )
    first = stated_event_round(1)
    second = stated_event_round(2)
    third = stated_event_round(3)
    fourth = stated_event_round(4)
    packets = (first.packet, second.packet, third.packet, fourth.packet)
    carried = round_records.EscalatedRegion.of(request_key, packets, 3)

    receiver.reserve_region(carried)

    assert receiver.reserved_bits_by_round == {
        (1, 1): LANDED_BITS_PER_ROUND,
        (1, 2): LANDED_BITS_PER_ROUND,
        (1, 3): 1,
        (1, 4): 1,
    }


def test_arrival_can_create_a_windows_first_round_hold() -> None:
    """gem5 base.cc services response targets before freeing their entry."""
    engine = engine_module.Engine()
    receiver = room_side(engine, bits=LANDED_BITS_PER_ROUND)
    receiver.store.open_operation(1)
    receiver.windows = _ArrivalConsumer(receiver.store)

    cross(engine, receiver, 1)
    engine.run()

    fragments = receiver.store.retained_fragments((1, 1))
    assert fragments is not None
    assert fragments[0].bits == (1, 0, 1)
    asked = packed_round(2)
    assert receiver.store.publication_tick((1, 1)) == LANDED_TICKS
    assert receiver.has_room(asked) is False


class _ArrivalConsumer:
    """A window consumer whose first claim is created by the arrival."""

    def __init__(self, store: syndrome_buffer_module.SyndromeBuffer) -> None:
        self.store = store

    def accept_room_round(
        self, store, operation_id: int, round_index: int
    ) -> None:
        del store
        reads = decoding_records.WindowReads((operation_id, 0))
        round_key = (operation_id, round_index)
        self.store.register_hold(reads, [round_key])


def _written_receiver(engine) -> receiver_module.SyndromeRoundReceiver:
    """The weak store's end: the controller writes its rounds."""
    return receiver_module.SyndromeRoundReceiver(engine, WEAK_STORE_INTAKE)


def _landed_receiver(engine) -> receiver_module.SyndromeRoundReceiver:
    """The strong store's end: its rounds land whole."""
    return receiver_module.SyndromeRoundReceiver(engine, STRONG_STORE_INTAKE)
