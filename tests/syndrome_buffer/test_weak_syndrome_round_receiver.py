"""The weak syndrome buffer's incoming port: room, and the slot a landing takes.

A round occupies a slot when its bits are in the store, at the landing
of the hop that carried them, and it is readable at that same instant:
ns-3 schedules the destination device's own Receive after the
propagation (the ns3-point-to-point copy of
point-to-point-channel.cc:88-92 into point-to-point-net-device.cc:324),
OMNeT++ takes ownership into the destination module and inserts inside
its handler (tmp/resources/omnetpp/src/sim/csimplemodule.cc:782-783,
:799, with queueinglib/Queue.cc:84-94), Ciw counts the individual in the
destination's own accept (Ciw/ciw/node.py:602 into :102-103), and Caune
2410.05202 lines 1243-1247 store the outcomes in the decoder sequencer's
memory only after the propagation. The sender still refuses before it
sends, so the room counts the bits of the writes in flight as taken
(gem5 src/dev/net/pktfifo.hh, `avail() = _maxsize - _size - _reserved`
with `reserve(len)`). The rest pin the store's two ordering laws: the
publication never precedes the store, and the window manager hears of a
round only once the store's record says it is readable.
"""

import decsim.config as config
import decsim.detector_error_model.detection_event_formation as formation
import decsim.engine as engine_module
import decsim.observe.log_writers as log_writers
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_syndrome_round_receiver,
)

LANDING_TICKS = 40_000
MEMORY_ROUTE = round_records.SyndromePacketRoute.feedback_memory_round(9)
# every round this file sends carries one fragment of two bits
BITS_PER_ROUND = 2


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


def _store(bits=None) -> syndrome_buffer_module.SyndromeBuffer:
    settings = syndrome_buffer_settings.SyndromeBufferSettings(bits=bits)
    return syndrome_buffer_module.SyndromeBuffer(settings)


def _receiver_with(engine, store, output=None, detection_events=None):
    """A receiver whose rounds arrive formed, unless a placement is given."""
    windows = _Windows(engine, store)
    receiver = weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(
        engine, store.settings
    )
    receiver.store = store
    receiver.windows = windows
    if detection_events is None:
        detection_events = formation.ControllerSideFormation(None, 0)
    receiver.detection_events = detection_events
    if output is not None:
        receiver.output = output
    return receiver, windows


def _cross(receiver, packed, landing_ticks=LANDING_TICKS):
    """One crossing: the room is taken at the send, the slot at the landing."""
    receiver.reserve_write(packed)
    receiver.engine.schedule(
        landing_ticks,
        lambda: receiver.receive_round(packed),
    )


def test_a_crossing_round_holds_no_slot_until_it_lands():
    engine = engine_module.Engine()
    store = _store()
    receiver, _windows = _receiver_with(engine, store)
    packed = _packed(1)

    _cross(receiver, packed)
    occupancy_while_crossing = store.occupancy
    readable_while_crossing = store.retained_fragments((1, 1))
    engine.run()

    assert occupancy_while_crossing == 0
    assert readable_while_crossing is None
    assert store.occupancy == 1
    assert store.retained_fragments((1, 1)) is not None


def test_the_reserved_bits_count_against_the_room_until_the_round_lands():
    engine = engine_module.Engine()
    store = _store(bits=BITS_PER_ROUND)
    receiver, _windows = _receiver_with(engine, store)
    packed = _packed(1)
    next_round = _packed(2)
    room_before = receiver.has_room(next_round)

    _cross(receiver, packed)
    room_while_crossing = receiver.has_room(next_round)
    reserved_while_crossing = dict(receiver.reserved_bits_by_round)
    engine.run()

    assert room_before is True
    assert room_while_crossing is False
    assert reserved_while_crossing == {(1, 1): BITS_PER_ROUND}
    assert receiver.has_room(next_round) is False
    assert receiver.reserved_bits_by_round == {}
    assert store.occupied_bits == BITS_PER_ROUND


def test_the_landing_stamps_the_publication_tick_of_the_end_it_reached():
    engine = engine_module.Engine()
    store = _store()
    receiver, _windows = _receiver_with(engine, store)
    packed = _packed(1)

    _cross(receiver, packed)
    engine.run()

    assert store.publication_tick((1, 1)) == LANDING_TICKS


def test_the_windows_hear_a_landed_round_only_once_it_is_published():
    engine = engine_module.Engine()
    store = _store()
    receiver, windows = _receiver_with(engine, store)
    packed = _packed(1)

    _cross(receiver, packed)
    engine.run()

    assert windows.published == [(LANDING_TICKS, (1, 1), LANDING_TICKS)]


def test_the_published_event_is_the_incoming_ports_own():
    engine = engine_module.Engine()
    store = _store()
    receiver, _windows = _receiver_with(engine, store)
    events = []
    receiver.trace.round_event.connect(events.append)
    packed = _packed(1)

    _cross(receiver, packed)
    engine.run()

    kinds_and_ticks = [(event.kind, event.tick) for event in events]
    assert kinds_and_ticks == [("PUBLISHED", LANDING_TICKS)]


def test_the_intake_copy_is_made_at_the_landing_by_this_end():
    engine = engine_module.Engine()
    store = _store()
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
            "weak syndrome buffer",
        )
    ]


def test_the_buffer_0_line_names_the_hop_the_round_arrived_by():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.io_line.connect(log.write)
    store = _store()
    receiver, _windows = _receiver_with(engine, store)
    silent_engine = engine_module.Engine()
    silent_log = log_writers.LogWriter()
    silent_engine.line.connect(silent_log.write)
    silent_store = _store()
    silent_receiver, _silent_windows = _receiver_with(
        silent_engine, silent_store
    )

    landed = _packed(1)
    _cross(receiver, landed)
    _cross(silent_receiver, landed)
    engine.run()
    silent_engine.run()

    assert silent_log.lines == []
    (line,) = log.lines
    assert line.endswith(
        "weak syndrome buffer: received round 1 of op 1 from "
        "controller_to_weak_buffer; defects {0}; holds op 1 rounds 1 (1)"
    )


def test_a_timing_only_round_takes_its_slot_here_and_is_never_published():
    """The round leaves the store, so the store's own port sends it."""
    engine = engine_module.Engine()
    store = _store()
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


def test_a_write_still_in_flight_at_the_end_of_a_run_is_a_failure():
    engine = engine_module.Engine()
    store = _store()
    receiver, _windows = _receiver_with(engine, store)

    crossing = _packed(1)
    receiver.reserve_write(crossing)

    try:
        receiver.check_settled()
    except RuntimeError as error:
        assert "1 controller_to_weak_buffer writes in flight" in str(error)
    else:
        raise AssertionError("an unfinished write settled")


def test_the_incoming_port_fills_the_declared_port():
    engine = engine_module.Engine()
    store = _store()
    receiver, _windows = _receiver_with(engine, store)

    assert isinstance(receiver, ports.WeakSyndromeRoundReceiver)


def test_write_cycles_move_every_reaction_point_by_the_store_periods():
    clocks = config.ClockSettings.from_yaml({"storage": 1.0})
    section = {"clock": "storage", "write_cycles": 3}
    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section,
        "weak_syndrome_buffer",
        clocks,
        syndrome_buffer_module.SYNDROME_BUFFERS,
    )
    free = declared_run.weak_only_run()
    charged = declared_run.weak_only_run(weak_syndrome_buffer=settings)
    free_ticks = declared_run.reaction_ticks(free)
    charged_ticks = declared_run.reaction_ticks(charged)
    paired = zip(charged_ticks, free_ticks)
    shifts = [charged_tick - free_tick for charged_tick, free_tick in paired]
    expected = 3 * settings.clock.period_ticks
    assert shifts == [expected] * 6


def test_a_priced_write_keeps_its_reservation_until_the_write_edge():
    engine = engine_module.Engine()
    engine.now = 1
    clock = config.Clock(10)
    settings = syndrome_buffer_settings.SyndromeBufferSettings(
        bits=BITS_PER_ROUND, clock=clock, write_cycles=3
    )
    store = syndrome_buffer_module.SyndromeBuffer(settings)
    receiver, windows = _receiver_with(engine, store)
    packed = _packed(1)
    receiver.reserve_write(packed)
    receiver.receive_round(packed)
    next_round = _packed(2)
    assert store.occupancy == 0
    assert receiver.reserved_bits_by_round == {(1, 1): BITS_PER_ROUND}
    assert receiver.has_room(next_round) is False
    assert windows.published == []

    engine.run()

    assert receiver.reserved_bits_by_round == {}
    assert store.occupancy == 1
    assert windows.published == [(40, (1, 1), 40)]


class _ChipFormer:
    """A formation table: every round's events are one set bit."""

    def form_round(self, operation_id, round_index, raw_bits):
        del operation_id, round_index, raw_bits
        return (1,)


def test_the_store_holds_the_events_when_the_chip_forms_them():
    """The hop's copy is the raw round; the store and the windows get events."""
    engine = engine_module.Engine()
    store = _store()
    former = _ChipFormer()
    on_the_chip = formation.WeakSyndromeBufferSideFormation(former, 0)
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


def test_the_chips_formation_cycles_are_added_to_the_write_cycles():
    engine = engine_module.Engine()
    clock = config.Clock(10)
    formation_cycles = 5
    write_cycles = 3
    settings = syndrome_buffer_settings.SyndromeBufferSettings(
        clock=clock,
        write_cycles=write_cycles,
        detection_event_cycles_per_round=formation_cycles,
    )
    store = syndrome_buffer_module.SyndromeBuffer(settings)
    former = _ChipFormer()
    on_the_chip = formation.WeakSyndromeBufferSideFormation(former, 0)
    receiver, windows = _receiver_with(
        engine, store, detection_events=on_the_chip
    )
    packed = _packed(1)

    receiver.reserve_write(packed)
    receiver.receive_round(packed)
    engine.run()

    charged_cycles = formation_cycles + write_cycles
    expected = charged_cycles * clock.period_ticks
    assert windows.published == [(expected, (1, 1), expected)]


def test_the_store_is_asked_for_the_round_beside_the_rounds_in_flight():
    """gem5 asks the responder with the packet (src/mem/port.hh:268)."""
    engine = engine_module.Engine()
    store = _RoomAskingStore()
    receiver, _windows = _receiver_with(engine, store)
    crossing = _packed(1)
    asked = _packed(2)
    receiver.reserve_write(crossing)

    receiver.has_room(asked)

    assert store.asked[-1] == ((1, 2), BITS_PER_ROUND, {(1, 1): BITS_PER_ROUND})


class _RoomAskingStore(syndrome_buffer_module.SyndromeBuffer):
    """An unbounded store recording every room question it is asked."""

    def __init__(self) -> None:
        settings = syndrome_buffer_settings.SyndromeBufferSettings()
        syndrome_buffer_module.SyndromeBuffer.__init__(self, settings)
        self.asked = []

    def has_room(self, round_key, bits, reserved_bits_by_round):
        reserved = dict(reserved_bits_by_round)
        self.asked.append((round_key, bits, reserved))
        return True
