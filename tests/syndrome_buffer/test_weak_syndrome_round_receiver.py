"""The weak syndrome buffer's incoming port: room, and the slot a landing takes.

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
"""

import pytest

import decsim.config as config
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.records.rounds as round_records
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


def _store(engine, bits=None) -> syndrome_buffer_module.SyndromeBuffer:
    settings = syndrome_buffer_module.SyndromeBufferSettings(bits=bits)
    return syndrome_buffer_module.SyndromeBuffer(settings, engine)


def _receiver_with(engine, store, output=None, detection_events=None):
    """A receiver whose rounds arrive formed, unless a placement is given."""
    windows = _Windows(engine, store)
    receiver = weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(engine)
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
        lambda: receiver.receive_round(packed, _no_sender_counts_it),
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
            "weak syndrome buffer",
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

    with pytest.raises(RuntimeError, match="weak syndrome buffer ended with 1"):
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
    receiver.receive_round(packed, sender_hears)
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
    receiver.receive_round(packed, _no_sender_counts_it)
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
        recipe = detector_formation.DetectorRecipe(
            detector_index=0,
            round_index=1,
            kind=detector_formation.LayerKind.PREPARATION,
            records=((1, 0),),
            reference_parity=0,
            coordinates=(),
        )
        return detector_formation.FormationTable(
            round_count=1,
            packet_width_by_round={1: 2},
            readout_slot_start=None,
            detectors=(recipe,),
            observables=(),
        )


def _on_the_chip(clock=None, latency_cycles=0):
    """The former seated at this store's receiving end."""
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_syndrome_buffer",),
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
    receiver.receive_round(packed, _no_sender_counts_it)
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
