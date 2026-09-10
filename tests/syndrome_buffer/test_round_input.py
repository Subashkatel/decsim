"""Buffer 0's incoming port: its room, and the slot a landing takes.

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
sends, so the room counts the writes in flight as taken (gem5
src/mem/cache/queue.hh:150-153 with the reserve at :87-93; Ruby
MessageBuffer.cc:181). The rest are the laws of the buffer contract
(validation/responsibility_audit_2026_08_30/buffer_contract.md): the
publication never precedes the store, and the window manager hears of a
round only once the store's record says it is readable.
"""

import decsim.engine as engine_module
import decsim.observe.log_writers as log_writers
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_input as round_input
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings

LANDING_TICKS = 40_000
MEMORY_ROUTE = round_records.SyndromePacketRoute.feedback_memory_round(9)


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
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    return round_records.PackedRound(packet, route, 2)


def _store(rounds=None) -> round_store_module.RoundStore:
    settings = round_store_settings.RoundStoreSettings(rounds=rounds)
    return round_store_module.RoundStore(settings)


def _input_with(engine, store, output=None):
    windows = _Windows(engine, store)
    return round_input.RoundStoreInput(engine, store, output, windows), windows


def _cross(store_input, packed, landing_ticks=LANDING_TICKS):
    """One crossing: the room is taken at the send, the slot at the landing."""
    store_input.reserve_write()
    store_input.engine.schedule(
        landing_ticks, lambda: store_input.receive_round(packed)
    )


def test_a_crossing_round_holds_no_slot_until_it_lands():
    engine = engine_module.Engine()
    store = _store()
    store_input, _windows = _input_with(engine, store)
    packed = _packed(1)

    _cross(store_input, packed)
    occupancy_while_crossing = store.occupancy
    readable_while_crossing = store.retained_fragments((1, 1))
    engine.run()

    assert occupancy_while_crossing == 0
    assert readable_while_crossing is None
    assert store.occupancy == 1
    assert store.retained_fragments((1, 1)) is not None


def test_the_room_counts_the_write_in_flight_against_the_capacity():
    engine = engine_module.Engine()
    store = _store(rounds=1)
    store_input, _windows = _input_with(engine, store)
    packed = _packed(1)
    room_before = store_input.has_room()

    _cross(store_input, packed)
    room_while_crossing = store_input.has_room()
    engine.run()

    assert room_before is True
    assert room_while_crossing is False
    assert store_input.has_room() is False
    assert store_input.writes_in_flight == 0


def test_the_landing_stamps_the_publication_tick_of_the_end_it_reached():
    engine = engine_module.Engine()
    store = _store()
    store_input, _windows = _input_with(engine, store)
    packed = _packed(1)

    _cross(store_input, packed)
    engine.run()

    assert store.publication_tick((1, 1)) == LANDING_TICKS


def test_the_windows_hear_a_landed_round_only_once_it_is_published():
    engine = engine_module.Engine()
    store = _store()
    store_input, windows = _input_with(engine, store)
    packed = _packed(1)

    _cross(store_input, packed)
    engine.run()

    assert windows.published == [(LANDING_TICKS, (1, 1), LANDING_TICKS)]


def test_the_published_event_is_the_incoming_ports_own():
    engine = engine_module.Engine()
    store = _store()
    store_input, _windows = _input_with(engine, store)
    events = []
    store_input.trace.round_event.connect(events.append)
    packed = _packed(1)

    _cross(store_input, packed)
    engine.run()

    kinds_and_ticks = [(event.kind, event.tick) for event in events]
    assert kinds_and_ticks == [("PUBLISHED", LANDING_TICKS)]


def test_the_intake_copy_is_made_at_the_landing_by_this_end():
    engine = engine_module.Engine()
    store = _store()
    store_input, _windows = _input_with(engine, store)
    copies = []

    def copy_made(round_key, bits, source, destination) -> None:
        made_at = engine.now
        copies.append((made_at, round_key, bits, source, destination))

    store_input.trace.copy_made.connect(copy_made)
    packed = _packed(1)

    _cross(store_input, packed)
    engine.run()

    assert copies == [
        (LANDING_TICKS, (1, 1), 2, "controller assembler", "Buffer 0")
    ]


def test_the_buffer_0_line_names_the_hop_the_round_arrived_by():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.io_line.connect(log.write)
    store = _store()
    store_input, _windows = _input_with(engine, store)
    silent_engine = engine_module.Engine()
    silent_log = log_writers.LogWriter()
    silent_engine.line.connect(silent_log.write)
    silent_store = _store()
    silent_input, _silent_windows = _input_with(silent_engine, silent_store)

    landed = _packed(1)
    _cross(store_input, landed)
    _cross(silent_input, landed)
    engine.run()
    silent_engine.run()

    assert silent_log.lines == []
    (line,) = log.lines
    assert line.endswith(
        "Buffer 0: received round 1 of op 1 from controller_to_weak_buffer; "
        "defects {0}; holds op 1 rounds 1 (1)"
    )


def test_a_timing_only_round_takes_its_slot_here_and_is_never_published():
    """The round leaves the store, so the store's own port sends it."""
    engine = engine_module.Engine()
    store = _store()
    output = _Output()
    store_input, windows = _input_with(engine, store, output)
    packed = _packed(2, route=MEMORY_ROUTE)
    delivered = []

    store_input.reserve_write()
    store_input.send_memory_round(packed, lambda: delivered.append(True))

    assert output.sent == [(1, 2)]
    assert delivered == [True]
    assert store.occupancy == 1
    assert store.publication_tick((1, 2)) is None
    assert windows.published == []


def test_a_write_still_in_flight_at_the_end_of_a_run_is_a_failure():
    engine = engine_module.Engine()
    store = _store()
    store_input, _windows = _input_with(engine, store)

    store_input.reserve_write()

    try:
        store_input.check_settled()
    except RuntimeError as error:
        assert "1 controller_to_weak_buffer writes in flight" in str(error)
    else:
        raise AssertionError("an unfinished write settled")


def test_the_incoming_port_fills_the_declared_port():
    engine = engine_module.Engine()
    store = _store()
    store_input, _windows = _input_with(engine, store)

    assert isinstance(store_input, ports.RoundStoreInput)
