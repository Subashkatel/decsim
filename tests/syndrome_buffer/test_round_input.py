"""A store's incoming port: what it does with a round that lands in it.

The receiving end handles the landing. gem5's requesting port hands the
packet to the peer's own recvTimingReq
(tmp/resources/gem5/src/mem/port.hh:603-614, whose
src/mem/protocol/timing.cc:49-53 makes that call), OMNeT++ gives the
message to the destination module before its handleMessage runs
(tmp/resources/omnetpp/src/sim/csimplemodule.cc:777-799), and ns-3
schedules Receive on the destination device (the ns3-point-to-point
copy of point-to-point-channel.cc:88-92). The laws pinned here are
Buffer 0's, as the buffer contract states them
(validation/responsibility_audit_2026_08_30/buffer_contract.md): the
publication never precedes the store, and the window manager hears of a
round only once the store's record says it is readable.
"""

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_input as round_input
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings

LANDING_TICKS = 40_000


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


def _packed(round_index: int) -> round_records.PackedRound:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
    route = round_records.WINDOW_INPUT_ROUTE
    return round_records.PackedRound(packet, route, 2)


def _store() -> round_store_module.RoundStore:
    settings = round_store_settings.RoundStoreSettings()
    return round_store_module.RoundStore(settings)


def test_the_windows_hear_a_landed_round_only_once_it_is_published():
    engine = engine_module.Engine()
    store = _store()
    windows = _Windows(engine, store)
    store_input = round_input.RoundStoreInput(engine, store, None, windows)
    packed = _packed(1)
    store.accept_packed_round(packed.packet, publication_tick=None)

    engine.schedule(LANDING_TICKS, lambda: store_input.receive_round(packed))
    engine.run()

    assert windows.published == [(LANDING_TICKS, (1, 1), LANDING_TICKS)]


def test_the_landing_stamps_the_publication_tick_of_the_end_it_reached():
    engine = engine_module.Engine()
    store = _store()
    windows = _Windows(engine, store)
    store_input = round_input.RoundStoreInput(engine, store, None, windows)
    packed = _packed(1)
    store.accept_packed_round(packed.packet, publication_tick=None)

    engine.schedule(LANDING_TICKS, lambda: store_input.receive_round(packed))
    engine.run()

    assert store.publication_tick((1, 1)) == LANDING_TICKS


def test_the_published_event_is_the_incoming_ports_own():
    engine = engine_module.Engine()
    store = _store()
    windows = _Windows(engine, store)
    store_input = round_input.RoundStoreInput(engine, store, None, windows)
    events = []
    store_input.trace.round_event.connect(events.append)
    packed = _packed(1)
    store.accept_packed_round(packed.packet, publication_tick=None)

    engine.schedule(LANDING_TICKS, lambda: store_input.receive_round(packed))
    engine.run()

    kinds_and_ticks = [(event.kind, event.tick) for event in events]
    assert kinds_and_ticks == [("PUBLISHED", LANDING_TICKS)]


def test_a_timing_only_round_is_asked_of_the_stores_outgoing_port():
    """The round leaves the store, so the store's own port sends it."""
    engine = engine_module.Engine()
    store = _store()
    windows = _Windows(engine, store)
    output = _Output()
    store_input = round_input.RoundStoreInput(engine, store, output, windows)
    packed = _packed(2)
    delivered = []

    store_input.send_memory_round(packed, lambda: delivered.append(True))

    assert output.sent == [(1, 2)]
    assert delivered == [True]


def test_the_incoming_port_fills_the_declared_port():
    engine = engine_module.Engine()
    store = _store()
    windows = _Windows(engine, store)
    store_input = round_input.RoundStoreInput(engine, store, None, windows)

    assert isinstance(store_input, ports.RoundStoreInput)
