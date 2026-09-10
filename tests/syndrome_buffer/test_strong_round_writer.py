"""The room-side end: room counts writes in flight, the landing stores.

Referent: gem5's queue counts its reserved entries as taken before they
are allocated (gem5 src/mem/cache/queue.hh:150-153, isFull over
allocated plus reserve); this end counts a round crossing toward it the
same way, reserved by the controller before the round leaves. The
crossing itself is the controller's send and is tested where it is
executed (tests/controller/test_round_writes.py).

The whole-run law at the end of the file places that landing in the
pipeline: under a strong-primary policy the landing is what makes a
window ready, on the declared card of tests/declared_run.py.
"""

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer
import tests.declared_run as declared_run

LANDING_TICKS = config.microseconds_to_ticks(0.5)


def packet(round_index: int) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0, 1),
        size_bits=3,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(1, round_index, (fragment,))


class RecordingListener:
    def __init__(self):
        self.stored = []
        self.released = []

    def round_stored(self, round_key, _packet):
        self.stored.append(round_key)

    def round_released(self, round_key):
        self.released.append(round_key)


def room_side(engine, rounds=None, listener=None, on_round_stored=None):
    store_settings = round_store_settings.RoundStoreSettings(rounds=rounds)
    store = round_store_module.RoundStore(store_settings)
    if listener is not None:
        store.trace.round_stored.connect(listener.round_stored)
        store.trace.round_released.connect(listener.round_released)
    return strong_round_writer.StrongRoundWriter(
        engine, store, on_round_stored=on_round_stored
    )


def cross(engine, writer, round_index: int) -> None:
    """One round on its way over the crossing, landing after 0.5 us.

    The controller reserves the room and sends; the send itself is the
    controller's and is tested at tests/controller/test_round_writes.py,
    so this file stands the round up at the landing tick instead.
    """
    writer.reserve_write()
    landing = packet(round_index)
    engine.schedule(LANDING_TICKS, lambda: writer.receive_round(landing, 3))


def test_a_write_lands_after_the_crossing_and_the_listener_hears_it_once():
    engine = engine_module.Engine()
    listener = RecordingListener()
    stored = []
    writer = room_side(
        engine,
        listener=listener,
        on_round_stored=lambda *key: stored.append(key),
    )
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])

    cross(engine, writer, 1)
    in_flight = writer.store.retained_fragments((1, 1))
    engine.run()

    assert in_flight is None
    assert writer.store.publication_tick((1, 1)) == LANDING_TICKS
    assert listener.stored == [(1, 1)]
    assert stored == [(1, 1)]


def test_the_writer_counts_a_write_in_flight_as_room_taken():
    engine = engine_module.Engine()
    writer = room_side(engine, rounds=1)
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])

    cross(engine, writer, 1)
    room_while_crossing = writer.has_room()
    engine.run()

    assert room_while_crossing is False
    assert writer.writes_in_flight == 0
    assert writer.has_room() is False
    assert writer.store.occupancy == 1


def test_a_round_whose_readers_resolved_while_crossing_is_dropped_at_landing():
    engine = engine_module.Engine()
    writer = room_side(engine)
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])

    cross(engine, writer, 1)
    writer.store.release_hold(reads)
    engine.run()

    assert writer.store.retained_fragments((1, 1)) is None
    writer.check_settled()


def test_a_round_that_lands_after_its_operation_closed_is_dropped():
    engine = engine_module.Engine()
    writer = room_side(engine, rounds=1)
    writer.store.open_operation(1)
    room_before = writer.has_room()

    cross(engine, writer, 1)
    writer.store.close_operation(1)
    engine.run()

    assert room_before is True
    assert writer.store.retained_fragments((1, 1)) is None
    assert writer.store.occupancy == 0
    assert writer.store.has_operation(1) is False
    assert writer.writes_in_flight == 0
    assert writer.has_room() is True
    writer.check_settled()


def test_settlement_reports_a_write_still_in_flight():
    engine = engine_module.Engine()
    writer = room_side(engine)
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])
    cross(engine, writer, 1)

    with pytest.raises(RuntimeError, match="1 controller_to_strong_buffer"):
        writer.check_settled()


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
