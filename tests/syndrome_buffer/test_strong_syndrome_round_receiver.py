"""The room-side end: room counts writes in flight, the landing stores.

Referent: gem5's queue counts its reserved entries as taken before they
are allocated (gem5 src/mem/cache/queue.hh:150-153, isFull over
allocated plus reserve); this end counts a round crossing toward it the
same way, reserved by the controller before the round leaves. The
crossing itself is the controller's send and is tested where it is
executed (tests/controller/test_syndrome_round_sender.py).

The whole-run law at the end of the file places that landing in the
pipeline: under a strong-primary policy the landing is what makes a
window ready, on the declared card of tests/declared_run.py.
"""

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver,
)

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

    def round_stored(self, round_key, _packet):
        self.stored.append(round_key)

    def round_released(self, round_key):
        self.released.append(round_key)


def room_side(engine, rounds=None, listener=None, windows=None):
    store_settings = syndrome_buffer_settings.SyndromeBufferSettings(
        rounds=rounds
    )
    store = syndrome_buffer_module.SyndromeBuffer(store_settings)
    if listener is not None:
        store.trace.round_stored.connect(listener.round_stored)
        store.trace.round_released.connect(listener.round_released)
    receiver = strong_syndrome_round_receiver.StrongSyndromeRoundReceiver(
        engine
    )
    receiver.store = store
    if windows is not None:
        receiver.windows = windows
    return receiver


def cross(engine, receiver, round_index: int) -> None:
    """One round on its way over the crossing, landing after 0.5 us.

    The controller reserves the room and sends; the send itself is the
    controller's and is tested at
    tests/controller/test_syndrome_round_sender.py, so this file stands
    the round up at the landing tick instead.
    """
    receiver.reserve_write()
    landing = packet(round_index)
    engine.schedule(LANDING_TICKS, lambda: receiver.receive_round(landing, 3))


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


def test_the_receiver_counts_a_write_in_flight_as_room_taken():
    engine = engine_module.Engine()
    receiver = room_side(engine, rounds=1)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])

    cross(engine, receiver, 1)
    room_while_crossing = receiver.has_room()
    engine.run()

    assert room_while_crossing is False
    assert receiver.writes_in_flight == 0
    assert receiver.has_room() is False
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
    receiver = room_side(engine, rounds=1)
    receiver.store.open_operation(1)
    room_before = receiver.has_room()

    cross(engine, receiver, 1)
    receiver.store.close_operation(1)
    engine.run()

    assert room_before is True
    assert receiver.store.retained_fragments((1, 1)) is None
    assert receiver.store.occupancy == 0
    assert receiver.store.has_operation(1) is False
    assert receiver.writes_in_flight == 0
    assert receiver.has_room() is True
    receiver.check_settled()


def test_settlement_reports_a_write_still_in_flight():
    engine = engine_module.Engine()
    receiver = room_side(engine)
    reads = decoding_records.WindowReads((1, 0))
    receiver.store.register_hold(reads, [(1, 1)])
    cross(engine, receiver, 1)

    with pytest.raises(RuntimeError, match="1 controller_to_strong_buffer"):
        receiver.check_settled()


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
