"""The strong writer: one crossing per round, room counts writes in flight.

Referent: gem5's queue counts its reserved entries as taken before they
are allocated (gem5 src/mem/cache/queue.hh:150-153,
isFull over allocated plus reserve); the writer counts a round crossing
the link the same way. The link law itself is the channel's
(tests/links/test_channel.py); here a priced controller_to_strong_buffer
hop of 0.5 us lands the round 0.5 us after the write.

The whole-run law at the end of the file places that landing in the
pipeline: under a strong-primary policy the landing is what makes a
window ready, on the declared card of tests/declared_run.py.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
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


def attribution(round_index: int) -> transfer_records.TransferAttribution:
    return transfer_records.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


class RecordingListener:
    def __init__(self):
        self.stored = []
        self.released = []

    def round_stored(self, round_key, _packet):
        self.stored.append(round_key)

    def round_released(self, round_key):
        self.released.append(round_key)


def priced_writer(engine, rounds=None, listener=None, on_round_stored=None):
    reference = link_profiles.logical_reference_profile()
    crossing = half_microsecond_crossing(reference)
    settings = dataclasses.replace(
        reference, controller_to_strong_buffer=crossing
    )
    link = fabric_module.LinkFabric(settings, engine)
    store_settings = round_store_settings.RoundStoreSettings(rounds=rounds)
    store = round_store_module.RoundStore(store_settings)
    if listener is not None:
        store.trace.round_stored.connect(listener.round_stored)
        store.trace.round_released.connect(listener.round_released)
    return strong_round_writer.StrongRoundWriter(
        engine, link, store, on_round_stored=on_round_stored
    )


def half_microsecond_crossing(reference):
    """The room-side write at half a microsecond, no rate bound."""
    base = reference.controller_to_strong_buffer
    latency_ticks = config.microseconds_to_ticks(0.5)
    channel = link_settings.ChannelSettings(
        base.channel.name, latency_ticks, None, "test"
    )
    return link_settings.PathSettings(
        channel, base.default_payload, base.actual_payload_source
    )


def test_a_write_lands_after_the_crossing_and_the_listener_hears_it_once():
    engine = engine_module.Engine()
    listener = RecordingListener()
    stored = []
    writer = priced_writer(
        engine,
        listener=listener,
        on_round_stored=lambda *key: stored.append(key),
    )
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])

    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)
    in_flight = writer.store.retained_fragments((1, 1))
    engine.run()

    assert in_flight is None
    assert writer.store.publication_tick((1, 1)) == LANDING_TICKS
    assert listener.stored == [(1, 1)]
    assert stored == [(1, 1)]


def test_the_writer_counts_a_write_in_flight_as_room_taken():
    engine = engine_module.Engine()
    writer = priced_writer(engine, rounds=1)
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])

    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)
    room_while_crossing = writer.has_room()
    engine.run()

    assert room_while_crossing is False
    assert writer.writes_in_flight == 0
    assert writer.has_room() is False
    assert writer.store.occupancy == 1


def test_a_round_whose_readers_resolved_while_crossing_is_dropped_at_landing():
    engine = engine_module.Engine()
    writer = priced_writer(engine)
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])

    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)
    writer.store.release_hold(reads)
    engine.run()

    assert writer.store.retained_fragments((1, 1)) is None
    writer.check_settled()


def test_settlement_reports_a_write_still_in_flight():
    engine = engine_module.Engine()
    writer = priced_writer(engine)
    reads = decoding_records.WindowReads((1, 0))
    writer.store.register_hold(reads, [(1, 1)])
    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)

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
