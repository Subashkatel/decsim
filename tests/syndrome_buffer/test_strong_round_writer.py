"""The strong writer: one crossing per round, room counts writes in flight.

Referent: gem5's queue counts its reserved entries as taken before they
are allocated (tmp/resources/gem5/src/mem/cache/queue.hh:150-153,
isFull over allocated plus reserve); the writer counts a round crossing
the link the same way. The link law itself is the channel's
(tests/links/test_channel.py); here a priced controller_to_strong_buffer
hop of 0.5 us lands the round 0.5 us after the write.
"""

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.message as message
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer

LANDING_TICKS = config.microseconds_to_ticks(0.5)


def packet(round_index: int) -> message.SyndromeRoundPacket:
    fragment = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0, 1),
        size_bits=3,
        fragment_index=0,
    )
    return message.SyndromeRoundPacket(1, round_index, (fragment,))


def attribution(round_index: int) -> message.TransferAttribution:
    return message.TransferAttribution(
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

    def round_stored(self, round_key):
        self.stored.append(round_key)

    def round_released(self, round_key):
        self.released.append(round_key)


def priced_writer(engine, rounds=None, listener=None, on_round_stored=None):
    reference = link_profiles.logical_reference_profile()
    settings = link_profiles.with_controller_to_strong_buffer_path(
        reference,
        latency_microseconds=0.5,
        aggregate_bits_per_microsecond=None,
        source="test",
    )
    link = fabric_module.LinkFabric(settings, engine)
    store_settings = round_store_settings.RoundStoreSettings(rounds=rounds)
    store = round_store_module.RoundStore(store_settings, listener=listener)
    return strong_round_writer.StrongRoundWriter(
        engine, link, store, on_round_stored=on_round_stored
    )


def free_writer(engine, on_round_stored=None):
    reference = link_profiles.logical_reference_profile()
    link = fabric_module.LinkFabric(reference, engine)
    store_settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(store_settings)
    return strong_round_writer.StrongRoundWriter(
        engine, link, store, on_round_stored=on_round_stored
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
    writer.store.register_hold("reader", [(1, 1)])

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
    writer.store.register_hold("reader", [(1, 1)])

    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)
    room_while_crossing = writer.has_room()
    engine.run()

    assert room_while_crossing is False
    assert writer.writes_in_flight == 0
    assert writer.has_room() is False
    assert writer.store.occupancy == 1


def test_an_unpriced_crossing_stores_at_the_write():
    engine = engine_module.Engine()
    stored = []
    writer = free_writer(
        engine, on_round_stored=lambda *key: stored.append(key)
    )
    writer.store.register_hold("reader", [(1, 1)])

    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)

    assert writer.store.publication_tick((1, 1)) == 0
    assert stored == [(1, 1)]


def test_a_round_whose_readers_resolved_while_crossing_is_dropped_at_landing():
    engine = engine_module.Engine()
    writer = priced_writer(engine)
    writer.store.register_hold("reader", [(1, 1)])

    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)
    writer.store.release_hold("reader")
    engine.run()

    assert writer.store.retained_fragments((1, 1)) is None
    writer.check_settled()


def test_settlement_reports_a_write_still_in_flight():
    engine = engine_module.Engine()
    writer = priced_writer(engine)
    writer.store.register_hold("reader", [(1, 1)])
    first = packet(1)
    first_attribution = attribution(1)
    writer.write(first, packet_bits=3, attribution=first_attribution)

    with pytest.raises(RuntimeError, match="1 controller_to_strong_buffer"):
        writer.check_settled()
