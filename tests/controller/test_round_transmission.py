"""The transmitter: a stored round leaves at the write and lands on its route.

A window-input round on a priced controller_to_weak_buffer hop is
published at delivery, where the store is stamped; on a free hop the
publication is the storage. A feedback-memory round tells the windows at
delivery and frees its slot. The sender never waits for a landing:
two memory rounds one QEC cycle apart on a 5 us weak_buffer_to_weak_decoder
land one cycle apart (Yang et al. 2605.04892 and Google 2408.13687 stream
every round; gem5 src/dev/dma_device.cc transmitList; ns-3
point-to-point-net-device.cc TransmitComplete). The link law itself is
the channel's (tests/links/test_channel.py).
"""

import dataclasses

import decsim.config as config
import decsim.controller.round_transmission as round_transmission
import decsim.engine as engine_module
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.message as message
import decsim.observe.round_events as round_events
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings

CWB_TICKS = config.microseconds_to_ticks(0.25)
WBD_TICKS = config.microseconds_to_ticks(5.0)
CYCLE_TICKS = config.microseconds_to_ticks(1.0)
MEMORY_ROUTE = message.SyndromePacketRoute.feedback_memory_round(7)


class RecordingWindows:
    def __init__(self, engine):
        self.engine = engine
        self.published = []
        self.memory_rounds = []

    def accept_window_input(self, packet):
        self.published.append((self.engine.now, packet.round_index))

    def accept_feedback_memory_round(self, source_operation_id):
        self.memory_rounds.append((self.engine.now, source_operation_id))


def packed(round_index, route=message.WINDOW_INPUT_ROUTE):
    fragment = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    packet = message.SyndromeRoundPacket(1, round_index, (fragment,))
    return message.PackedRound(packet, route, 2)


def priced_cwb_profile():
    reference = link_profiles.logical_reference_profile()
    return link_profiles.with_controller_to_weak_buffer_path(
        reference,
        latency_microseconds=0.25,
        aggregate_bits_per_microsecond=None,
        source="test",
    )


def five_microsecond_wbd_profile():
    reference = link_profiles.logical_reference_profile()
    edge = reference.weak_buffer_to_weak_decoder
    channel = link_settings.ChannelSettings(
        edge.channel.name, WBD_TICKS, None, "test"
    )
    path = link_settings.PathSettings(
        channel, edge.default_payload, edge.actual_payload_source
    )
    return dataclasses.replace(reference, weak_buffer_to_weak_decoder=path)


def transmitter_with(engine, profile):
    links = fabric_module.LinkFabric(profile, engine)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings)
    windows = RecordingWindows(engine)
    recorder = round_events.RoundEventRecorder(engine)
    transmitter = round_transmission.RoundTransmitter(
        engine, links, store, windows, recorder
    )
    return transmitter, store, windows, recorder


def test_a_priced_hop_publishes_at_delivery_and_stamps_the_store():
    engine = engine_module.Engine(verbose=False)
    profile = priced_cwb_profile()
    transmitter, store, windows, recorder = transmitter_with(engine, profile)
    first = packed(1)
    stored_tick = transmitter.publication_tick_at_storage(first.route)
    store.accept_packed_round(first.packet, publication_tick=stored_tick)

    transmitter.send(first)
    engine.run()

    assert stored_tick is None
    assert store.publication_tick((1, 1)) == CWB_TICKS
    assert windows.published == [(CWB_TICKS, 1)]
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("CWB_SENT", 0), ("PUBLISHED", CWB_TICKS)]
    assert transmitter.in_flight == 0


def test_a_free_hop_publishes_as_the_round_is_stored():
    engine = engine_module.Engine(verbose=False)
    profile = link_profiles.logical_reference_profile()
    transmitter, store, windows, recorder = transmitter_with(engine, profile)
    first = packed(1)
    stored_tick = transmitter.publication_tick_at_storage(first.route)
    store.accept_packed_round(first.packet, publication_tick=stored_tick)

    transmitter.send(first)
    engine.run()

    assert stored_tick == 0
    assert windows.published == [(0, 1)]
    assert recorder.events == []


def test_a_memory_round_tells_the_windows_at_delivery_and_frees_its_slot():
    engine = engine_module.Engine(verbose=False)
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, recorder = transmitter_with(engine, profile)
    memory_round = packed(1, route=MEMORY_ROUTE)
    stored_tick = transmitter.publication_tick_at_storage(memory_round.route)
    store.accept_packed_round(memory_round.packet, publication_tick=stored_tick)

    transmitter.send(memory_round)
    engine.run()

    assert stored_tick is None
    assert windows.memory_rounds == [(WBD_TICKS, 7)]
    assert store.occupancy == 0
    kinds_and_ticks = [(event.kind, event.tick) for event in recorder.events]
    assert kinds_and_ticks == [("FEEDBACK_MEMORY_DELIVERED", WBD_TICKS)]


def test_memory_rounds_pipeline_onto_the_link_without_a_landing_wait():
    engine = engine_module.Engine(verbose=False)
    profile = five_microsecond_wbd_profile()
    transmitter, store, windows, _recorder = transmitter_with(engine, profile)
    first = packed(1, route=MEMORY_ROUTE)
    second = packed(2, route=MEMORY_ROUTE)

    def send(memory_round):
        store.accept_packed_round(memory_round.packet, publication_tick=None)
        transmitter.send(memory_round)

    engine.schedule(0, lambda: send(first))
    engine.schedule(CYCLE_TICKS, lambda: send(second))
    engine.run()

    assert windows.memory_rounds == [
        (WBD_TICKS, 7),
        (CYCLE_TICKS + WBD_TICKS, 7),
    ]
