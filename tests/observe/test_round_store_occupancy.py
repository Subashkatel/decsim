"""The occupancy listener: the integral equals the residence sum, exactly.

Referent: the sample-path form of Little's law (Stidham 1974, "A last
word on L = lambda W"): over a window in which every arrival departs,
the integral of the number in the system equals the sum of the times
each one spent there. The listener hears round_stored and round_released
from a RoundStore and needs nothing else; here a random trace of stores
and releases drives a real store.
"""

import functools
import random

import decsim.engine as engine_module
import decsim.message as message
import decsim.observe.round_store_occupancy as round_store_occupancy
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings


def packet(round_index: int) -> message.SyndromeRoundPacket:
    fragment = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1,),
        size_bits=1,
        fragment_index=0,
    )
    return message.SyndromeRoundPacket(1, round_index, (fragment,))


def run_random_trace(seed: int):
    """Stores at random ticks, each released a random time later."""
    generator = random.Random(seed)
    engine = engine_module.Engine(verbose=False)
    listener = round_store_occupancy.RoundStoreOccupancy(engine)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings, listener=listener)
    residences = []
    tick = 0
    for round_index in range(1, 41):
        tick += generator.randint(0, 5)
        residence = generator.randint(0, 20)
        residences.append(residence)
        stored = packet(round_index)
        accept = functools.partial(
            store.accept_packed_round, stored, publication_tick=None
        )
        release = functools.partial(store.release_round, (1, round_index))
        release_tick = tick + residence
        engine.schedule(tick, accept)
        engine.schedule(release_tick, release)
    engine.run()
    return listener, residences


def test_the_occupancy_integral_equals_the_residence_sum_over_random_traces():
    for seed in range(10):
        listener, residences = run_random_trace(seed)
        assert listener.integral == listener.residence_sum, seed
        assert listener.residence_sum == sum(residences), seed
        assert listener.arrivals == 40, seed


def test_the_peak_is_the_most_rounds_stored_at_once():
    engine = engine_module.Engine(verbose=False)
    listener = round_store_occupancy.RoundStoreOccupancy(engine)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings, listener=listener)
    first = packet(1)
    second = packet(2)
    third = packet(3)

    store.accept_packed_round(first, publication_tick=None)
    store.accept_packed_round(second, publication_tick=None)
    store.release_round((1, 1))
    store.accept_packed_round(third, publication_tick=None)

    assert listener.peak == 2
    assert listener.arrivals == 3


def test_the_time_average_is_the_integral_over_the_span():
    engine = engine_module.Engine(verbose=False)
    listener = round_store_occupancy.RoundStoreOccupancy(engine)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings, listener=listener)
    first = packet(1)
    accept = functools.partial(
        store.accept_packed_round, first, publication_tick=None
    )
    release = functools.partial(store.release_round, (1, 1))
    engine.schedule(10, accept)
    engine.schedule(40, release)

    engine.run()

    assert listener.integral == 30
    assert listener.time_average() == 1.0
