"""A factory delivers a magic state when one is ready, and no earlier.

Sources: Silva et al. 2411.04270 Sec. II B (a multi-level factory: level 0
prepares, each level consumes inputs_per_round lower states per round of
logical_cycles_per_round * distance QEC rounds, and a request pulls demand
down the chain); Litinski 1905.06903 (the 15-to-1 protocol's 11 commuting
rotations, one correction decode each). The supply stall is the delay
between a request and its delivery.
"""

from decsim.engine import Engine
from decsim.qpu.magic_state_factories import (
    DistillationFactory,
    DistillLevel,
    InfiniteFactory,
    MultiLevelDistillationFactory,
)


class DecodeLog:
    def __init__(self, engine, latency_ticks):
        self.engine = engine
        self.latency_ticks = latency_ticks
        self.submitted = []

    def submit_decode(self, round_count, on_done, label):
        self.submitted.append((self.engine.now, round_count, label))
        self.engine.schedule(self.latency_ticks, on_done)


def test_the_infinite_factory_delivers_at_once():
    engine = Engine(verbose=False)
    factory = InfiniteFactory(engine)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [0]


def test_a_state_arrives_after_the_attempt_and_the_return_trip():
    engine = Engine(verbose=False)
    factory = DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        return_ticks=30,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [130]
    assert factory.total_stall_ticks == 130
    assert factory.produced == 1


def test_a_warm_store_delivers_without_a_stall():
    engine = Engine(verbose=False)
    factory = DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        initial_store=1,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [0]
    assert factory.total_stall_ticks == 0


def test_eleven_correction_decodes_run_in_parallel_before_delivery():
    engine = Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=40)
    factory = DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=decoder,
        correction_round_count=3,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert len(decoder.submitted) == 11
    assert decoder.submitted[0] == (100, 3, "MSF-corr")
    assert delivered == [140]
    snapshot = factory.latency_aggregate_snapshot()
    assert snapshot["distill"] == {"sum": 100, "max": 100, "n": 1}
    assert snapshot["corr_decode"] == {"sum": 40, "max": 40, "n": 1}
    assert snapshot["total"] == {"sum": 140, "max": 140, "n": 1}


def test_a_cancelled_request_is_never_delivered():
    engine = Engine(verbose=False)
    factory = DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
    )
    delivered = []
    ticket = factory.request(1, lambda: delivered.append(engine.now))
    assert ticket.cancel() is True
    assert ticket.cancel() is False
    engine.run()
    assert delivered == []
    assert factory.store == 1


def test_one_final_state_costs_fifteen_prepared_states_and_one_round():
    engine = Engine(verbose=False)
    level = DistillLevel(unit_count=1, distance=3, logical_cycles_per_round=13)
    factory = MultiLevelDistillationFactory(
        engine,
        [level],
        round_ticks=10,
        preparation_unit_count=15,
        preparation_logical_cycles=1,
        preparation_distance=1,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert factory.produced_by_level == {0: 15, 1: 1}
    assert delivered == [400]
    assert factory.total_stall_ticks == 400


def test_a_second_request_waits_for_a_second_round():
    engine = Engine(verbose=False)
    level = DistillLevel(unit_count=1, distance=3, logical_cycles_per_round=13)
    factory = MultiLevelDistillationFactory(
        engine,
        [level],
        round_ticks=10,
        preparation_unit_count=15,
        preparation_logical_cycles=1,
        preparation_distance=1,
    )
    delivered = []

    def note_first():
        delivered.append((1, engine.now))

    def note_second():
        delivered.append((2, engine.now))

    factory.request(1, note_first)
    factory.request(2, note_second)
    engine.run()
    assert delivered == [(1, 400), (2, 790)]
