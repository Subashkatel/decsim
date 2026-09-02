"""A factory delivers a magic state when one is ready, and no earlier.

Sources: Silva et al. 2411.04270 Sec. II B (a multi-level factory: level 0
prepares, each level consumes inputs_per_round lower states per round of
logical_cycles_per_round * distance QEC rounds, and a request pulls demand
down the chain; text lines 223-224 and 249-251: 13 logical cycles per
round at the first level, 15 above it); Litinski 1905.06903 Sec. 4, text
lines 1038-1043 (the 15-to-1 protocol's rotations 5 to 15 are the 11
multi-qubit pi/8 rotations that need a correction decode each; the first
four are single-qubit rotations with Pauli corrections). The supply stall
is the delay between a request and its delivery.
"""

import pytest

import decsim.engine
import decsim.qpu.magic_state_factories as magic_state_factories


class DecodeLog:
    """A decode service that finishes every job after a fixed latency."""

    def __init__(self, engine, latency_ticks):
        self.engine = engine
        self.latency_ticks = latency_ticks
        self.submitted = []

    def submit_decode(self, round_count, on_done, label):
        self.submitted.append((self.engine.now, round_count, label))
        self.engine.schedule(self.latency_ticks, on_done)


def single_stage(engine, **settings):
    return magic_state_factories.DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        **settings,
    )


def chain(engine, levels, **settings):
    return magic_state_factories.MultiLevelDistillationFactory(
        engine,
        levels,
        round_ticks=10,
        preparation_unit_count=15,
        preparation_logical_cycles=1,
        preparation_distance=1,
        **settings,
    )


def test_the_infinite_factory_delivers_at_once():
    engine = decsim.engine.Engine(verbose=False)
    factory = magic_state_factories.InfiniteFactory(engine)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [0]


def test_a_state_arrives_after_the_attempt_and_the_return_trip():
    engine = decsim.engine.Engine(verbose=False)
    factory = single_stage(engine, return_ticks=30)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [130]
    assert factory.total_stall_ticks == 130
    assert factory.produced_count == 1


def test_a_warm_store_delivers_without_a_stall():
    engine = decsim.engine.Engine(verbose=False)
    factory = single_stage(engine, initial_store=1)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [0]
    assert factory.total_stall_ticks == 0
    assert factory.stored_state_count == 0


def test_eleven_correction_decodes_run_in_parallel_before_delivery():
    engine = decsim.engine.Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=40)
    factory = magic_state_factories.DistillationFactory(
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
    assert decoder.submitted[10] == (100, 3, "MSF-corr")
    assert delivered == [140]
    assert factory.produced_count == 1
    assert factory.in_flight_count == 0
    assert factory.stored_state_count == 0
    snapshot = factory.latency_aggregate_snapshot()
    assert snapshot["distill"] == {"sum": 100, "max": 100, "n": 1}
    assert snapshot["corr_decode"] == {"sum": 40, "max": 40, "n": 1}
    assert snapshot["total"] == {"sum": 140, "max": 140, "n": 1}


def test_the_return_trip_after_the_decodes_is_the_deliver_stage():
    engine = decsim.engine.Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=40)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=decoder,
        correction_round_count=3,
        return_ticks=30,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [170]
    snapshot = factory.latency_aggregate_snapshot()
    assert snapshot["deliver"] == {"sum": 30, "max": 30, "n": 1}
    assert snapshot["total"] == {"sum": 170, "max": 170, "n": 1}


def test_a_delivered_state_carries_the_tick_of_every_stage():
    engine = decsim.engine.Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=40)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_service=decoder,
        correction_round_count=3,
        return_ticks=30,
    )
    factory.request(1, lambda: None)
    engine.run()
    assert list(factory.traces) == [
        magic_state_factories.StateTrace(
            state_id=0,
            distill_start_tick=0,
            physical_done_tick=100,
            correction_submit_tick=100,
            correction_done_tick=140,
            released_tick=170,
            delivered_tick=170,
        )
    ]


def test_two_units_hold_two_states_in_flight_at_the_peak():
    engine = decsim.engine.Engine(verbose=False)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=2,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        return_ticks=50,
    )
    delivered = []

    def note_first():
        delivered.append((1, engine.now))

    def note_second():
        delivered.append((2, engine.now))

    factory.request(1, note_first)
    factory.request(2, note_second)
    engine.run()
    assert delivered == [(1, 150), (2, 150)]
    assert factory.peak_in_flight_count == 2
    assert factory.in_flight_count == 0


def test_a_failed_attempt_is_discarded_and_retried():
    engine = decsim.engine.Engine(verbose=False)
    factory = single_stage(engine, success_probability=0.5, seed=2)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [300]
    assert factory.produced_count == 1
    assert engine.log_lines[1] == (
        "[  0.000 us] Factory: a unit's distillation DISCARDED, retrying"
    )
    assert engine.log_lines[2] == (
        "[  0.000 us] Factory: a unit's distillation DISCARDED, retrying"
    )


def test_a_fixed_seed_repeats_the_failure_sequence():
    first_engine = decsim.engine.Engine(verbose=False)
    first = single_stage(first_engine, success_probability=0.5, seed=2)
    first_delivered = []
    first.request(1, lambda: first_delivered.append(first_engine.now))
    first_engine.run()
    second_engine = decsim.engine.Engine(verbose=False)
    second = single_stage(second_engine, success_probability=0.5, seed=2)
    second_delivered = []
    second.request(1, lambda: second_delivered.append(second_engine.now))
    second_engine.run()
    assert first_delivered == [300]
    assert second_delivered == [300]


def test_continuous_production_keeps_the_buffer_full_ahead_of_demand():
    engine = decsim.engine.Engine(verbose=False)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=2,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=2,
    )
    engine.run()
    assert engine.now == 100
    assert factory.stored_state_count == 2
    assert factory.produced_count == 2
    assert factory.busy_unit_count == 0


def test_continuous_production_refills_the_slot_a_delivery_takes():
    engine = decsim.engine.Engine(verbose=False)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=2,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=2,
    )
    engine.run()
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [100]
    assert factory.stored_state_count == 1
    assert factory.busy_unit_count == 1
    engine.run()
    assert engine.now == 200
    assert factory.stored_state_count == 2
    assert factory.produced_count == 3


def test_a_shut_down_factory_launches_no_attempt():
    engine = decsim.engine.Engine(verbose=False)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=2,
        attempt_ticks=100,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=2,
    )
    factory.shutdown()
    engine.run()
    assert engine.now == 0
    assert factory.produced_count == 0


def test_a_cancelled_request_is_never_delivered():
    engine = decsim.engine.Engine(verbose=False)
    factory = single_stage(engine)
    delivered = []
    ticket = factory.request(1, lambda: delivered.append(engine.now))
    assert ticket.cancel() is True
    assert ticket.cancel() is False
    engine.run()
    assert delivered == []
    assert factory.stored_state_count == 1


def test_the_production_mode_is_checked_before_the_counts():
    engine = decsim.engine.Engine(verbose=False)
    with pytest.raises(ValueError, match="production_mode must be"):
        magic_state_factories.DistillationFactory(
            engine,
            1,
            1,
            None,
            0,
            correction_decode_count=0,
            production_mode="x",
            success_probability=2,
        )


def test_the_chains_production_mode_is_checked_before_the_levels():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=0, distance=3)
    with pytest.raises(ValueError, match="production_mode must be"):
        chain(engine, [level], production_mode="x")


def test_one_final_state_costs_fifteen_prepared_states_and_one_round():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, logical_cycles_per_round=13
    )
    factory = chain(engine, [level])
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert factory.counters_by_level[0].produced_count == 15
    assert factory.counters_by_level[1].produced_count == 1
    assert factory.counters_by_level[0].stored_state_count == 0
    assert factory.counters_by_level[1].stored_state_count == 0
    assert delivered == [400]
    assert factory.total_stall_ticks == 400


def test_a_second_request_waits_for_a_second_round():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, logical_cycles_per_round=13
    )
    factory = chain(engine, [level])
    delivered = []

    def note_first():
        delivered.append((1, engine.now))

    def note_second():
        delivered.append((2, engine.now))

    factory.request(1, note_first)
    factory.request(2, note_second)
    engine.run()
    assert delivered == [(1, 400), (2, 790)]


def test_two_queued_requests_stall_for_the_sum_of_their_waits():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, logical_cycles_per_round=13
    )
    factory = chain(engine, [level])
    factory.request(1, lambda: None)
    factory.request(2, lambda: None)
    engine.run()
    assert factory.total_stall_ticks == 1190


def test_a_failed_round_discards_its_inputs_and_is_counted():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, success_probability=0.5
    )
    factory = chain(engine, [level], seed=1)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [800]
    assert factory.counters_by_level[0].produced_count == 30
    assert factory.counters_by_level[1].failure_count == 1
    assert factory.counters_by_level[1].produced_count == 1
    assert factory.counters_by_level[0].stored_state_count == 0


def test_a_chains_correction_decodes_carry_their_level_label():
    engine = decsim.engine.Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=40)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine,
        [level],
        decode_service=decoder,
        correction_round_count=3,
        correction_decode_count=2,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert decoder.submitted == [
        (400, 3, "MSF-corr-L1"),
        (400, 3, "MSF-corr-L1"),
    ]
    assert delivered == [440]


def test_a_second_level_round_takes_fifteen_logical_cycles_by_default():
    # Silva 2411.04270, text lines 223-224: a first-level unit distills
    # every 13 logical cycles, a higher-level unit only every 15, because
    # loading its inputs takes two more. With round_ticks=10 and d=3 a
    # first-level round is 390 ticks and a second-level round 450; the
    # fifteen first-level rounds end at 10 + 15 * 390 = 5860.
    engine = decsim.engine.Engine(verbose=False)
    levels = [
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
    ]
    factory = chain(engine, levels)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert factory.round_ticks_by_level == {1: 390, 2: 450}
    assert delivered == [6310]


def test_a_two_level_chain_feeds_fifteen_first_level_states_to_the_top():
    engine = decsim.engine.Engine(verbose=False)
    levels = [
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
    ]
    factory = chain(engine, levels)
    factory.request(1, lambda: None)
    engine.run()
    assert factory.counters_by_level[0].produced_count == 225
    assert factory.counters_by_level[1].produced_count == 15
    assert factory.counters_by_level[2].produced_count == 1
    assert factory.counters_by_level[1].stored_state_count == 0
    assert factory.counters_by_level[2].stored_state_count == 0
    assert factory.peak_in_flight_count == 16


def test_a_shut_down_chain_serves_no_request():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(engine, [level])
    factory.shutdown()
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert engine.now == 0
    assert delivered == []
    assert factory.counters_by_level[0].produced_count == 0


def test_continuous_production_refuses_an_empty_buffer_capacity():
    engine = decsim.engine.Engine(verbose=False)
    with pytest.raises(ValueError, match="buffer_capacity >= 1"):
        single_stage(engine, production_mode="continuous", buffer_capacity=0)


def test_a_continuous_chain_refuses_an_empty_buffer_capacity():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    with pytest.raises(ValueError, match="buffer_capacity >= 1"):
        chain(engine, [level], production_mode="continuous", buffer_capacity=0)
