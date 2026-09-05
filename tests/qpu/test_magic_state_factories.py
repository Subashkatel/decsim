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

    def enqueue_without_input(self, round_count, on_done, label):
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
    assert decoder.submitted == [(100, 3, "MSF-corr")] * 11
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


def eight_requests(factory, note):
    factory.request(1, note)
    factory.request(2, note)
    factory.request(3, note)
    factory.request(4, note)
    factory.request(5, note)
    factory.request(6, note)
    factory.request(7, note)
    factory.request(8, note)


def test_a_fixed_seed_gives_the_same_eight_delivery_ticks_in_another_engine():
    first_engine = decsim.engine.Engine(verbose=False)
    first = single_stage(first_engine, success_probability=0.5, seed=2)
    first_delivered = []
    eight_requests(first, lambda: first_delivered.append(first_engine.now))
    first_engine.run()
    second_engine = decsim.engine.Engine(verbose=False)
    second = single_stage(second_engine, success_probability=0.5, seed=2)
    second_delivered = []
    eight_requests(second, lambda: second_delivered.append(second_engine.now))
    second_engine.run()
    assert first_delivered == [300, 400, 800, 1200, 1300, 1400, 1900, 2000]
    assert second_delivered == [300, 400, 800, 1200, 1300, 1400, 1900, 2000]


def test_another_seed_gives_another_delivery_sequence():
    engine = decsim.engine.Engine(verbose=False)
    factory = single_stage(engine, success_probability=0.5, seed=3)
    delivered = []
    eight_requests(factory, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [100, 300, 600, 700, 900, 1000, 1200, 1400]


def test_the_latency_maximum_is_kept_over_several_deliveries():
    # The first state waits 50 ticks in the buffer before a request takes
    # it (total 150); the refill is taken as soon as it is ready (total
    # 100), so the maximum is the earlier delivery's.
    engine = decsim.engine.Engine(verbose=False)
    factory = single_stage(
        engine, production_mode="continuous", buffer_capacity=1
    )
    delivered = []

    def two_requests():
        factory.request(1, lambda: delivered.append(engine.now))
        factory.request(2, lambda: delivered.append(engine.now))

    engine.schedule(150, two_requests)
    engine.run()
    assert delivered == [150, 250]
    snapshot = factory.latency_aggregate_snapshot()
    assert snapshot["total"] == {"sum": 250, "max": 150, "n": 2}
    assert snapshot["deliver"] == {"sum": 50, "max": 50, "n": 2}


def test_the_single_stage_log_names_the_request_the_ready_state_and_delivery():
    engine = decsim.engine.Engine(verbose=False)
    factory = magic_state_factories.DistillationFactory(
        engine,
        unit_count=1,
        attempt_ticks=2_000_000,
        decode_service=None,
        correction_round_count=0,
        correction_decode_count=0,
        return_ticks=500_000,
    )
    factory.request(1, lambda: None)
    engine.run()
    assert engine.log_lines == [
        "[  0.000 us] Factory: op#1 requests a magic state "
        "(store 0, waiting 1)",
        "[  2.500 us] Factory: magic state ready (store now 1)",
        "[  2.500 us] Factory:   -> delivered to op#1 (store now 0)"
        "  (supply stall 2.500 us)",
    ]


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


def test_an_unknown_production_mode_is_refused():
    engine = decsim.engine.Engine(verbose=False)
    with pytest.raises(ValueError, match="production_mode must be"):
        single_stage(engine, production_mode="batch")


def test_correction_decodes_need_a_decode_service():
    engine = decsim.engine.Engine(verbose=False)
    with pytest.raises(ValueError, match="decode_service is required"):
        magic_state_factories.DistillationFactory(
            engine, 1, 1, None, 0, correction_decode_count=11
        )


def test_a_decode_service_without_correction_decodes_is_refused():
    engine = decsim.engine.Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=1)
    with pytest.raises(ValueError, match="decode_service must be None"):
        magic_state_factories.DistillationFactory(
            engine, 1, 1, decoder, 0, correction_decode_count=0
        )


def test_a_negative_correction_decode_count_is_refused():
    engine = decsim.engine.Engine(verbose=False)
    decoder = DecodeLog(engine, latency_ticks=1)
    with pytest.raises(ValueError, match="must be nonnegative"):
        magic_state_factories.DistillationFactory(
            engine, 1, 1, decoder, 0, correction_decode_count=-1
        )


def test_a_factory_without_a_unit_is_refused():
    engine = decsim.engine.Engine(verbose=False)
    with pytest.raises(ValueError, match="unit_count must be positive"):
        magic_state_factories.DistillationFactory(
            engine, 0, 1, None, 0, correction_decode_count=0
        )


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


def test_a_failed_preparation_is_counted_and_retried():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine, [level], preparation_success_probability=0.5, seed=1
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [430]
    assert factory.counters_by_level[0].failure_count == 11
    assert factory.counters_by_level[0].produced_count == 15


def test_a_preparation_success_probability_above_one_is_refused():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        chain(engine, [level], preparation_success_probability=2)


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


def test_the_chain_log_names_a_failed_round_and_a_distilled_round():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, success_probability=0.5
    )
    factory = magic_state_factories.MultiLevelDistillationFactory(
        engine,
        [level],
        round_ticks=10_000,
        preparation_unit_count=15,
        preparation_logical_cycles=1,
        preparation_distance=1,
        seed=1,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [800_000]
    assert engine.log_lines == [
        "[  0.000 us] Factory: op#1 requests a magic state "
        "(top-level store 0, waiting 1)",
        "[  0.400 us] Factory: level 1 distillation failed "
        "(inputs discarded), retrying",
        "[  0.800 us] Factory: level 1 distilled a state "
        "(final state to core buffer; consumed 15 level-0 states)",
        "[  0.800 us] Factory:   -> delivered final state to op#1"
        "  (supply stall 0.800 us)",
    ]


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


def test_an_explicit_cycle_count_wins_over_the_default_at_a_higher_level():
    engine = decsim.engine.Engine(verbose=False)
    levels = [
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
        magic_state_factories.DistillLevel(
            unit_count=1, distance=3, logical_cycles_per_round=20
        ),
    ]
    factory = chain(engine, levels)
    assert factory.round_ticks_by_level == {1: 390, 2: 600}


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


def test_a_continuous_chain_fills_its_buffer_before_any_request():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine, [level], production_mode="continuous", buffer_capacity=1
    )
    engine.run()
    assert engine.now == 400
    assert factory.counters_by_level[1].stored_state_count == 1
    assert factory.counters_by_level[1].produced_count == 1


def test_a_continuous_chain_refills_the_state_a_delivery_takes():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine, [level], production_mode="continuous", buffer_capacity=1
    )
    engine.run()
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [400]
    assert factory.counters_by_level[1].stored_state_count == 0
    engine.run()
    assert engine.now == 800
    assert factory.counters_by_level[1].stored_state_count == 1
    assert factory.counters_by_level[1].produced_count == 2


def test_a_cancelled_chain_request_starts_no_round():
    engine = decsim.engine.Engine(verbose=False)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(engine, [level])
    delivered = []
    ticket = factory.request(1, lambda: delivered.append(engine.now))
    assert ticket.cancel() is True
    engine.run()
    assert delivered == []
    assert engine.now == 10
    assert factory.counters_by_level[0].stored_state_count == 15
    assert factory.counters_by_level[1].produced_count == 0


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
