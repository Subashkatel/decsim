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
import decsim.ports as ports
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.trace_source as trace_source


class DecodeLog:
    """A decode service that finishes every job after a fixed latency."""

    def __init__(self, engine, latency_ticks):
        self.engine = engine
        self.latency_ticks = latency_ticks
        self.submitted = []

    def enqueue_without_input(self, round_count, on_done, label):
        self.submitted.append((self.engine.now, round_count, label))
        self.engine.schedule(self.latency_ticks, on_done)


def infinite(engine):
    """The idealized row, built and started the way the root does."""
    factory = magic_state_factories.InfiniteFactory(engine)
    factory.start()
    return factory


def distillation(engine, *, decode_queue=None, **arguments):
    """One 15-to-1 row, built, bound and started the way the root does."""
    factory = unstarted_distillation(
        engine, decode_queue=decode_queue, **arguments
    )
    factory.start()
    return factory


def unstarted_distillation(engine, *, decode_queue=None, **arguments):
    """The same row, built and bound and not started."""
    row = magic_state_factories.DistillationFactory
    settings = row.Settings(**arguments)
    factory = settings.build(engine, 0)
    if decode_queue is not None:
        factory.decode_queue = decode_queue
    return factory


def multi_level(engine, *, round_ticks, decode_queue=None, **arguments):
    """One level chain, built, bound and started the way the root does."""
    row = magic_state_factories.MultiLevelDistillationFactory
    settings = row.Settings(**arguments)
    factory = settings.build(engine, round_ticks)
    if decode_queue is not None:
        factory.decode_queue = decode_queue
    factory.start()
    return factory


def single_stage(engine, **settings):
    """The plainest 15-to-1 row: one unit, no correction decodes."""
    return distillation(
        engine,
        unit_count=1,
        attempt_ticks=100,
        correction_round_count=0,
        correction_decode_count=0,
        **settings,
    )


def chain(engine, levels, **settings):
    """The level chain the paper's numbers are read on."""
    return multi_level(
        engine,
        round_ticks=10,
        levels=levels,
        preparation_unit_count=15,
        preparation_logical_cycles=1,
        preparation_distance=1,
        **settings,
    )


def test_every_factory_row_fills_the_port_it_is_built_behind():
    """The rows are in qpu and the port is in decsim/ports.py.

    The runtime that asks for a state never names a row, so the promise
    is only real if every row answers the whole port.
    """
    engine = decsim.engine.Engine()
    rows = [infinite(engine), single_stage(engine)]
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, logical_cycles_per_round=13
    )
    level_chain = chain(engine, [level])
    rows.append(level_chain)
    fills_the_port = [isinstance(row, ports.MagicStateFactory) for row in rows]
    assert fills_the_port == [True, True, True]


class _StartRequestShutdown:
    """A factory row that keeps no engine: three calls and its trace."""

    def __init__(self):
        self.trace = _SilentFactoryTrace()

    def start(self):
        """Nothing ahead of a request."""

    def request(self, operation_id, callback):
        """Deliver at once."""
        del operation_id
        callback()

    def shutdown(self):
        """Nothing runs."""


class _SilentFactoryTrace:
    """The one source the port declares, for a row that never waits."""

    state_delivered = trace_source.SilentSource()


def test_the_port_asks_a_row_for_its_three_calls_and_no_engine():
    """No caller reads a factory's engine, so the port does not ask for one."""
    row = _StartRequestShutdown()
    assert isinstance(row, ports.MagicStateFactory)


def test_the_infinite_factory_delivers_at_once():
    engine = decsim.engine.Engine()
    factory = infinite(engine)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [0]


def test_a_warm_store_delivers_without_a_stall():
    engine = decsim.engine.Engine()
    factory = single_stage(engine, initial_store=1)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [0]
    assert factory.state.total_stall_ticks == 0
    assert factory.state.stored_state_count == 0


def test_eleven_correction_decodes_run_in_parallel_before_delivery():
    engine = decsim.engine.Engine()
    decoder = DecodeLog(engine, latency_ticks=40)
    factory = distillation(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_queue=decoder,
        correction_round_count=3,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert decoder.submitted == [(100, 3, "MSF-corr")] * 11
    assert delivered == [140]
    assert factory.state.produced_count == 1
    assert factory.state.in_flight_count == 0
    assert factory.state.stored_state_count == 0


def test_the_return_trip_follows_the_correction_decodes():
    engine = decsim.engine.Engine()
    decoder = DecodeLog(engine, latency_ticks=40)
    factory = distillation(
        engine,
        unit_count=1,
        attempt_ticks=100,
        decode_queue=decoder,
        correction_round_count=3,
        return_ticks=30,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [170]


def test_two_units_hold_two_states_in_flight_at_the_peak():
    engine = decsim.engine.Engine()
    factory = distillation(
        engine,
        unit_count=2,
        attempt_ticks=100,
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
    assert factory.state.peak_in_flight_count == 2
    assert factory.state.in_flight_count == 0


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
    first_engine = decsim.engine.Engine()
    first = single_stage(first_engine, success_probability=0.5, seed=2)
    first_delivered = []
    eight_requests(first, lambda: first_delivered.append(first_engine.now))
    first_engine.run()
    second_engine = decsim.engine.Engine()
    second = single_stage(second_engine, success_probability=0.5, seed=2)
    second_delivered = []
    eight_requests(second, lambda: second_delivered.append(second_engine.now))
    second_engine.run()
    assert first_delivered == [300, 400, 800, 1200, 1300, 1400, 1900, 2000]
    assert second_delivered == [300, 400, 800, 1200, 1300, 1400, 1900, 2000]


def test_continuous_production_keeps_the_buffer_full_ahead_of_demand():
    engine = decsim.engine.Engine()
    factory = distillation(
        engine,
        unit_count=2,
        attempt_ticks=100,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=2,
    )
    engine.run()
    assert engine.now == 100
    assert factory.state.stored_state_count == 2
    assert factory.state.produced_count == 2
    assert factory.state.busy_unit_count == 0


def test_continuous_production_refills_the_slot_a_delivery_takes():
    engine = decsim.engine.Engine()
    factory = distillation(
        engine,
        unit_count=2,
        attempt_ticks=100,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=2,
    )
    engine.run()
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [100]
    assert factory.state.stored_state_count == 1
    assert factory.state.busy_unit_count == 1
    engine.run()
    assert engine.now == 200
    assert factory.state.stored_state_count == 2
    assert factory.state.produced_count == 3


def test_a_shut_down_factory_launches_no_attempt():
    engine = decsim.engine.Engine()
    factory = distillation(
        engine,
        unit_count=2,
        attempt_ticks=100,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=2,
    )
    factory.shutdown()
    engine.run()
    assert engine.now == 0
    assert factory.state.produced_count == 0


def test_an_unknown_production_mode_is_refused():
    engine = decsim.engine.Engine()
    with pytest.raises(ValueError, match="production_mode"):
        single_stage(engine, production_mode="batch")


def test_a_card_with_no_correction_decode_ignores_the_decode_queue():
    """The root binds every row's decode queue; this card sends it none."""
    engine = decsim.engine.Engine()
    decoder = DecodeLog(engine, latency_ticks=1)
    factory = distillation(
        engine,
        decode_queue=decoder,
        unit_count=1,
        attempt_ticks=100,
        correction_round_count=0,
        correction_decode_count=0,
        return_ticks=30,
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [130]
    assert decoder.submitted == []


def test_a_negative_correction_decode_count_is_refused():
    engine = decsim.engine.Engine()
    decoder = DecodeLog(engine, latency_ticks=1)
    with pytest.raises(ValueError, match="correction_decode_count"):
        distillation(
            engine,
            decode_queue=decoder,
            unit_count=1,
            attempt_ticks=1,
            correction_round_count=0,
            correction_decode_count=-1,
        )


def test_a_factory_without_a_unit_is_refused():
    engine = decsim.engine.Engine()
    with pytest.raises(ValueError, match="unit_count must be positive"):
        distillation(
            engine,
            unit_count=0,
            attempt_ticks=1,
            correction_round_count=0,
            correction_decode_count=0,
        )


def test_one_final_state_costs_fifteen_prepared_states_and_one_round():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, logical_cycles_per_round=13
    )
    factory = chain(engine, [level])
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert factory.state.counters_by_level[0].produced_count == 15
    assert factory.state.counters_by_level[1].produced_count == 1
    assert factory.state.counters_by_level[0].stored_state_count == 0
    assert factory.state.counters_by_level[1].stored_state_count == 0
    assert delivered == [400]
    assert factory.state.total_stall_ticks == 400


def test_a_second_request_waits_for_a_second_round():
    engine = decsim.engine.Engine()
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
    assert factory.state.total_stall_ticks == 400 + 790


def test_a_failed_preparation_is_counted_and_retried():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine, [level], preparation_success_probability=0.5, seed=1
    )
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [430]
    assert factory.state.counters_by_level[0].failure_count == 11
    assert factory.state.counters_by_level[0].produced_count == 15


def test_a_preparation_success_probability_above_one_is_refused():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    with pytest.raises(ValueError, match="preparation_success_probability"):
        chain(engine, [level], preparation_success_probability=2)


def test_a_failed_round_discards_its_inputs_and_is_counted():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(
        unit_count=1, distance=3, success_probability=0.5
    )
    factory = chain(engine, [level], seed=1)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert delivered == [800]
    assert factory.state.counters_by_level[0].produced_count == 30
    assert factory.state.counters_by_level[1].failure_count == 1
    assert factory.state.counters_by_level[1].produced_count == 1
    assert factory.state.counters_by_level[0].stored_state_count == 0


def test_a_chains_correction_decodes_carry_their_level_label():
    engine = decsim.engine.Engine()
    decoder = DecodeLog(engine, latency_ticks=40)
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine,
        [level],
        decode_queue=decoder,
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
    engine = decsim.engine.Engine()
    levels = [
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
        magic_state_factories.DistillLevel(
            unit_count=1, distance=3, logical_cycles_per_round=20
        ),
    ]
    factory = chain(engine, levels)
    assert factory.state.round_ticks_by_level == {1: 390, 2: 600}


def test_a_second_level_round_takes_fifteen_logical_cycles_by_default():
    # Silva 2411.04270, text lines 223-224: a first-level unit distills
    # every 13 logical cycles, a higher-level unit only every 15, because
    # loading its inputs takes two more. With round_ticks=10 and d=3 a
    # first-level round is 390 ticks and a second-level round 450; the
    # fifteen first-level rounds end at 10 + 15 * 390 = 5860.
    engine = decsim.engine.Engine()
    levels = [
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
    ]
    factory = chain(engine, levels)
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert factory.state.round_ticks_by_level == {1: 390, 2: 450}
    assert delivered == [6310]


def test_a_two_level_chain_feeds_fifteen_first_level_states_to_the_top():
    engine = decsim.engine.Engine()
    levels = [
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
        magic_state_factories.DistillLevel(unit_count=1, distance=3),
    ]
    factory = chain(engine, levels)
    factory.request(1, lambda: None)
    engine.run()
    assert factory.state.counters_by_level[0].produced_count == 225
    assert factory.state.counters_by_level[1].produced_count == 15
    assert factory.state.counters_by_level[2].produced_count == 1
    assert factory.state.counters_by_level[1].stored_state_count == 0
    assert factory.state.counters_by_level[2].stored_state_count == 0
    assert factory.state.peak_in_flight_count == 16


def test_a_continuous_chain_fills_its_buffer_before_any_request():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine, [level], production_mode="continuous", buffer_capacity=1
    )
    engine.run()
    assert engine.now == 400
    assert factory.state.counters_by_level[1].stored_state_count == 1
    assert factory.state.counters_by_level[1].produced_count == 1


def test_a_continuous_chain_refills_the_state_a_delivery_takes():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(
        engine, [level], production_mode="continuous", buffer_capacity=1
    )
    engine.run()
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    assert delivered == [400]
    assert factory.state.counters_by_level[1].stored_state_count == 0
    engine.run()
    assert engine.now == 800
    assert factory.state.counters_by_level[1].stored_state_count == 1
    assert factory.state.counters_by_level[1].produced_count == 2


def test_a_shut_down_chain_serves_no_request():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    factory = chain(engine, [level])
    factory.shutdown()
    delivered = []
    factory.request(1, lambda: delivered.append(engine.now))
    engine.run()
    assert engine.now == 0
    assert delivered == []
    assert factory.state.counters_by_level[0].produced_count == 0


def test_continuous_production_refuses_an_empty_buffer_capacity():
    engine = decsim.engine.Engine()
    with pytest.raises(ValueError, match="continuous production needs"):
        single_stage(engine, production_mode="continuous", buffer_capacity=0)


def test_a_continuous_chain_refuses_an_empty_buffer_capacity():
    engine = decsim.engine.Engine()
    level = magic_state_factories.DistillLevel(unit_count=1, distance=3)
    with pytest.raises(ValueError, match="continuous production needs"):
        chain(engine, [level], production_mode="continuous", buffer_capacity=0)


def test_a_continuous_row_queues_nothing_until_it_is_started():
    """The constructor wires the row; start queues its first attempt.

    gem5 splits the constructor from startup, "the appropriate place to
    schedule initial event(s)"
    (gem5 src/sim/sim_object.hh lines 194 and 280), so the
    order the root builds its components in cannot move a tick.
    """
    engine = decsim.engine.Engine()
    factory = unstarted_distillation(
        engine,
        unit_count=1,
        attempt_ticks=100,
        correction_round_count=0,
        correction_decode_count=0,
        production_mode="continuous",
        buffer_capacity=1,
    )
    assert engine.idle is True

    factory.start()

    assert engine.idle is False
    engine.run()
    assert factory.state.stored_state_count == 1


def test_factory_level_cycles_require_integers_by_key():
    with pytest.raises(ValueError, match="level.logical_cycles_per_round"):
        magic_state_factories.DistillLevel(
            unit_count=1, distance=3, logical_cycles_per_round=0.5
        )
