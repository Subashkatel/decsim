"""The pool's offer: the unit with the rounds, the least loaded, least work.

A job that may not start yet is staged where the least work is left,
and a free unit a staged job waits on is not an empty one: the two
forced-class solves of one window are two ordinary jobs, so the unit
count alone decides whether they overlap.

The law: with known deterministic work, dispatching each job to the
server with the least work left starts it at the tick a central FIFO
queue over the pool would (test_decoder_manager.py pins it through the
manager). The textbook treatment is Harchol-Balter, Performance
Modeling and Design of Computer Systems, Cambridge 2013.
"""

import types

import pytest

import decsim.config as config
import decsim.decoders.decoder_pool as decoder_pool
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records

DECODE_TICKS = config.microseconds_to_ticks(1.0)


def _settings(unit_count):
    return decoder_pool.PoolSettings(name="default", unit_count=unit_count)


def _pool(unit_count):
    decoder = decoders.PresetLatencyDecoder(1.0)
    manager = types.SimpleNamespace(decoder=decoder)
    settings = _settings(unit_count)
    return decoder_pool.DecoderPool(manager, settings)


def _job(label):
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=1, label=label
    )


def _blocked_job(label):
    """A job whose window still owes a boundary, so it may not start."""
    window = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=1,
        buffer_hi=1,
        round_count=1,
        deps_remaining=1,
    )
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=1, label=label, window=window
    )


class _MeasuredDecoder:
    """A decoder that declares no cost before its decode runs."""

    def occupancy(self, job):
        del job
        return None


def _no_demand(_job):
    return 0


def _input_nowhere(_job, _unit):
    return False


def _offer(pool, job, carries_input, input_is_on_the_unit=_input_nowhere):
    return pool.offer(
        job,
        now=0,
        carries_input=carries_input,
        resident_capacity=2,
        memory_demand_of=_no_demand,
        input_is_on_the_unit=input_is_on_the_unit,
    )


def test_a_free_unit_with_room_is_offered_with_its_compute():
    pool = _pool(2)
    first, second = pool.units
    busy = _job("busy")
    pool.claim(first, busy)
    next_job = _job("next")
    unit, has_free_compute = _offer(pool, next_job, carries_input=True)
    assert unit is second
    assert has_free_compute is True


def test_the_free_unit_the_fewest_residents_await_is_offered():
    pool = _pool(2)
    first, second = pool.units
    waiting = _job("waiting")
    first.admit(waiting)
    next_job = _job("next")
    unit, has_free_compute = _offer(pool, next_job, carries_input=True)
    assert unit is second
    assert has_free_compute is True


def test_a_free_unit_that_has_the_jobs_rounds_is_offered_before_an_empty_one():
    pool = _pool(2)
    _first, second = pool.units
    waiting = _job("waiting")
    second.admit(waiting)
    next_job = _job("next")

    def input_is_on_the_second_unit(_job, unit):
        return unit is second

    unit, has_free_compute = _offer(
        pool,
        next_job,
        carries_input=True,
        input_is_on_the_unit=input_is_on_the_second_unit,
    )
    assert unit is second
    assert has_free_compute is True


def test_a_job_with_input_is_staged_on_the_busy_unit_that_frees_earliest():
    pool = _pool(2)
    first, second = pool.units
    long_job = _job("long")
    pool.claim(first, long_job)
    first.expect_compute_free(30)
    short_job = _job("short")
    pool.claim(second, short_job)
    second.expect_compute_free(20)
    next_job = _job("next")
    unit, has_free_compute = _offer(pool, next_job, carries_input=True)
    assert unit is second
    assert has_free_compute is False


def test_a_blocked_job_is_staged_on_the_unit_with_the_least_work_left():
    """A free unit a staged job waits on owes a whole decode."""
    pool = _pool(2)
    first, second = pool.units
    running = _job("running")
    running.service_started = True
    first.admit(running)
    pool.claim(first, running)
    half_a_decode_ticks = DECODE_TICKS // 2
    first.expect_compute_free(half_a_decode_ticks)
    staged = _job("staged")
    second.admit(staged)
    blocked = _blocked_job("blocked")
    unit, has_free_compute = _offer(pool, blocked, carries_input=True)
    assert unit is first
    assert has_free_compute is False


def test_a_holder_past_its_predicted_free_tick_is_unbounded_work():
    """A result not yet read holds the unit for a time nobody declared."""
    pool = _pool(2)
    first, second = pool.units
    finished = _job("finished")
    finished.service_started = True
    first.admit(finished)
    pool.claim(first, finished)
    first.expect_compute_free(DECODE_TICKS)
    staged = _job("staged")
    second.admit(staged)
    blocked = _blocked_job("blocked")
    long_after_ticks = 5 * DECODE_TICKS
    unit, has_free_compute = pool.offer(
        blocked,
        now=long_after_ticks,
        carries_input=True,
        resident_capacity=2,
        memory_demand_of=_no_demand,
        input_is_on_the_unit=_input_nowhere,
    )
    assert unit is second
    assert has_free_compute is True


def test_a_blocked_job_leaves_the_unit_its_companion_is_staged_on():
    """The rounds being there never outweigh a decode to wait behind."""
    pool = _pool(2)
    first, second = pool.units
    companion = _blocked_job("companion")
    first.admit(companion)
    blocked = _blocked_job("blocked")

    def input_is_on_the_first_unit(_job, unit):
        return unit is first

    unit, has_free_compute = _offer(
        pool,
        blocked,
        carries_input=True,
        input_is_on_the_unit=input_is_on_the_first_unit,
    )
    assert unit is second
    assert has_free_compute is True


def test_a_blocked_job_takes_the_rounds_in_place_when_the_work_is_equal():
    pool = _pool(2)
    _first, second = pool.units
    blocked = _blocked_job("blocked")

    def input_is_on_the_second_unit(_job, unit):
        return unit is second

    unit, _has_free_compute = _offer(
        pool,
        blocked,
        carries_input=True,
        input_is_on_the_unit=input_is_on_the_second_unit,
    )
    assert unit is second


def test_where_no_cost_is_declared_the_unit_with_the_fewest_jobs_is_taken():
    decoder = _MeasuredDecoder()
    manager = types.SimpleNamespace(decoder=decoder)
    settings = _settings(2)
    pool = decoder_pool.DecoderPool(manager, settings)
    first, second = pool.units
    first_waiting = _job("first waiting")
    first.admit(first_waiting)
    second_waiting = _job("second waiting")
    first.admit(second_waiting)
    third_waiting = _job("third waiting")
    second.admit(third_waiting)
    blocked = _blocked_job("blocked")
    unit, _has_free_compute = pool.offer(
        blocked,
        now=0,
        carries_input=True,
        resident_capacity=3,
        memory_demand_of=_no_demand,
        input_is_on_the_unit=_input_nowhere,
    )
    assert unit is second


def test_a_job_without_input_waits_when_no_unit_is_free():
    pool = _pool(1)
    (unit,) = pool.units
    busy = _job("busy")
    pool.claim(unit, busy)
    next_job = _job("next")
    assert _offer(pool, next_job, carries_input=False) is None


def test_a_released_unit_is_offered_after_the_ones_freed_before_it():
    pool = _pool(2)
    first, second = pool.units
    job_a = _job("a")
    pool.claim(first, job_a)
    job_b = _job("b")
    pool.claim(second, job_b)
    pool.release(second)
    pool.release(first)
    next_job = _job("next")
    unit, _has_free_compute = _offer(pool, next_job, carries_input=True)
    assert unit is second


def test_a_tier_with_no_unit_is_refused():
    """A pool of no units would hold every job forever.

    The tier's record is where the count enters; the build makes every
    pool from it, or one unit when the run has no decoder.
    """
    algorithm = decoders.PresetLatencyDecoder.Settings(1.0)

    with pytest.raises(ValueError):
        decoder_settings.DecoderPoolSettings(algorithm=algorithm, unit_count=0)
