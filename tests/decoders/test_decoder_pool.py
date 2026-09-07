"""The pool's offer: a free unit with room first, else least work left.

Harchol-Balter, Performance Modeling and Design of Computer Systems
(2013), Ch. 24: with known deterministic work, dispatching each job to
the server with the least work left starts it when a central FIFO queue
would (rowD2, compare_overlap_laws.py, law_lwl_pool).
"""

import pytest

import decsim.decoders.decoder_pool as decoder_pool
import decsim.decoders.decoders as decoders
import decsim.records.decoding as decoding_records


def _pool(unit_count):
    decoder = decoders.PresetLatencyDecoder(1.0)
    router = decoders.CodeRouter(decoder)
    return decoder_pool.DecoderPool(router, {"default": unit_count})


def _job(label):
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, n_rounds=1, label=label
    )


def _no_demand(_job):
    return 0


def _offer(pool, job, carries_input):
    return pool.offer(
        "default",
        job,
        carries_input=carries_input,
        resident_capacity=2,
        memory_demand_of=_no_demand,
    )


def test_a_free_unit_with_room_is_offered_with_its_compute():
    pool = _pool(2)
    first, second = pool.units_by_pool["default"]
    busy = _job("busy")
    pool.claim(first, busy)
    next_job = _job("next")
    unit, has_free_compute = _offer(pool, next_job, carries_input=True)
    assert unit is second
    assert has_free_compute is True


def test_a_job_with_input_is_staged_on_the_busy_unit_that_frees_earliest():
    pool = _pool(2)
    first, second = pool.units_by_pool["default"]
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


def test_a_job_without_input_waits_when_no_unit_is_free():
    pool = _pool(1)
    (unit,) = pool.units_by_pool["default"]
    busy = _job("busy")
    pool.claim(unit, busy)
    next_job = _job("next")
    assert _offer(pool, next_job, carries_input=False) is None


def test_a_released_unit_is_offered_after_the_ones_freed_before_it():
    pool = _pool(2)
    first, second = pool.units_by_pool["default"]
    job_a = _job("a")
    pool.claim(first, job_a)
    job_b = _job("b")
    pool.claim(second, job_b)
    pool.release(second)
    pool.release(first)
    next_job = _job("next")
    unit, _has_free_compute = _offer(pool, next_job, carries_input=True)
    assert unit is second


def test_a_pool_map_without_a_default_pool_is_refused():
    with pytest.raises(ValueError, match='must include a "default" pool'):
        decoder_pool.DecoderPool(None, {"strong": 1})
