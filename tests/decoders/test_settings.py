"""The decoder settings records of decsim/decoders/settings.py.

The pool, the engine and the unit memory each refuse a bad field by its
own name, and the manager holds its scheduler as a record that builds
it. linear_decoder_pool is Toshio et al.'s linear decode time,
T_dec(r) = tau_dec r (2510.25222 lines 968-971), charged as fetch
cycles a round on the pool's clock.
"""

import pytest

import decsim.config as config
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.records.decoding as decoding_records


def test_the_manager_holds_its_scheduler_as_a_record_that_builds_it():
    """A settings record holds data: the rule's record, not its class."""
    settings = decoder_settings.DecoderManagerSettings()
    record = settings.scheduler

    first = record.build()
    second = record.build()

    assert record.__dataclass_params__.frozen
    assert type(first) is schedulers.FifoScheduler
    assert first is not second


def test_an_engine_stage_of_negative_cycles_is_refused_by_its_record():
    with pytest.raises(ValueError, match="fetch_cycles_per_round"):
        decoder_settings.EngineSettings(fetch_cycles_per_round=-1)


def test_a_unit_memory_of_no_bits_is_refused_by_its_record():
    with pytest.raises(ValueError, match="unit_memory.bits"):
        decoder_settings.UnitMemorySettings(bits=0)


def test_a_unit_memory_word_of_no_whole_bits_is_refused_by_its_record():
    """A fraction of a bit would price every fetch at a fraction of a word."""
    sentence = "unit_memory.word_bits must be a whole number of bits"

    with pytest.raises(ValueError, match=sentence):
        decoder_settings.UnitMemorySettings(word_bits=2.5)


def test_a_word_width_on_a_pool_that_reads_in_place_is_refused():
    """The store prices that read; a width here would price it twice."""
    algorithm = matching.PyMatchingDecoder.Settings()
    memory = decoder_settings.UnitMemorySettings(word_bits=8)
    sentence = "reads the rounds where the store keeps them"

    with pytest.raises(ValueError, match=sentence):
        decoder_settings.DecoderPoolSettings(
            algorithm, unit_memory=memory, copies_input=False
        )


def test_a_result_blocking_value_that_is_not_a_boolean_is_refused_by_name():
    """A word is truthy, so "no" would block the unit; only a flag is read."""
    algorithm = matching.PyMatchingDecoder.Settings()
    sentence = "result_blocks_unit 'yes' is not a flag"

    with pytest.raises(ValueError, match=sentence):
        decoder_settings.DecoderPoolSettings(
            algorithm, result_blocks_unit="yes"
        )


def test_a_linear_pool_charges_tau_dec_for_every_round_of_the_job():
    """T_dec(r) = tau_dec r, 2510.25222 lines 968-971: 0.4 us x 10 rounds."""
    clock = config.Clock(period_ticks=4_000)
    pool = decoder_settings.linear_decoder_pool(0.4, clock, solves_per_window=1)
    engine = pool.engine
    fetch = staged_decoder.MemoryFetchStage(
        "fetch",
        cycles_per_job=engine.fetch_cycles_per_job,
        cycles_per_round=engine.fetch_cycles_per_round,
    )
    job = decoding_records.DecodeJob(
        operation_id=0, window_id=0, round_count=10
    )

    fetch_ticks = fetch.cycles_for(job) * clock.period_ticks
    assert fetch_ticks == config.microseconds_to_ticks(4.0)
    assert engine.release_cycles_per_job == 0
    assert engine.release_cycles_per_round == 0
    assert pool.algorithm.preset_latency_microseconds == 0.0


def test_a_linear_decode_time_off_the_clock_is_refused():
    """1 ns a round is a quarter of a 4 ns cycle."""
    clock = config.Clock(period_ticks=4_000)
    with pytest.raises(ValueError, match="is not a whole number of cycles"):
        decoder_settings.linear_decoder_pool(0.001, clock, solves_per_window=1)


def test_a_window_solved_twice_pays_half_of_the_time_on_each_solve():
    """One unit, one time a round: 0.4 us over two solves is 50 cycles each."""
    clock = config.Clock(period_ticks=4_000)
    pool = decoder_settings.linear_decoder_pool(0.4, clock, solves_per_window=2)

    assert pool.engine.fetch_cycles_per_round == 50
