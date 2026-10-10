"""Values written with their units (parameters/units.py).

The referent is decsim's own conversion: a written time becomes the
ticks config.microseconds_to_ticks gives, cycles on a Clock the count
Clock.cycles_for gives, and a frequency the Clock Clock.from_megahertz
builds, so a value set by text runs exactly as the same value set in
Python.
"""

import pytest

import decsim.config as config
import decsim.parameters.units as units

TIMES = {"157ns": 0.157, "1.15us": 1.15, "4ns": 0.004, "3ms": 3000.0}


@pytest.mark.parametrize("written", sorted(TIMES))
def test_a_time_becomes_the_ticks_decsim_rounds_it_to(written):
    quantity = units.parse(written, units.TIME)
    microseconds = TIMES[written]

    ticks = quantity.ticks()

    assert ticks == config.microseconds_to_ticks(microseconds)


@pytest.mark.parametrize("written", ["4ns", "8ns", "10ns", "1.1us"])
def test_a_time_on_a_clock_is_the_clocks_own_cycle_count(written):
    clock = config.Clock.from_megahertz(250.0)
    quantity = units.parse(written, units.DURATION)
    ticks = quantity.ticks()

    cycles = quantity.cycles_on(clock)

    assert cycles == clock.cycles_for(ticks)


def test_a_count_of_cycles_is_whole_and_reads_no_clock():
    quantity = units.parse("3 cycles", units.DURATION)
    half = units.parse("1.5 cycles", units.DURATION)

    cycles = quantity.cycles_on(None)

    assert cycles == 3
    with pytest.raises(units.UnitError, match="whole"):
        half.cycles_on(None)


def test_a_frequency_becomes_the_clock_decsim_builds():
    quantity = units.parse("250MHz", units.FREQUENCY)

    clock = quantity.clock()

    assert clock == config.Clock.from_megahertz(250.0)


def test_a_number_without_its_unit_is_refused_with_an_example():
    with pytest.raises(units.UnitError, match='has no unit.*"157ns"'):
        units.parse(3, units.TIME, "weak_decoder.engine.fetch_per_round")


def test_a_value_of_another_dimension_is_refused():
    with pytest.raises(units.UnitError, match="is a frequency"):
        units.parse("250MHz", units.TIME)


def test_the_limit_words_are_values_of_their_own():
    unlimited = units.parse("unlimited", units.BITS)
    nothing = units.parse(" none ", units.BITS)

    assert unlimited == units.UNLIMITED
    assert nothing == units.NONE


def test_did_you_mean_names_the_near_names_and_nothing_else():
    near = units.did_you_mean("distnce", ["distance", "clock"])
    far = units.did_you_mean("zzz", ["distance"])

    assert near == " Did you mean distance?"
    assert far == ""
