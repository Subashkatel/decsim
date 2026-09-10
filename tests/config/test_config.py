"""The tick, the microsecond stamp, and the clock domains a yaml prices on.

One microsecond is a million ticks and every duration in the machine is
an integer count of them, so a duration stated in microseconds is
rounded once, here. The rounding is Python's own round(), which is
half-even (IEEE 754's roundTiesToEven, the default in the language
reference's built-in functions), so a duration exactly between two
ticks lands on the even one and a sweep of many such durations does not
drift upwards.

The clock section follows XQsim's shape: domain labels with frequencies
over one tick core. Both shipped domains start at LILLIPUT's 250 MHz
(2108.06569 Table 4).
"""

import decimal

import pytest

import decsim.config as config


def test_a_microsecond_is_a_million_ticks():
    assert config.microseconds_to_ticks(1.25) == 1_250_000
    assert config.microseconds_to_ticks(0) == 0


def test_a_duration_between_two_ticks_rounds_to_the_even_tick():
    half_a_tick = decimal.Decimal("0.0000005")
    one_and_a_half_ticks = decimal.Decimal("0.0000015")
    two_and_a_half_ticks = decimal.Decimal("0.0000025")
    assert config.microseconds_to_ticks(half_a_tick) == 0
    assert config.microseconds_to_ticks(one_and_a_half_ticks) == 2
    assert config.microseconds_to_ticks(two_and_a_half_ticks) == 2


def test_the_stamp_is_seven_columns_of_microseconds_with_three_decimals():
    assert config.format_ticks(0) == "  0.000 us"
    assert config.format_ticks(1_234_567) == "  1.235 us"
    assert config.format_ticks(-1_000_000) == " -1.000 us"


def test_a_negative_duration_is_refused_by_name():
    with pytest.raises(
        ValueError, match="round_period must be a finite nonnegative number"
    ):
        config.check_duration("round_period", -1)


def test_a_duration_that_is_not_a_number_is_refused_by_name():
    with pytest.raises(
        ValueError, match="round_period must be a finite nonnegative number"
    ):
        config.check_duration("round_period", float("nan"))


def test_an_infinite_duration_is_refused_by_name():
    with pytest.raises(
        ValueError, match="round_period must be a finite nonnegative number"
    ):
        config.check_duration("round_period", float("inf"))


def test_a_positive_duration_that_rounds_to_no_ticks_is_refused():
    a_quarter_tick = decimal.Decimal("0.00000025")
    with pytest.raises(
        ValueError, match="round_period is positive but rounds to zero ticks"
    ):
        config.check_duration("round_period", a_quarter_tick)


def test_cycles_of_a_named_domain_are_microseconds_at_its_frequency():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0, "room": 500})
    assert clocks.megahertz("fridge") == 250.0
    assert clocks.microseconds(25, "fridge") == 0.1
    assert clocks.microseconds(8, "room") == 0.016


def test_a_clock_that_is_not_a_positive_frequency_is_refused():
    with pytest.raises(
        ValueError,
        match="clock fridge must be a positive frequency in megahertz, got 0",
    ):
        config.ClockSettings.from_yaml({"fridge": 0})
