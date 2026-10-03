"""The tick and the clock domains a yaml prices on.

One microsecond is a million ticks and every duration in the machine is
an integer count of them, so a duration stated in microseconds is
rounded once, here. The rounding is Python's own round(), which is
half-even (IEEE 754's roundTiesToEven, the default in the language
reference's built-in functions), so a duration exactly between two
ticks lands on the even one and a sweep of many such durations does not
drift upwards.

The clock section follows XQsim's shape: domain labels with frequencies
over one tick core. Both shipped domains start at LILLIPUT's 250 MHz
(2108.06569 Table 4), so a period is 4000 ticks. A domain hands out a
Clock, whose two methods are gem5's clockEdge and ticksToCycles
(gem5 src/sim/clocked_object.hh lines 174-186 and
224-227): a cost of n cycles is charged from the edge at or after the
current tick, and a span of ticks is rounded up to whole cycles.
"""

import dataclasses
import decimal
from typing import Optional

import pytest

import decsim.config as config


def test_a_duration_between_two_ticks_rounds_to_the_even_tick():
    half_a_tick = decimal.Decimal("0.0000005")
    one_and_a_half_ticks = decimal.Decimal("0.0000015")
    two_and_a_half_ticks = decimal.Decimal("0.0000025")
    assert config.microseconds_to_ticks(half_a_tick) == 0
    assert config.microseconds_to_ticks(one_and_a_half_ticks) == 2
    assert config.microseconds_to_ticks(two_and_a_half_ticks) == 2


def test_a_negative_duration_is_refused_by_name():
    with pytest.raises(ValueError, match="round_period"):
        config.check_duration("round_period", -1)


def test_a_positive_duration_that_rounds_to_no_ticks_is_refused():
    a_quarter_tick = decimal.Decimal("0.00000025")
    with pytest.raises(
        ValueError, match="round_period is positive but rounds to zero ticks"
    ):
        config.check_duration("round_period", a_quarter_tick)


def test_a_named_domains_clock_has_that_frequencys_period_in_ticks():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0, "room": 500})
    assert clocks.megahertz("fridge") == 250.0
    fridge = clocks.clock("fridge")
    room = clocks.clock("room")
    assert fridge.period_ticks == 4000
    assert room.period_ticks == 2000


def test_a_clock_from_megahertz_rounds_its_period_to_whole_ticks():
    """A third of a microsecond is 333333.33 ticks, rounded to 333333."""
    assert config.Clock.from_megahertz(250.0) == config.Clock(4000)
    assert config.Clock.from_megahertz(3) == config.Clock(333_333)


def test_a_clock_from_megahertz_is_the_clock_its_domain_hands_out():
    clocks = config.ClockSettings.from_yaml({"helios": 100.0})

    assert clocks.clock("helios") == config.Clock.from_megahertz(100.0)


def test_a_clock_from_no_positive_frequency_is_refused():
    with pytest.raises(ValueError):
        config.Clock.from_megahertz(0)


def test_a_clock_faster_than_one_tick_is_refused():
    with pytest.raises(ValueError, match="has a period below one tick"):
        config.Clock.from_megahertz(3e6)


@dataclasses.dataclass(frozen=True)
class _PartSettings:
    """A part's settings, which may name its own clock."""

    clock: Optional[config.Clock] = None


def test_settings_that_name_no_clock_run_on_the_machines():
    """A part with no clock takes the machine's; one with a clock keeps it.

    The two periods differ, so a fallback that handed every part the
    machine's clock, or none, fails here; the lock's fridge and room are
    both 250 MHz and cannot tell them apart.
    """
    machine_clock = config.Clock(4000)
    own_clock = config.Clock(2000)
    unclocked = _PartSettings()
    clocked = _PartSettings(clock=own_clock)

    on_the_machine = config.with_machine_clock(unclocked, machine_clock)
    on_its_own = config.with_machine_clock(clocked, machine_clock)

    assert on_the_machine.clock == machine_clock
    assert on_its_own.clock == own_clock


def test_a_cost_charged_mid_cycle_runs_from_the_next_edge():
    clock = config.Clock(4000)
    assert clock.edge(3, 1) == 16000
    assert clock.edge(3, 3999) == 16000
    assert clock.edge(0, 1) == 4000


def test_a_span_of_ticks_is_rounded_up_to_whole_cycles():
    clock = config.Clock(4000)
    assert clock.cycles_for(0) == 0
    assert clock.cycles_for(1) == 1
    assert clock.cycles_for(4000) == 1
    assert clock.cycles_for(4001) == 2


@pytest.mark.parametrize(
    "megahertz",
    [0, -250.0, float("inf"), float("nan"), True, "250", "x", None],
    ids=[
        "zero",
        "negative",
        "infinity",
        "nan",
        "true",
        "quoted_number",
        "word",
        "null",
    ],
)
def test_a_clock_that_is_not_a_positive_frequency_is_refused(megahertz):
    with pytest.raises(
        ValueError,
        match="clock fridge must be a positive frequency in megahertz, got",
    ):
        config.ClockSettings.from_yaml({"fridge": megahertz})


def test_a_cycle_count_requires_an_integer_by_name():
    with pytest.raises(ValueError, match="packing_cycles"):
        config.check_cycles("packing_cycles", 0.5)


def test_a_negative_cycle_count_is_refused_by_name():
    with pytest.raises(ValueError, match="packing_cycles"):
        config.check_cycles("packing_cycles", -1)


def test_a_whole_count_is_read_or_defaulted():
    section = {"window_rounds": 4}

    written = config.whole_count(section, "detector", "window_rounds", 9, "")
    silent = config.whole_count(section, "detector", "hold_rounds", 9, "")

    assert written == 4
    assert silent == 9


@pytest.mark.parametrize("value", [0, True, 2.0, "3"])
def test_a_whole_count_is_an_integer_of_at_least_one(value):
    section = {"window_rounds": value}

    with pytest.raises(ValueError) as refusal:
        config.whole_count(section, "detector", "window_rounds", 4, "rounds")

    assert str(refusal.value) == (
        "detector.window_rounds must be a whole number of rounds, at least "
        f"1 (got {value!r})"
    )


def test_a_whole_count_may_start_at_its_minimum():
    section = {"deadline_rounds": 0, "late_rounds": -1}

    read = config.whole_count(section, "detector", "deadline_rounds", 9, "", 0)
    with pytest.raises(ValueError, match="at least 0 .got -1."):
        config.whole_count(section, "detector", "late_rounds", 9, "", 0)

    assert read == 0


def test_a_knob_is_read_or_off():
    section = {"trace": True}

    assert config.boolean(section, "observation", "trace") is True
    assert config.boolean(section, "observation", "log") is False


def test_a_knob_is_true_or_false_and_never_a_count():
    section = {"trace": 1}

    with pytest.raises(ValueError) as refusal:
        config.boolean(section, "observation", "trace")

    assert str(refusal.value) == (
        "observation.trace must be true or false, got 1"
    )


@pytest.mark.parametrize(
    ("value", "is_number"),
    [(3, True), (0.5, True), (True, False), ("3", False)],
)
def test_a_number_is_an_int_or_a_float_and_not_a_bool(value, is_number):
    assert config.is_number(value) is is_number
