"""The tick, the microsecond stamp, and the clock domains a yaml prices on.

The tick resolution and the half-even rounding are the module's own
contract; the clock section follows XQsim's shape (domain labels with
frequencies over one tick core).
"""

from decimal import Decimal

import pytest

from decsim.config import (
    TICKS_PER_MICROSECOND,
    ClockSettings,
    check_duration,
    format_ticks,
    microseconds_to_ticks,
)
from decsim.machine import Machine, MachineSettings
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.qpu.settings import QpuSettings


def test_us_converts_with_fixed_resolution_and_half_even_rounding():
    """Microseconds convert at fixed resolution with half-even tie rounding."""
    half_a_tick = Decimal("0.0000005")
    one_and_a_half_ticks = Decimal("0.0000015")
    two_and_a_half_ticks = Decimal("0.0000025")
    assert microseconds_to_ticks(1.25) == 1_250_000
    assert microseconds_to_ticks(0) == 0
    assert microseconds_to_ticks(half_a_tick) == 0
    assert microseconds_to_ticks(one_and_a_half_ticks) == 2
    assert microseconds_to_ticks(two_and_a_half_ticks) == 2


def test_us_performs_no_domain_validation():
    """The conversion helper leaves input-domain restrictions to its callers."""
    a_quarter_tick = Decimal("0.00000025")
    assert microseconds_to_ticks(-1.25) == -1_250_000
    assert microseconds_to_ticks(a_quarter_tick) == 0
    assert microseconds_to_ticks(True) == TICKS_PER_MICROSECOND

    with pytest.raises(ValueError):
        microseconds_to_ticks(float("nan"))
    with pytest.raises(OverflowError):
        microseconds_to_ticks(float("inf"))
    with pytest.raises(TypeError):
        microseconds_to_ticks("1.0")


def test_fmt_renders_the_exact_log_format():
    """Tick formatting uses fixed padding, precision, sign, and units."""
    assert format_ticks(0) == "  0.000 us"
    assert format_ticks(1_234_567) == "  1.235 us"
    assert format_ticks(-1_000_000) == " -1.000 us"
    assert format_ticks(True) == "  0.000 us"

    with pytest.raises(TypeError):
        format_ticks("1000000")


@pytest.mark.parametrize(
    "invalid_value",
    [-1, float("nan"), float("inf"), float("-inf")],
    ids=["negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_a_duration_refuses_negative_and_nonfinite_values_by_name(
    invalid_value,
):
    with pytest.raises(
        ValueError, match="round_period must be a finite nonnegative number"
    ):
        check_duration("round_period", invalid_value)


def test_a_positive_duration_that_collapses_to_zero_ticks_is_refused():
    with pytest.raises(
        ValueError, match="round_period is positive but rounds to zero ticks"
    ):
        a_quarter_tick = Decimal("0.00000025")
        check_duration("round_period", a_quarter_tick)


def test_the_clocks_section_turns_cycles_into_microseconds():
    clocks = ClockSettings.from_yaml({"fridge": 250.0, "room": 500})
    assert clocks.megahertz("fridge") == 250.0
    assert clocks.microseconds(25, "fridge") == 0.1
    assert clocks.microseconds(8, "room") == 0.016


def test_a_clock_that_is_not_a_positive_frequency_is_refused():
    with pytest.raises(
        ValueError,
        match="clock fridge must be a positive frequency in megahertz, got 0",
    ):
        ClockSettings.from_yaml({"fridge": 0})


def test_a_cycle_count_on_an_unknown_clock_names_the_clocks():
    clocks = ClockSettings.from_yaml({"fridge": 250.0, "room": 250.0})
    with pytest.raises(
        ValueError,
        match="clock 'sfq' is not a clocks entry; the clocks are "
        r"\['fridge', 'room'\]",
    ):
        clocks.microseconds(1, "sfq")


def test_planner_rejects_a_zero_round_period_only_when_selected():
    """A selected zero cadence is refused; a card cadence overrides it."""
    zero_qpu = QpuSettings(round_period_microseconds=0.0)
    zero_period = MachineSettings(qpu=zero_qpu)
    with pytest.raises(
        ValueError, match="resolved round cadence must be at least one tick"
    ):
        Machine.build(zero_period)

    card = SurfaceCodeModel(round_microseconds=2.0)
    carded_qpu = QpuSettings(round_period_microseconds=0.0, code=card)
    carded = MachineSettings(qpu=carded_qpu)
    machine = Machine.build(carded)
    assert machine.qpu.cycle_ticks == 2_000_000


def test_the_round_period_comes_from_the_code_card_then_the_qpu_settings():
    """The code card cadence wins over the QPU settings round period."""
    fallback_qpu = QpuSettings(round_period_microseconds=0.75)
    fallback = MachineSettings(qpu=fallback_qpu)
    machine = Machine.build(fallback)
    assert machine.qpu.cycle_ticks == 750_000

    card = SurfaceCodeModel(round_microseconds=2.0)
    carded_qpu = QpuSettings(round_period_microseconds=1.25, code=card)
    carded = MachineSettings(qpu=carded_qpu)
    machine = Machine.build(carded)
    assert machine.qpu.cycle_ticks == 2_000_000
