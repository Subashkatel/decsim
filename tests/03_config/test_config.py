from dataclasses import FrozenInstanceError, fields
from decimal import Decimal

import pytest

import decsim
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.config import TICKS_PER_US, TimingConfig, fmt, us
from decsim.run_spec import RunSpec


def timing_with(field_name, value):
    values = {
        "round_us": 1.1,
        "t_binary_availability_us": 0.0,
        "t_pack_us": 0.0,
    }
    values[field_name] = value
    return TimingConfig(**values)


def built_round_ticks(**overrides):
    return RunSpec(ops=[], **overrides).build().qpu.cycle_ticks


def test_us_converts_with_fixed_resolution_and_half_even_rounding():
    """Microseconds convert at fixed resolution with half-even tie rounding."""
    assert us(1.25) == 1_250_000
    assert us(0) == 0
    assert us(Decimal("0.0000005")) == 0
    assert us(Decimal("0.0000015")) == 2
    assert us(Decimal("0.0000025")) == 2


def test_us_performs_no_domain_validation():
    """The conversion helper leaves input-domain restrictions to its callers."""
    assert us(-1.25) == -1_250_000
    assert us(Decimal("0.00000025")) == 0
    assert us(True) == TICKS_PER_US

    with pytest.raises(ValueError):
        us(float("nan"))
    with pytest.raises(OverflowError):
        us(float("inf"))
    with pytest.raises(TypeError):
        us("1.0")


def test_fmt_renders_the_exact_log_format():
    """Tick formatting uses fixed padding, precision, sign, and units."""
    assert decsim.fmt is fmt
    assert fmt(0) == "  0.000 us"
    assert fmt(1_234_567) == "  1.235 us"
    assert fmt(-1_000_000) == " -1.000 us"
    assert fmt(True) == "  0.000 us"

    with pytest.raises(TypeError):
        fmt("1000000")


def test_timing_config_defaults_field_order_and_public_exports():
    """Timing configuration has the accepted ordered defaults and package surface."""
    config = TimingConfig()

    assert [field.name for field in fields(TimingConfig)] == [
        "round_us",
        "t_binary_availability_us",
        "t_pack_us",
    ]
    assert (
        config.round_us,
        config.t_binary_availability_us,
        config.t_pack_us,
    ) == (1.1, 0.0, 0.0)
    assert RunSpec(ops=[]).timing == config
    assert decsim.TimingConfig is TimingConfig
    assert decsim.us is us
    assert not hasattr(decsim, "TICKS_PER_US")


def test_timing_config_is_frozen_hashable_and_value_comparable():
    """Timing configurations compare by value, hash, and reject assignment."""
    first = TimingConfig(t_pack_us=0.25)
    second = TimingConfig(t_pack_us=0.25)

    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(FrozenInstanceError):
        first.t_pack_us = 0.5


def test_timing_config_accepts_natural_numeric_and_bool_values():
    """Numeric scalars and bools follow their natural arithmetic behavior."""
    config = TimingConfig(
        round_us=True,
        t_binary_availability_us=False,
        t_pack_us=Decimal("0.000001"),
    )

    assert config.round_us is True
    assert config.t_binary_availability_us is False
    assert config.ticks("t_binary_availability") == 0
    assert config.ticks("t_pack") == 1

    assert TimingConfig(round_us=False).round_us is False
    with pytest.raises(TypeError):
        TimingConfig(round_us="1.0")


@pytest.mark.parametrize(
    "field_name",
    ["round_us", "t_binary_availability_us", "t_pack_us"],
)
@pytest.mark.parametrize(
    "invalid_value",
    [-1, float("nan"), float("inf"), float("-inf")],
    ids=["negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_timing_config_rejects_nonfinite_or_negative_fields(
    field_name, invalid_value
):
    """Every timing field rejects negative and nonfinite values by name."""
    with pytest.raises(ValueError, match=field_name):
        timing_with(field_name, invalid_value)


@pytest.mark.parametrize(
    "field_name",
    ["round_us", "t_binary_availability_us", "t_pack_us"],
)
def test_timing_config_rejects_positive_values_that_collapse_to_zero_ticks(
    field_name,
):
    """Every positive timing field must survive integer-tick discretization."""
    with pytest.raises(ValueError, match=field_name):
        timing_with(field_name, Decimal("0.00000025"))


@pytest.mark.parametrize(
    "field_name",
    ["round_us", "t_binary_availability_us", "t_pack_us"],
)
def test_timing_config_accepts_zero_and_one_tick_on_every_axis(field_name):
    """Every timing axis accepts exact zero and the smallest positive tick."""
    assert getattr(timing_with(field_name, 0), field_name) == 0
    one_tick = timing_with(field_name, Decimal("0.000001"))
    assert us(getattr(one_tick, field_name)) == 1


def test_optional_timing_axes_are_independent():
    """Either optional stage can be enabled without changing the other stage."""
    binary_only = TimingConfig(t_binary_availability_us=Decimal("0.000002"))
    pack_only = TimingConfig(t_pack_us=Decimal("0.000003"))
    both = TimingConfig(
        t_binary_availability_us=Decimal("0.000002"),
        t_pack_us=Decimal("0.000003"),
    )

    assert (
        binary_only.ticks("t_binary_availability"),
        binary_only.ticks("t_pack"),
    ) == (2, 0)
    assert (
        pack_only.ticks("t_binary_availability"),
        pack_only.ticks("t_pack"),
    ) == (0, 3)
    assert (
        both.ticks("t_binary_availability"),
        both.ticks("t_pack"),
    ) == (2, 3)


def test_ticks_accepts_only_optional_names_and_preserves_key_error():
    """Named tick lookup accepts two stages and preserves natural missing-key errors."""
    config = TimingConfig(
        t_binary_availability_us=0.25,
        t_pack_us=0.5,
    )

    assert config.ticks("t_binary_availability") == 250_000
    assert config.ticks("t_pack") == 500_000
    for missing_name in ("round_us", "unknown"):
        with pytest.raises(KeyError) as error:
            config.ticks(missing_name)
        assert error.value.args == (missing_name,)


def test_round_ticks_is_absent():
    """Cadence ticks are not exposed as a TimingConfig property."""
    assert not hasattr(TimingConfig(), "round_ticks")


def test_planner_rejects_zero_cadence_only_when_selected():
    """Planning rejects a selected zero cadence but ignores an overridden fallback."""
    zero_timing = TimingConfig(round_us=0.0)
    assert TimingConfig(round_us=False).round_us is False

    with pytest.raises(
        ValueError, match="resolved round cadence must be at least one tick"
    ):
        RunSpec(ops=[], timing=zero_timing).build()

    assert built_round_ticks(timing=zero_timing, round_us=0.75) == 750_000


def test_run_cadence_uses_code_then_run_spec_then_timing_precedence():
    """Cadence resolves from code, run override, then timing fallback."""
    timing = TimingConfig(round_us=0.75)

    assert built_round_ticks(timing=timing) == 750_000
    assert built_round_ticks(timing=timing, round_us=1.25) == 1_250_000
    assert built_round_ticks(
        timing=timing,
        round_us=1.25,
        code=SurfaceCodeModel(round_us=2.0),
    ) == 2_000_000
