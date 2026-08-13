#==================================================================
# TESTS FOR TIMING CONSTANTS
#==================================================================
import pytest

from decsim.config import TICKS_PER_US, TimingConfig, us, fmt
from decsim.links import LinkModelConfig


def test_ticks_per_us():
    assert TICKS_PER_US == 1_000_000

def test_us():
    assert us(1.1) == 1_100_000
    assert us(2.0) == 2_000_000

def test_fmt():
    assert fmt(1_100_000) == "  1.100 us"
    assert fmt(2_000_000) == "  2.000 us"


def test_link_timing_is_not_owned_by_timing_config():
    profile = LinkModelConfig.logical_reference_profile()
    assert profile.dd.channel.propagation_latency_ticks == us(0.5)
    assert profile.wsd.channel.propagation_latency_ticks == us(0.5)
    with pytest.raises(TypeError):
        TimingConfig(t_dd_us=0.7)


def test_controller_readout_cost_is_explicit_non_link_timing():
    assert TimingConfig().ticks("t_binary_availability") == 0
    assert TimingConfig(t_binary_availability_us=0.4).ticks("t_binary_availability") == 400_000


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_non_link_timing_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        TimingConfig(t_binary_availability_us=value)


def test_non_link_timing_rejects_wrong_type_and_subtick_values():
    with pytest.raises(TypeError, match="int or float"):
        TimingConfig(t_binary_availability_us="1")
    with pytest.raises(ValueError, match="rounds to zero"):
        TimingConfig(t_binary_availability_us=0.0000001)
