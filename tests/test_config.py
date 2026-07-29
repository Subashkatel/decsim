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
    profile = LinkModelConfig.reference_fixed_latency_profile()
    assert profile.dd.channel.propagation_latency_ticks == us(0.5)
    assert profile.wsd.channel.propagation_latency_ticks == us(0.5)
    with pytest.raises(TypeError):
        TimingConfig(t_dd_us=0.7)
