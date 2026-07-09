#==================================================================
# TESTS FOR TIMING CONSTANTS
#==================================================================
from decsim.config import TICKS_PER_US, TimingConfig, us, fmt


def test_ticks_per_us():
    assert TICKS_PER_US == 1_000_000

def test_us():
    assert us(1.1) == 1_100_000
    assert us(2.0) == 2_000_000

def test_fmt():
    assert fmt(1_100_000) == "  1.100 us"
    assert fmt(2_000_000) == "  2.000 us"


def test_ws_link_defaults_to_dd():
    """ws is the weak<->strong escalation link: follows t_dd unless overridden."""
    t = TimingConfig(t_dd_us=0.7)
    assert t.ticks("t_dd") == us(0.7) and t.ticks("t_ws") == us(0.7)
    t = TimingConfig(t_dd_us=0.7, t_ws_us=0.2)
    assert t.ticks("t_ws") == us(0.2)
