"""Kernel timing module: ticks, conversions, frozen link defaults."""
import dataclasses

import pytest

from decsim.config import TICKS_PER_US, TimingConfig, fmt, us


def test_tick_conversion():
    assert TICKS_PER_US == 1_000_000
    assert us(1.1) == 1_100_000
    assert us(0.15) == 150_000
    assert fmt(1_100_000).strip() == "1.100 us"


def test_link_defaults_match_today():
    t = TimingConfig()
    assert t.ticks("t_qc") == 150_000
    assert t.ticks("t_cd") == 2_000_000
    assert t.ticks("t_dd") == 500_000
    assert t.ticks("t_do") == 1_000_000
    assert t.ticks("t_oc") == 4_000_000
    assert t.ticks("t_cq") == 150_000
    assert t.round_ticks == 1_100_000


def test_ws_default_and_override():
    # today: LinkModel defaults ws=us(0.5) (links.py:47); SimConfig's t_ws_us=None
    # means "use that default"
    assert TimingConfig().ticks("t_ws") == 500_000
    assert TimingConfig(t_ws_us=0.7).ticks("t_ws") == 700_000


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TimingConfig().round_us = 2.0
