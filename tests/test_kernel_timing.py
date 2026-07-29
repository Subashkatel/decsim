"""Kernel timing module: ticks, conversions, frozen link defaults."""
import dataclasses

import pytest

from decsim.config import TICKS_PER_US, TimingConfig, fmt, us
from decsim.links import LinkModelConfig


def test_tick_conversion():
    assert TICKS_PER_US == 1_000_000
    assert us(1.1) == 1_100_000
    assert us(0.15) == 150_000
    assert fmt(1_100_000).strip() == "1.100 us"


def test_link_defaults_match_today():
    links = LinkModelConfig.reference_fixed_latency_profile()
    assert links.qc.channel.propagation_latency_ticks == 150_000
    assert links.cwd.channel.propagation_latency_ticks == 2_000_000
    assert links.dd.channel.propagation_latency_ticks == 500_000
    assert links.wdo.channel.propagation_latency_ticks == 1_000_000
    assert links.oc.channel.propagation_latency_ticks == 4_000_000
    assert links.cq.channel.propagation_latency_ticks == 150_000
    assert TimingConfig().round_ticks == 1_100_000


def test_non_link_timing_rejects_link_names():
    with pytest.raises(ValueError, match="unknown non-link timing"):
        TimingConfig().ticks("t_ws")


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TimingConfig().round_us = 2.0
