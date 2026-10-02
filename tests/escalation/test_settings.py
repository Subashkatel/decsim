"""Escalation control stages use gem5's integer cycles and named domains."""

import math

import pytest

import decsim.confidence.signals as confidence_signals
import decsim.config as config
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources


def test_switching_reads_both_cycle_costs_on_the_named_clock():
    clocks = config.ClockSettings({"decisions": 125.0})
    section = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "clock": "decisions",
        "threshold_cycles": 3,
        "switch_cycles": 4,
    }
    settings = escalation_settings.EscalationSettings.from_yaml(
        section,
        clocks,
        confidence_settings=confidence_signals.confidence_settings,
    )
    assert settings.clock.period_ticks == 8000
    assert settings.threshold_cycles == 3
    assert settings.switch_cycles == 4


def test_an_unnamed_escalation_clock_uses_the_controller_clock():
    clocks = config.ClockSettings({})
    controller_clock = config.Clock(123)
    settings = escalation_settings.EscalationSettings.from_yaml(
        {}, clocks, default_clock=controller_clock
    )
    assert settings.clock is controller_clock
    assert settings.threshold_cycles == 0
    assert settings.switch_cycles == 0


def test_a_charged_escalation_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged escalation costs need a clock"
    ):
        escalation_settings.EscalationSettings(threshold_cycles=1)


def test_a_likelihood_ratio_of_one_hundred_is_twenty_decibels():
    """Decibels are 10 log10 of the ratio, the weight its natural log."""
    weight_nats = math.log(100.0)

    assert threshold_sources.nats_to_decibels(weight_nats) == pytest.approx(
        20.0
    )
