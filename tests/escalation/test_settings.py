"""Escalation control stages use gem5's integer cycles and named domains."""

import pytest

import decsim.config as config
import decsim.escalation.settings as escalation_settings


@pytest.mark.parametrize("key", ["threshold_cycles", "switch_cycles"])
@pytest.mark.parametrize("cycles", [True, 0.5, float("nan"), float("inf")])
def test_escalation_cycle_keys_refuse_nonintegers_by_name(key, cycles):
    clocks = config.ClockSettings({"decisions": 250.0})
    section = {"clock": "decisions", key: cycles}
    sentence = f"escalation.{key} must be a nonnegative integer"
    with pytest.raises(ValueError, match=sentence):
        escalation_settings.EscalationSettings.from_yaml(section, clocks)


def test_switching_reads_both_cycle_costs_on_the_named_clock():
    clocks = config.ClockSettings({"decisions": 125.0})
    section = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "clock": "decisions",
        "threshold_cycles": 3,
        "switch_cycles": 4,
    }
    settings = escalation_settings.EscalationSettings.from_yaml(section, clocks)
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


def test_an_escalation_clock_must_name_a_declared_domain():
    clocks = config.ClockSettings({})
    with pytest.raises(
        ValueError, match="clock 'missing' is not a clocks entry"
    ):
        escalation_settings.EscalationSettings.from_yaml(
            {"clock": "missing"}, clocks
        )


def test_a_charged_escalation_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged escalation costs need a clock"
    ):
        escalation_settings.EscalationSettings(threshold_cycles=1)
