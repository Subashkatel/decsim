"""The switching slot read from the escalation section.

The verdict's control stages use gem5's integer cycles and named
domains, and the section's kind says which decode slots a run fills.
"""

import math

import pytest

import decsim.confidence.complementary as complementary
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
    settings = escalation_settings.SwitchingSettings.from_yaml(
        section, clocks, None, None, confidence_signals.confidence_settings
    )
    assert settings.clock.period_ticks == 8000
    assert settings.threshold_cycles == 3
    assert settings.switch_cycles == 4


def test_an_unnamed_escalation_clock_uses_the_controller_clock():
    clocks = config.ClockSettings({})
    controller_clock = config.Clock(123)
    section = {"kind": "switching", "gap_threshold_db": 20.0}
    settings = escalation_settings.SwitchingSettings.from_yaml(
        section,
        clocks,
        None,
        controller_clock,
        confidence_signals.confidence_settings,
    )
    assert settings.clock is controller_clock
    assert settings.threshold_cycles == 0
    assert settings.switch_cycles == 0


def test_a_charged_escalation_cost_needs_its_clock():
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(threshold_nats=2.0)
    with pytest.raises(
        ValueError, match="charged escalation costs need a clock"
    ):
        escalation_settings.SwitchingSettings(
            confidence=confidence, threshold=threshold, threshold_cycles=1
        )


@pytest.mark.parametrize(
    "kind, slots",
    [
        ("weak_baseline", ("weak_decoder",)),
        ("strong_only", ("strong_decoder",)),
        ("switching", ("weak_decoder", "strong_decoder")),
    ],
)
def test_each_kind_names_the_decoder_slots_a_run_fills(kind, slots):
    section = {"kind": kind}

    named = escalation_settings.escalation_kind(section)

    assert escalation_settings.ESCALATION_KINDS[named] == slots


def test_an_escalation_kind_that_names_no_row_is_refused():
    with pytest.raises(ValueError) as refusal:
        escalation_settings.escalation_kind({"kind": "sometimes"})

    sentence = str(refusal.value)
    assert "escalation.kind" in sentence
    assert "sometimes" in sentence


def test_a_kind_that_keeps_one_decoder_fills_no_switching_slot():
    clocks = config.ClockSettings({})
    section = {"kind": "weak_baseline", "threshold_cycles": 0}

    switching = escalation_settings.SwitchingSettings.from_yaml(
        section, clocks, None, None, confidence_signals.confidence_settings
    )

    assert switching is None


def test_a_kind_that_keeps_one_decoder_refuses_the_confidence_keys():
    clocks = config.ClockSettings({})
    section = {"kind": "strong_only", "gap_threshold_db": 20.0}

    with pytest.raises(ValueError, match="decides on no confidence"):
        escalation_settings.SwitchingSettings.from_yaml(
            section, clocks, None, None, confidence_signals.confidence_settings
        )


def test_a_likelihood_ratio_of_one_hundred_is_twenty_decibels():
    """Decibels are 10 log10 of the ratio, the weight its natural log."""
    weight_nats = math.log(100.0)

    assert threshold_sources.nats_to_decibels(weight_nats) == pytest.approx(
        20.0
    )
