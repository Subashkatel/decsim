"""The switching slot's record and the decoder slots it switches between.

SwitchingSettings (decsim/escalation/settings.py) is Toshio et al.
2510.25222 Sec. III A's switch, weak first and strong on low
confidence; MachineSettings (decsim/settings.py) holds the two decoder
slots. The laws here are the slot's own refusals and the build's stop
when a slot it needs is empty.
"""

import dataclasses

import pytest

import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.machine as machine_module
import decsim.settings as machine_settings
import tests.escalation.test_strong_window_shapes as shape_tests
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


def _switching(**changes) -> escalation_settings.SwitchingSettings:
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, **changes
    )


def test_a_switching_record_refuses_a_cost_by_its_own_field():
    with pytest.raises(ValueError, match="threshold_cycles"):
        _switching(threshold_cycles=-1)


def test_a_switching_record_refuses_a_strong_answer_route_it_lacks():
    with pytest.raises(ValueError) as refusal:
        _switching(strong_answer_route="by_courier")
    assert str(refusal.value) == (
        "switching.strong_answer_route must be one of "
        "('direct', 'through_weak_chip'), got 'by_courier'"
    )


def test_a_route_through_the_weak_chip_with_no_clock_is_refused():
    """The chip's commit is one cycle, and a run with no clock has none."""
    settings = shape_tests.gate_switching()
    switching = dataclasses.replace(
        settings.switching, strong_answer_route="through_weak_chip"
    )
    clockless = dataclasses.replace(settings, clock=None, switching=switching)

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(clockless, 0)
    assert str(refusal.value) == (
        "switching.strong_answer_route 'through_weak_chip' commits on the "
        "weak chip in clock cycles, so it needs switching.clock or the "
        "machine's clock"
    )


def test_two_decoders_without_switching_are_refused():
    """With no switching every window decodes on one decoder.

    The strong decoder would be built and never handed a window, so the
    run would go on as if it were not there.
    """
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    pool = decoder_settings.DecoderPoolSettings(algorithm=matching)

    with pytest.raises(ValueError, match="both set and switching is not"):
        machine_settings.MachineSettings(weak_decoder=pool, strong_decoder=pool)


@pytest.mark.parametrize("empty_slot", ["weak_decoder", "strong_decoder"])
def test_switching_with_an_empty_decoder_slot_still_stops(empty_slot):
    """No weak tier stops the build; no strong tier, the first escalation."""
    settings = shape_tests.gate_switching()
    one_decoder = dataclasses.replace(settings, **{empty_slot: None})

    with pytest.raises(AttributeError):
        machine = machine_module.Machine.build(one_decoder, 0)
        machine.run()
