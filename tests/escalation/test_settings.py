"""The switching slot's record: its threshold, its costs, its slots."""

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

    with pytest.raises(AttributeError, match="object has no attribute"):
        machine = machine_module.Machine.build(one_decoder, 0)
        machine.run()
