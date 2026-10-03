"""The switching slot's record: its threshold, its costs, its slots."""

import math

import pytest

import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.settings as machine_settings
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


def test_a_likelihood_ratio_of_one_hundred_is_twenty_decibels():
    """Decibels are 10 log10 of the ratio, the weight its natural log."""
    weight_nats = math.log(100.0)

    assert threshold_sources.nats_to_decibels(weight_nats) == pytest.approx(
        20.0
    )


def test_a_switching_record_refuses_a_cost_by_its_own_field():
    with pytest.raises(ValueError, match="^threshold_cycles must not be"):
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
