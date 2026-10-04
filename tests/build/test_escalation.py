"""Building the switching part: the signal, the policy, the strong side.

A run whose switching slot is filled gets the confidence its weak
decoder reports, the policy that decides on it with the slot's
threshold, and the strong window side; a run with none gets nothing,
and the escalation ports stay unbound.
"""

import dataclasses

import decsim.build.escalation as escalation_build
import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


class _OwnPolicy:
    """A policy of a researcher's own; the test reads only its class."""

    def __init__(self, threshold) -> None:
        self.threshold = threshold


@dataclasses.dataclass(frozen=True)
class _OwnPolicySwitching(escalation_settings.SwitchingSettings):
    """A switching record that builds a policy of its own."""

    def build_policy(self, threshold) -> _OwnPolicy:
        """The researcher's policy on the slot's threshold."""
        return _OwnPolicy(threshold)


def _switching(
    record=escalation_settings.SwitchingSettings, **changes
) -> escalation_settings.SwitchingSettings:
    """A switching slot on a fixed threshold and the complementary gap."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return record(confidence=confidence, threshold=threshold, **changes)


def _weak() -> decoder_settings.DecoderPoolSettings:
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    return decoder_settings.DecoderPoolSettings(algorithm=matching)


def test_every_machine_builds_its_own_policy_from_one_settings_record():
    """One settings record builds a machine per shot, each its own policy."""
    settings = _switching()
    weak = _weak()

    first_engine = engine_module.Engine()
    second_engine = engine_module.Engine()

    first = escalation_build.Switching.build(settings, weak, first_engine)
    second = escalation_build.Switching.build(settings, weak, second_engine)

    assert first.policy is not second.policy
    assert first.strong_redecode is not second.strong_redecode


def test_a_switching_record_builds_the_policy_its_run_binds():
    """A record that subclasses the switching slot brings its own policy."""
    settings = _switching(_OwnPolicySwitching)
    engine = engine_module.Engine()

    weak = _weak()
    switching = escalation_build.Switching.build(settings, weak, engine)

    assert isinstance(switching.policy, _OwnPolicy)
