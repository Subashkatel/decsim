"""Building the switching part: the signal, the policy, the strong side.

A run whose switching slot is filled gets the confidence its weak
decoder reports, the policy that decides on it with the slot's
threshold, and the strong window side; a run with none gets nothing,
and the escalation ports stay unbound.
"""

import dataclasses

import pytest

import decsim.build.escalation as escalation_build
import decsim.build.layout as layout_module
import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.settings as machine_settings
import tests.topology.test_graph_layout as layout_tests
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


def _built(
    settings: escalation_settings.SwitchingSettings,
    engine: engine_module.Engine,
) -> escalation_build.Switching:
    """The switching part of the one edge a weak and a strong pool make."""
    weak = _weak()
    machine = machine_settings.MachineSettings(
        weak_decoder=weak, strong_decoder=weak, switching=settings
    )
    layout = layout_module.legacy_layout(machine)
    switching_parts = escalation_build.build_switching(layout, engine)
    return switching_parts[0]


def test_every_machine_builds_its_own_policy_from_one_settings_record():
    """One settings record builds a machine per shot, each its own policy."""
    settings = _switching()
    first_engine = engine_module.Engine()
    second_engine = engine_module.Engine()

    first = _built(settings, first_engine)
    second = _built(settings, second_engine)

    assert first.policy is not second.policy
    assert first.strong_redecode is not second.strong_redecode


def test_a_switching_record_builds_the_policy_its_run_binds():
    """A record that subclasses the switching slot brings its own policy."""
    settings = _switching(_OwnPolicySwitching)
    engine = engine_module.Engine()

    switching = _built(settings, engine)

    assert isinstance(switching.policy, _OwnPolicy)


_DOUBLE_WINDOW = strong_window_shapes.DoubleWindow.Settings()
_TWO_SOLVES = complementary.ComplementaryGap.Settings()
_PAST_THE_ENTRY_CHANGES = [
    {"run_both_at_once": True},
    {"strong_window": _DOUBLE_WINDOW},
    {"confidence": _TWO_SOLVES},
]
_PAST_THE_ENTRY_IDS = ["run_both_at_once", "double_window", "two_solves"]


@pytest.mark.parametrize(
    "change", _PAST_THE_ENTRY_CHANGES, ids=_PAST_THE_ENTRY_IDS
)
def test_an_edge_past_the_entry_refuses_what_only_the_entry_does(change):
    """An edge past the entry waits for stage 5 on what reads its source.

    A speculative decode, an absorbing region and a gap of two solves
    each work on the entry pool's windows alone.
    """
    deep_switching = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, **change
    )
    chain = layout_tests.chain_machine(deep_switching)
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()

    with pytest.raises(NotImplementedError, match=r"stage 5"):
        escalation_build.build_switching(layout, engine)


def test_an_edge_past_the_entry_builds_on_one_solve():
    chain = layout_tests.chain_machine()
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()

    parts = escalation_build.build_switching(layout, engine)

    edges = [part.edge.name for part in parts]
    assert edges == ["switching", layout_tests.DEEP_SWITCHING]
