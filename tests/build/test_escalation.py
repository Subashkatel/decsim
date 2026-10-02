"""Building the switching part: the signal, the policy, the strong side.

A run whose switching slot is filled gets the confidence its weak
decoder reports, the policy that decides on it with the slot's
threshold, and the strong window side; a run with none gets nothing,
and the escalation ports stay unbound.
"""

import types

import decsim.build.escalation as escalation_build
import decsim.burst_detectors.event_count.detector as event_count
import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
from decsim.burst_detectors.masked_regional_cusum import (
    detector as masked_regional_cusum,
)
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# The round period the qpu card names by default.
ROUND_PERIOD_MICROSECONDS = 1.1


def _switching(**changes) -> escalation_settings.SwitchingSettings:
    """A switching slot on a fixed threshold and the complementary gap."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, **changes
    )


def _weak() -> decoder_settings.DecoderPoolSettings:
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    return decoder_settings.DecoderPoolSettings(algorithm=matching)


def _plan_that_scores_nothing() -> types.SimpleNamespace:
    """A plan that scores no operation calibrates a detector on nothing."""
    run_plan = types.SimpleNamespace(resolved_operations=())
    return types.SimpleNamespace(run_plan=run_plan, planned_operations=())


def test_a_run_with_no_switching_builds_no_switching_part():
    engine = engine_module.Engine()

    switching = escalation_build.build_switching(None, None, engine)

    assert switching is None


def test_the_policy_decides_on_the_signals_source_and_the_threshold():
    settings = _switching()
    weak = _weak()
    engine = engine_module.Engine()

    switching = escalation_build.Switching.build(settings, weak, engine)

    policy = switching.policy
    twenty_decibels = threshold_sources.decibels_to_nats(20.0)
    assert policy.threshold.threshold_nats == twenty_decibels
    assert policy.expected_source is switching.confidence_signal.source
    assert policy.run_both_at_once is False


def test_every_machine_builds_its_own_policy_from_one_settings_record():
    """One settings record builds a machine per shot.

    Each machine binds its burst detector onto the policy's port, and a
    port takes one peer, so each build makes a policy of its own.
    """
    settings = _switching()
    weak = _weak()

    first_engine = engine_module.Engine()
    second_engine = engine_module.Engine()

    first = escalation_build.Switching.build(settings, weak, first_engine)
    second = escalation_build.Switching.build(settings, weak, second_engine)

    assert first.policy is not second.policy
    assert first.strong_redecode is not second.strong_redecode


def test_the_strong_window_record_declares_whether_it_absorbs():
    redo_window = _switching()
    double_window = strong_window_shapes.DoubleWindow.Settings()
    forward = _switching(strong_window=double_window)

    assert redo_window.strong_window.name == "redo_window"
    assert forward.strong_window.name == "double_window"
    assert redo_window.strong_window.absorbs_weak_windows is False
    assert forward.strong_window.absorbs_weak_windows is True


def test_the_switching_part_builds_the_strong_window_its_slot_names():
    double_window = strong_window_shapes.DoubleWindow.Settings()
    settings = _switching(strong_window=double_window)
    weak = _weak()
    engine = engine_module.Engine()

    switching = escalation_build.Switching.build(settings, weak, engine)

    assert isinstance(switching.shape, strong_window_shapes.DoubleWindow)


def test_the_confidence_row_is_built_with_the_sections_walk_card():
    confidence = complementary.ComplementaryGap.Settings(walk_microseconds=0.25)
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    settings = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    weak = _weak()
    engine = engine_module.Engine()

    switching = escalation_build.Switching.build(settings, weak, engine)

    assert switching.confidence_signal.walk_microseconds == 0.25


def test_the_switching_slots_detector_is_the_one_built():
    detector = event_count.EventCountBurstDetector.Settings()
    switching = _switching(burst_detector=detector)
    plan = _plan_that_scores_nothing()
    engine = engine_module.Engine()

    built = escalation_build.build_burst_detector(
        switching, ROUND_PERIOD_MICROSECONDS, None, engine, plan
    )

    assert isinstance(built, event_count.EventCountBurstDetector)


def test_a_count_that_names_no_clock_counts_on_the_machines():
    machine_clock = config.Clock(4000)
    detector = event_count.EventCountBurstDetector.Settings(cycles_per_round=1)
    switching = _switching(burst_detector=detector)
    plan = _plan_that_scores_nothing()
    engine = engine_module.Engine()

    built = escalation_build.build_burst_detector(
        switching, ROUND_PERIOD_MICROSECONDS, machine_clock, engine, plan
    )

    assert built.settings.clock == machine_clock


def test_a_chart_bank_that_names_no_clock_stays_unpriced():
    """With no clock the bank publishes a verdict as its round is formed."""
    detector = masked_regional_cusum.MaskedRegionalCusumBurstDetector.Settings()
    switching = _switching(burst_detector=detector)
    machine_clock = config.Clock(4000)
    plan = _plan_that_scores_nothing()
    engine = engine_module.Engine()

    built = escalation_build.build_burst_detector(
        switching, ROUND_PERIOD_MICROSECONDS, machine_clock, engine, plan
    )

    assert built.settings.clock is None


def test_a_run_with_no_switching_builds_no_detector():
    engine = engine_module.Engine()

    built = escalation_build.build_burst_detector(
        None, ROUND_PERIOD_MICROSECONDS, None, engine, None
    )

    assert built is None
