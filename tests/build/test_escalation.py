"""Building the switching part: the signal, the policy, the strong side.

A run whose switching slot is filled gets the confidence its weak
decoder reports, the policy that decides on it with the slot's
threshold, and the strong window side; a run with none gets nothing,
and the escalation ports stay unbound.
"""

import pytest

import decsim.build.escalation as escalation_build
import decsim.burst_detectors.event_count.detector as event_count
import decsim.burst_detectors.settings as burst_detector_settings
import decsim.confidence.complementary as complementary
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.settings as machine_settings


def _switching(**changes) -> escalation_settings.SwitchingSettings:
    """A switching slot on a fixed threshold and the complementary gap."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(threshold_nats=2.0)
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, **changes
    )


def _weak() -> decoder_settings.DecoderPoolSettings:
    matching = mwpm.PyMatchingDecoder.Settings()
    return decoder_settings.DecoderPoolSettings(algorithm=matching)


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
    assert policy.threshold.threshold_nats == 2.0
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


def test_a_table_threshold_with_no_number_from_the_experiment_is_refused():
    """The table row is resolved per sweep point, before the root builds."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.TableThreshold.Settings(
        table="calibration.csv"
    )
    settings = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    weak = _weak()
    engine = engine_module.Engine()

    with pytest.raises(ValueError) as refusal:
        escalation_build.Switching.build(settings, weak, engine)

    assert "resolves the threshold per sweep point" in str(refusal.value)


def test_the_strong_window_record_declares_whether_it_absorbs():
    redo_window = _switching()
    double_window = strong_window_shapes.DoubleWindow.Settings()
    forward = _switching(strong_window=double_window)

    assert redo_window.strong_window.name == "redo_window"
    assert forward.strong_window.name == "double_window"
    assert escalation_build.absorbs_weak_windows(redo_window) is False
    assert escalation_build.absorbs_weak_windows(forward) is True
    assert escalation_build.absorbs_weak_windows(None) is False


def test_the_switching_part_builds_the_strong_window_its_slot_names():
    double_window = strong_window_shapes.DoubleWindow.Settings()
    settings = _switching(strong_window=double_window)
    weak = _weak()
    engine = engine_module.Engine()

    switching = escalation_build.Switching.build(settings, weak, engine)

    assert isinstance(switching.shape, strong_window_shapes.DoubleWindow)


def test_the_confidence_row_is_built_with_the_sections_walk_card():
    confidence = complementary.ComplementaryGap.Settings(walk_microseconds=0.25)
    threshold = threshold_sources.FixedThreshold.Settings(threshold_nats=2.0)
    settings = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    weak = _weak()
    engine = engine_module.Engine()

    switching = escalation_build.Switching.build(settings, weak, engine)

    assert switching.confidence_signal.walk_microseconds == 0.25


def test_a_burst_detector_beside_a_run_with_no_switching_is_refused():
    """A flagged window goes to the strong tier, which weak_baseline lacks."""
    row_settings = event_count.EventCountBurstDetector.Settings()
    section = burst_detector_settings.BurstDetectorSettings(
        kind="event_count", row_settings=row_settings
    )
    settings = machine_settings.MachineSettings(burst_detector=section)
    engine = engine_module.Engine()

    with pytest.raises(ValueError, match="only escalation.kind switching"):
        escalation_build.build_burst_detector(settings, engine, None)
