"""Building the escalation policy: the object first, the table second.

The policy instance is the authority over its own tier and the table row
is only how a yaml names one, which is sinter's order too
(sinter/_collection/_mux_sampler.py:33-40 resolves the caller's own
sampler before its built-in table) and gem5's
(src/python/m5/SimObject.py:204-205 reads a built object's own params
rather than its class table). Every row is built from one collaborators
record, so a row written outside decsim reaches the root the same way.
"""

import pytest

import decsim.build.escalation as escalation_build
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.records.windows as window_records


def test_the_kinds_row_answers_the_tier_when_no_policy_was_built():
    settings = escalation_settings.EscalationSettings(kind="strong_only")

    tier = escalation_build.primary_tier(settings)

    assert tier == window_records.DecoderTier.STRONG.value


def test_a_built_policy_answers_for_itself_and_the_kind_is_ignored():
    """The object is the one fact; the kind is only a yaml's name for one."""
    built = escalation_policies.StrongOnly(escalation_policies.NO_CONFIDENCE)
    settings = escalation_settings.EscalationSettings(
        kind="weak_baseline", policy=built
    )

    tier = escalation_build.primary_tier(settings)
    row = escalation_build.escalation_row(settings)
    policy = escalation_build.build_escalation_policy(settings)

    assert tier == window_records.DecoderTier.STRONG.value
    assert row is built
    assert policy is built


def test_an_escalation_kind_that_names_no_row_is_refused():
    settings = escalation_settings.EscalationSettings(kind="sometimes")

    with pytest.raises(ValueError) as refusal:
        escalation_build.escalation_row(settings)

    sentence = str(refusal.value)
    assert "escalation.kind" in sentence
    assert "sometimes" in sentence


def test_a_row_that_decides_on_no_confidence_is_built_from_an_empty_record():
    """The section carries none of the confidence keys for such a row."""
    settings = escalation_settings.EscalationSettings(kind="weak_baseline")
    row = escalation_build.escalation_row(settings)

    policy = escalation_build.build_escalation_policy(settings)

    assert row.decides_on_a_confidence is False
    assert isinstance(policy, row)


def test_a_row_that_decides_on_a_confidence_gets_the_three_fields():
    settings = escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
    )

    signal = escalation_build.confidence_signal(settings)
    policy = escalation_build.build_escalation_policy(settings)

    assert policy.decides_on_a_confidence is True
    assert policy.threshold.threshold_nats == 2.0
    assert policy.expected_source is signal.source
    assert policy.run_both_at_once is False


def test_a_table_threshold_with_no_number_from_the_front_is_refused():
    """The table row is resolved per sweep point, before the root builds."""
    settings = escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="table",
        gap_threshold_nats=None,
        confidence="complementary_gap",
    )

    with pytest.raises(ValueError) as refusal:
        escalation_build.build_escalation_policy(settings)

    assert "resolves the threshold per sweep point" in str(refusal.value)


def test_the_strong_window_row_is_named_and_declares_whether_it_absorbs():
    two_sided = escalation_settings.EscalationSettings(
        strong_window="two_sided_context"
    )
    forward = escalation_settings.EscalationSettings(strong_window="forward")

    two_sided_row = escalation_build.strong_window_row(two_sided)
    forward_row = escalation_build.strong_window_row(forward)

    assert two_sided_row.absorbs_weak_windows is False
    assert forward_row.absorbs_weak_windows is True
    assert escalation_build.absorbs_weak_windows(two_sided) is False
    assert escalation_build.absorbs_weak_windows(forward) is True


def test_a_strong_window_that_names_no_row_is_refused():
    settings = escalation_settings.EscalationSettings(strong_window="sideways")

    with pytest.raises(ValueError) as refusal:
        escalation_build.strong_window_row(settings)

    assert "escalation.strong_window" in str(refusal.value)


def test_the_confidence_row_is_built_with_the_sections_walk_card():
    settings = escalation_settings.EscalationSettings(
        confidence="complementary_gap", confidence_walk_microseconds=0.25
    )

    signal = escalation_build.confidence_signal(settings)

    assert signal.walk_microseconds == 0.25
