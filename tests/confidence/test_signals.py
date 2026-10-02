"""The soft-output rows a switching run's weak decoder can report.

CONFIDENCE_SIGNALS is the table escalation.confidence names a row of,
and every row builds itself from the escalation section and the weak
decoder's settings (from_settings), so a row written outside decsim
reaches the root through that one call. Each row declares what evidence
it needs from the decode and which logical classes the window must be
decoded in, so the window side asks the weak decoder for exactly those
solves and never for the row's class.
"""

import pytest

import decsim.confidence.signals as confidence_signals
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings

ESCALATION = escalation_settings.EscalationSettings(
    kind="switching",
    confidence_walk_microseconds=0.5,
    gap_threshold_nats=2.0,
)
UNION_FIND = union_find.UnionFindDecoder.Settings()
WEAK = decoder_settings.DecoderPoolSettings(algorithm=UNION_FIND)


def test_the_shipped_rows_are_reachable_by_the_name_the_yaml_writes():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    assert sorted(rows) == [
        "cluster_gap",
        "complementary_gap",
        "extra_cluster_gap",
    ]


@pytest.mark.parametrize("name", sorted(confidence_signals.CONFIDENCE_SIGNALS))
def test_every_row_builds_from_the_two_settings_and_reads_the_card(name):
    """A row written outside decsim reaches the root through this call."""
    row = confidence_signals.CONFIDENCE_SIGNALS[name]
    built = row.from_settings(ESCALATION, WEAK)

    assert built.walk_microseconds == 0.5


def test_every_row_declares_the_classes_the_window_must_be_solved_in():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    complementary = rows["complementary_gap"].from_settings(ESCALATION, WEAK)
    cluster = rows["cluster_gap"].from_settings(ESCALATION, WEAK)
    extra = rows["extra_cluster_gap"].from_settings(ESCALATION, WEAK)

    assert complementary.forced_logical_classes == (0, 1)
    assert cluster.forced_logical_classes == ()
    assert extra.forced_logical_classes == ()


@pytest.mark.parametrize("name", sorted(confidence_signals.CONFIDENCE_SIGNALS))
def test_every_row_declares_the_evidence_it_needs_from_the_decode(name):
    row = confidence_signals.CONFIDENCE_SIGNALS[name]

    assert row.decoder_evidence_requirement is not None
    assert row.evidence_refusal


def test_every_row_names_its_own_source():
    """The threshold is held in one unit, so a source says which signal."""
    rows = confidence_signals.CONFIDENCE_SIGNALS
    built_rows = [row.from_settings(ESCALATION, WEAK) for row in rows.values()]
    methods = {built.source.method for built in built_rows}

    assert methods == {"complementary_gap", "cluster_gap", "extra_cluster_gap"}
