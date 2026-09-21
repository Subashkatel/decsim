"""The soft-output rows a switching run's weak decoder can report.

CONFIDENCE_SIGNALS is the table escalation.confidence names a row of,
and every row builds itself from the escalation section and the weak
decoder's settings (from_settings), so a row written outside decsim
reaches the root through that one call. Each row declares what evidence
it needs from the decode and which logical classes the window must be
decoded in, so the window side asks the weak decoder for exactly those
solves and never for the row's class.
"""

import decsim.confidence.signals as confidence_signals
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings

ESCALATION = escalation_settings.EscalationSettings(
    kind="switching",
    confidence_walk_microseconds=0.5,
    gap_threshold_nats=2.0,
)
WEAK = decoder_settings.DecoderSettings(kind="union_find", weight_step=0.1)


def test_the_shipped_rows_are_reachable_by_the_name_the_yaml_writes():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    assert sorted(rows) == [
        "cluster_gap",
        "complementary_gap",
        "extra_cluster_gap",
    ]


def test_every_row_builds_from_the_two_settings_and_reads_the_card():
    """A row written outside decsim reaches the root through this call."""
    for name, row in confidence_signals.CONFIDENCE_SIGNALS.items():
        built = row.from_settings(ESCALATION, WEAK)

        assert built.walk_microseconds == 0.5, name


def test_every_row_declares_the_classes_the_window_must_be_solved_in():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    complementary = rows["complementary_gap"].from_settings(ESCALATION, WEAK)
    cluster = rows["cluster_gap"].from_settings(ESCALATION, WEAK)
    extra = rows["extra_cluster_gap"].from_settings(ESCALATION, WEAK)

    assert complementary.forced_logical_classes == (0, 1)
    assert cluster.forced_logical_classes == ()
    assert extra.forced_logical_classes == ()


def test_every_row_declares_the_evidence_it_needs_from_the_decode():
    for name, row in confidence_signals.CONFIDENCE_SIGNALS.items():
        assert row.decoder_evidence_requirement is not None, name
        assert row.evidence_refusal, name


def test_every_row_names_its_own_source():
    """The threshold is held in one unit, so a source says which signal."""
    rows = confidence_signals.CONFIDENCE_SIGNALS
    methods = set()

    for row in rows.values():
        built = row.from_settings(ESCALATION, WEAK)
        methods.add(built.source.method)

    assert methods == {"complementary_gap", "cluster_gap", "extra_cluster_gap"}
