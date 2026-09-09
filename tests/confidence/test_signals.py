"""The soft-output rows a switching run's weak decoder can report.

CONFIDENCE_SIGNALS is the table escalation.confidence names a row of,
and every row is built by one call from one argument, the card that
prices its own walk on the weak unit (I6 slice 5a). Each row declares
what evidence it needs from the decode and which logical classes the
window must be decoded in, so the window side asks the weak decoder for
exactly those solves and never for the row's class.
"""

import decsim.confidence.signals as confidence_signals


def test_both_shipped_rows_are_reachable_by_the_name_the_yaml_writes():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    assert sorted(rows) == ["cluster_gap", "complementary_gap"]


def test_every_row_takes_the_one_walk_card_and_nothing_else():
    """A row written outside decsim reaches the root through this call."""
    for name, row in confidence_signals.CONFIDENCE_SIGNALS.items():
        built = row(walk_microseconds=0.5)

        assert built.walk_microseconds == 0.5, name


def test_every_row_declares_the_classes_the_window_must_be_solved_in():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    complementary = rows["complementary_gap"](walk_microseconds=None)
    cluster = rows["cluster_gap"](walk_microseconds=None)

    assert complementary.forced_logical_classes == (0, 1)
    assert cluster.forced_logical_classes == ()


def test_every_row_declares_the_evidence_it_needs_from_the_decode():
    for name, row in confidence_signals.CONFIDENCE_SIGNALS.items():
        built = row(walk_microseconds=None)

        assert built.decoder_evidence_requirement is not None, name
        assert built.evidence_refusal, name


def test_every_row_names_its_own_source():
    """The threshold is held in one unit, so a source says which signal."""
    rows = confidence_signals.CONFIDENCE_SIGNALS
    methods = set()

    for row in rows.values():
        built = row(walk_microseconds=None)
        methods.add(built.source.method)

    assert methods == {"complementary_gap", "cluster_gap"}
