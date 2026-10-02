"""The soft-output rows a switching run's weak decoder can report.

CONFIDENCE_SIGNALS is the table escalation.confidence names a row of,
and every row's Settings record builds the row from the weak decoder's
own settings and the threshold, so a row written outside decsim
reaches the root through that one call. Each row declares what evidence
it needs from the decode and which logical classes the window must be
decoded in, so the window side asks the weak decoder for exactly those
solves and never for the row's class.
"""

import pytest

import decsim.confidence.signals as confidence_signals
import decsim.decoders.union_find.decoder as union_find

UNION_FIND = union_find.UnionFindDecoder.Settings()
THRESHOLD_NATS = 2.0


def _built(row):
    """The row built from its record, priced by a half-microsecond card."""
    settings = row.Settings(walk_microseconds=0.5)
    return settings.build(UNION_FIND, THRESHOLD_NATS)


def test_the_shipped_rows_are_reachable_by_the_name_the_yaml_writes():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    assert sorted(rows) == [
        "cluster_gap",
        "complementary_gap",
        "extra_cluster_gap",
    ]


@pytest.mark.parametrize("name", sorted(confidence_signals.CONFIDENCE_SIGNALS))
def test_every_row_builds_from_its_record_and_reads_the_card(name):
    """A row written outside decsim reaches the root through this call."""
    row = confidence_signals.CONFIDENCE_SIGNALS[name]
    built = _built(row)

    assert built.walk_microseconds == 0.5
    assert row.Settings.name == name


def test_every_row_declares_the_classes_the_window_must_be_solved_in():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    complementary = _built(rows["complementary_gap"])
    cluster = _built(rows["cluster_gap"])
    extra = _built(rows["extra_cluster_gap"])

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
    built_rows = [_built(row) for row in rows.values()]
    methods = {built.source.method for built in built_rows}

    assert methods == {"complementary_gap", "cluster_gap", "extra_cluster_gap"}


@pytest.mark.parametrize("name", sorted(confidence_signals.CONFIDENCE_SIGNALS))
@pytest.mark.parametrize("walk", [-1.0, float("nan"), "fast"])
def test_every_rows_record_refuses_a_walk_card_that_is_no_duration(name, walk):
    row = confidence_signals.CONFIDENCE_SIGNALS[name]
    sentence = "walk_microseconds must be finite and not negative"

    with pytest.raises(ValueError, match=sentence):
        row.Settings(walk_microseconds=walk)
