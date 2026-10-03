"""The soft-output rows a switching run's weak decoder can report.

Every row's Settings record builds the row from the weak decoder's own
settings and the threshold, so a row written outside decsim reaches the
root through that one call. Each row declares what evidence it needs
from the decode and which logical classes the window must be decoded
in, so the window side asks the weak decoder for exactly those solves
and never for the row's class.
"""

import pytest

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.confidence.extra_cluster as extra_cluster
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find

HOST_TIME = cycle_count_module.HostMeasuredTime()
UNION_FIND = union_find.UnionFindDecoder.Settings(timing=HOST_TIME)
THRESHOLD_NATS = 2.0
# the three rows decsim ships, by the name each record carries
ROWS = {
    "complementary_gap": complementary.ComplementaryGap,
    "cluster_gap": cluster.ClusterGap,
    "extra_cluster_gap": extra_cluster.ExtraClusterGap,
}


def _built(row):
    """The row built from its record, priced by a half-microsecond card."""
    settings = row.Settings(walk_microseconds=0.5)
    return settings.build(UNION_FIND, THRESHOLD_NATS)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_every_row_builds_from_its_record_and_reads_the_card(name):
    """A row written outside decsim reaches the root through this call."""
    row = ROWS[name]
    built = _built(row)

    assert built.walk_microseconds == 0.5
    assert row.Settings.name == name


def test_every_row_declares_the_classes_the_window_must_be_solved_in():
    complementary_gap = _built(ROWS["complementary_gap"])
    cluster_gap = _built(ROWS["cluster_gap"])
    extra = _built(ROWS["extra_cluster_gap"])

    assert complementary_gap.forced_logical_classes == (0, 1)
    assert cluster_gap.forced_logical_classes == ()
    assert extra.forced_logical_classes == ()


@pytest.mark.parametrize("name", sorted(ROWS))
def test_every_row_declares_the_evidence_it_needs_from_the_decode(name):
    row = ROWS[name]

    assert row.decoder_evidence_requirement is not None
    assert row.evidence_refusal


def test_every_row_names_its_own_source():
    """The threshold is held in one unit, so a source says which signal."""
    built_rows = [_built(row) for row in ROWS.values()]
    methods = {built.source.method for built in built_rows}

    assert methods == {"complementary_gap", "cluster_gap", "extra_cluster_gap"}


@pytest.mark.parametrize("name", sorted(ROWS))
@pytest.mark.parametrize("walk", [-1.0, float("nan"), "fast"])
def test_every_rows_record_refuses_a_walk_card_that_is_no_duration(name, walk):
    row = ROWS[name]
    sentence = "walk_microseconds must be finite and not negative"

    with pytest.raises(ValueError, match=sentence):
        row.Settings(walk_microseconds=walk)
