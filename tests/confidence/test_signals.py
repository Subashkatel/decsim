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

# the three rows decsim ships, by the name each record carries
ROWS = {
    "complementary_gap": complementary.ComplementaryGap,
    "cluster_gap": cluster.ClusterGap,
    "extra_cluster_gap": extra_cluster.ExtraClusterGap,
}


@pytest.mark.parametrize("name", sorted(ROWS))
def test_every_row_built_from_its_record_keeps_the_records_walk_card(name):
    """The lock prices only the cluster gap on a card, so this guards the rest.

    The default card is None; 0.5 microseconds must reach the row whole.
    """
    row = ROWS[name]
    host_time = cycle_count_module.HostMeasuredTime()
    weak = union_find.UnionFindDecoder.Settings(timing=host_time)
    record = row.Settings(walk_microseconds=0.5)

    built = record.build(weak, 2.0)

    assert built.walk_microseconds == 0.5


@pytest.mark.parametrize("name", sorted(ROWS))
def test_every_rows_record_refuses_a_negative_walk_card(name):
    """A negative walk would shorten the unit's hold without a stop."""
    row = ROWS[name]

    with pytest.raises(ValueError):
        row.Settings(walk_microseconds=-1.0)
