"""The soft output rows a switching run's weak decoder can report.

escalation.confidence names one row. A row says what evidence it needs
from the decode and what classes the window must be decoded in; the
switching policy expects its source, and the window side asks the weak
decoder for exactly those solves. A run holds the row's Settings record,
which builds the row from the weak decoder and the threshold, so the
root asks nothing about which row it holds.
"""

from typing import Optional

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.confidence.extra_cluster as extra_cluster
import decsim.tables as tables

CONFIDENCE_SIGNALS = {
    "complementary_gap": complementary.ComplementaryGap,
    "cluster_gap": cluster.ClusterGap,
    "extra_cluster_gap": extra_cluster.ExtraClusterGap,
}


def confidence_settings(name: str, walk_microseconds: Optional[float]):
    """The Settings record of the row escalation.confidence names.

    The record carries the card that prices the row's own computation.
    """
    row = tables.row(CONFIDENCE_SIGNALS, "escalation.confidence", name)
    return row.Settings(walk_microseconds=walk_microseconds)
