"""The soft output rows a switching run's weak decoder can report.

escalation.confidence names one row. A row says what evidence it needs
from the decode and what classes the window must be decoded in; the
switching policy expects its source, and the window side asks the weak
decoder for exactly those solves. Every row builds itself from the
escalation section and the weak decoder's settings (from_settings), the
way sinter's decoders build from their own names, so the root asks
nothing about which row it holds.
"""

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.confidence.extra_cluster as extra_cluster

CONFIDENCE_SIGNALS = {
    "complementary_gap": complementary.ComplementaryGap,
    "cluster_gap": cluster.ClusterGap,
    "extra_cluster_gap": extra_cluster.ExtraClusterGap,
}
