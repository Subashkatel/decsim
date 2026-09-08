"""The soft output rows a switching run's weak decoder can report.

escalation.confidence names one row. A row says what evidence it needs
from the decode and what classes the window must be decoded in; the
switching policy expects its source, and the window side asks the weak
decoder for exactly those solves.
"""

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary

CONFIDENCE_SIGNALS = {
    "complementary_gap": complementary.ComplementaryGap,
    "cluster_gap": cluster.ClusterGap,
}
