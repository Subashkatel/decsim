"""A run's escalate edges, as the window side looks them up.

Each edge carries one pool's decodes of a window to another pool, with
its own verdict, costs, shape and held windows (ports.EscalateEdge). The
window side asks for the edge that leaves the pool a result came from,
to run its verdict, and the edge that enters it, to find the manager
that serves it. A run with no switching has no edge, and every result is
final where it was decoded. Edges are kept in graph order, which is the
order every edge hears a commit or a stored round.
"""

from typing import Optional

import decsim.ports as ports
import decsim.records.windows as window_records


class EscalateEdges:
    """Every escalate edge of a run, by source pool and by target pool."""

    def __init__(self, edges: tuple = ()) -> None:
        self.edges = edges
        self.by_source: dict = {}
        self.by_target: dict = {}
        for edge in edges:
            self.by_source[edge.source] = edge
            self.by_target[edge.target] = edge

    def leaving(
        self, pool: window_records.PoolId
    ) -> Optional[ports.EscalateEdge]:
        """The edge whose verdict a result of pool meets; None if final."""
        return self.by_source.get(pool)

    def entering(
        self, pool: window_records.PoolId
    ) -> Optional[ports.EscalateEdge]:
        """The edge that asks pool for its decodes; None for the entry."""
        return self.by_target.get(pool)
