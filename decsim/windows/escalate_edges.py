"""A run's escalate edges, as the window side looks them up.

Each edge carries one pool's decodes of a window to another pool, with
its own verdict, costs, shape and held windows (ports.EscalateEdge). A
pool leaves by at most one edge, so a window climbs one chain from the
entry pool the router names for it (routers.py); several chains may
share a target, one strong pool serving every weak pool (Delfosse et al.
2001.11427 Sec. V; Toshio et al. 2510.25222 Sec. III C). The window side
asks for the edge that leaves the pool a result came from, to run its
verdict, and the edge on the window's chain that enters a pool, to find
its rules. A run with no switching has no edge, and every result is
final where it was decoded. Edges are kept in graph order, which is the
order every edge hears a commit or a stored round.
"""

from typing import Optional

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.windows.routers as routers


class EscalateEdges:
    """Every escalate edge of a run, by source pool and along each chain."""

    def __init__(self, edges: tuple = ()) -> None:
        self.edges = edges
        self.by_source: dict = {}
        for edge in edges:
            self.by_source[edge.source] = edge

    def leaving(
        self, pool: window_records.PoolId
    ) -> Optional[ports.EscalateEdge]:
        """The edge whose verdict a result of pool meets; None if final."""
        return self.by_source.get(pool)

    def chain_of(self, entry: window_records.PoolId) -> tuple:
        """The edges a window decoded first on entry may climb, in order."""
        chain = []
        edge = self.leaving(entry)
        while edge is not None:
            chain.append(edge)
            edge = self.leaving(edge.target)
        return tuple(chain)

    def entering(
        self, pool: window_records.PoolId, entry: window_records.PoolId
    ) -> ports.EscalateEdge:
        """The edge on entry's chain that asks pool for its decodes."""
        for edge in self.chain_of(entry):
            if edge.target == pool:
                return edge
        raise RuntimeError(
            f"no escalate edge on the chain from {entry.word} reaches "
            f"{pool.word}"
        )

    def target_queues(self) -> tuple:
        """Each target pool's manager once, in graph order."""
        queues = []
        for edge in self.edges:
            if edge.decode_queue not in queues:
                queues.append(edge.decode_queue)
        return tuple(queues)

    def queue_of(self, pool: window_records.PoolId) -> ports.DecodeQueue:
        """The manager of a pool some edge targets; every edge in shares it."""
        for edge in self.edges:
            if edge.target == pool:
                return edge.decode_queue
        raise RuntimeError(f"no escalate edge targets {pool.word}")


class StrongResultLearner:
    """Teaches a target result to the policy of the edge that asked for it.

    A target pool's manager hears every strong result it serves; the
    edge into that pool on the window's chain escalated it, so its
    policy learns it (an online threshold audits its own escalations).
    """

    def __init__(
        self,
        edges: EscalateEdges,
        router: routers.Router,
        target: window_records.PoolId,
    ) -> None:
        self.edges = edges
        self.router = router
        self.target = target

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """The escalating edge's policy hears the strong result."""
        stream_id, _window_index = window_key
        entry = self.router.pool_of_stream(stream_id)
        edge = self.edges.entering(self.target, entry)
        edge.policy.learn_from_strong_result(window_key, result)
