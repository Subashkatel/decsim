"""Cluster-gap soft output: grown-ball Union-Find quotient logical distance."""
from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

from ..message import SoftOutput

if TYPE_CHECKING:
    import stim
    from ..adapters.window_error_models import WindowErrorModel

_CITATION = "Meister/Pattison/Preskill arXiv:2405.07433 Def. 9, Alg. 2; Thm. 13"

BOUNDARY = -1
_EPS = 1e-9


def _edge_pieces(dets, obs, weight):
    """One DEM error component -> a graph edge tuple ``(u, v, weight, obs_parity)``."""
    if not dets and not obs:
        return None
    parity = len(obs) % 2
    if len(dets) == 1:
        return (dets[0], BOUNDARY, weight, parity)
    if len(dets) == 2:
        return (dets[0], dets[1], weight, parity)
    return None   # hyperedge (>2) -- not graphlike; skip (DEM should be decomposed)


def dem_to_graph(dem: "stim.DetectorErrorModel"):
    """Build the weighted decoding graph (edge list) from a graphlike DEM."""
    import numpy as np

    edges = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        prob = instr.args_copy()[0]
        weight = float(np.log((1 - prob) / prob)) if 0 < prob < 1 else 50.0
        dets: list = []
        obs: list = []
        for target in instr.targets_copy():
            if target.is_separator():
                edge = _edge_pieces(dets, obs, weight)
                if edge is not None:
                    edges.append(edge)
                dets, obs = [], []
            elif target.is_relative_detector_id():
                dets.append(target.val)
            elif target.is_logical_observable_id():
                obs.append(target.val)
        edge = _edge_pieces(dets, obs, weight)
        if edge is not None:
            edges.append(edge)
    return edges


def edges_from_matrices(check, obs, weights):
    """Build the decoding graph edge list from check/obs matrices + per-fault weights."""
    import numpy as np

    check = np.asarray(check, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    weights = np.asarray(weights, dtype=float)
    edges = []
    for j in range(check.shape[1]):
        dets = list(np.nonzero(check[:, j])[0])
        parity = int(obs[:, j].sum() % 2) if obs.shape[0] else 0
        if len(dets) == 1:
            edges.append((int(dets[0]), BOUNDARY, float(weights[j]), parity))
        elif len(dets) == 2:
            edges.append((int(dets[0]), int(dets[1]), float(weights[j]), parity))
    return edges


def union_find_clusters(edges, syndrome_set, n_det, mode="simultaneous"):
    """Grow odd clusters as metric balls; returns ``(cluster_of, fill, dual_mass)``.

    ``mode="simultaneous"`` advances all odd clusters at unit rate (LP-feasible dual,
    robust); ``mode="smallest"`` grows only the smallest odd cluster per step. ``fill[e]``
    is the covered length per edge; ``dual_mass`` is the UF cluster-dual ``D_UF``, a
    feasible MWPM-LP dual with ``D_UF <= w_min``.
    """
    nodes = list(range(n_det)) + [BOUNDARY]
    parent = {v: v for v in nodes}
    parity = {v: (1 if v in syndrome_set else 0) for v in nodes}
    parity[BOUNDARY] = 0
    touches_boundary = {v: (v == BOUNDARY) for v in nodes}
    # incident edge indices per node (for frontier scans)
    incident: dict = {v: [] for v in nodes}
    for idx, (u, v, _w, _o) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def valid(root):
        return touches_boundary[root] or (parity[root] % 2 == 0)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[rb] = ra
        parity[ra] = (parity[ra] + parity[rb]) % 2
        touches_boundary[ra] = touches_boundary[ra] or touches_boundary[rb]

    fill = [0.0] * len(edges)
    dual_mass = 0.0

    if mode == "simultaneous":
        while True:
            root_of = {v: find(v) for v in nodes}
            invalid = {r for r in root_of.values() if not valid(r)}
            if not invalid:
                break
            best = None
            for idx, (u, v, w, _o) in enumerate(edges):
                ru, rv = root_of[u], root_of[v]
                if ru == rv:
                    continue
                growers = (ru in invalid) + (rv in invalid)
                if growers == 0:
                    continue
                need = (w - fill[idx]) / growers
                if best is None or need < best:
                    best = need
            if best is None:
                break
            if best < 0.0:
                best = 0.0
            dual_mass += best * len(invalid)   # each odd cluster S grows y_S += best
            for idx, (u, v, w, _o) in enumerate(edges):
                ru, rv = root_of[u], root_of[v]
                if ru == rv:
                    continue
                growers = (ru in invalid) + (rv in invalid)
                if growers:
                    nf = fill[idx] + best * growers
                    fill[idx] = w if nf > w else nf
                if fill[idx] >= w - _EPS:
                    union(u, v)
        return {v: find(v) for v in nodes}, fill, dual_mass

    # mode == "smallest": grow the smallest odd cluster (fewest frontier edges) to completion
    members: dict = {v: [v] for v in nodes}

    def frontier_of(r):
        front = []
        seen = set()
        for v in members[r]:
            for idx in incident[v]:
                if idx in seen:
                    continue
                a, b = edges[idx][0], edges[idx][1]
                if find(a) != find(b):
                    front.append(idx)
                    seen.add(idx)
        return front

    while True:
        roots = {find(v) for v in nodes}
        invalid = [r for r in roots if not valid(r)]
        if not invalid:
            break
        # pick the smallest odd cluster by frontier size; tie-break by root id for determinism
        best_root = None
        best_front = None
        for r in sorted(invalid):
            front = frontier_of(r)
            if best_front is None or len(front) < len(best_front):
                best_root, best_front = r, front
        if not best_front:
            break  # isolated odd cluster, no growth target
        # delta to complete this cluster's nearest frontier edge (only this side grows)
        delta = max(0.0, min((edges[idx][2] - fill[idx]) for idx in best_front))
        dual_mass += delta   # the single grown odd set S gets y_S += delta
        completed = []
        for idx in best_front:
            w = edges[idx][2]
            nf = fill[idx] + delta
            fill[idx] = w if nf > w else nf
            if fill[idx] >= w - _EPS:
                completed.append(idx)
        for idx in completed:
            u, v = edges[idx][0], edges[idx][1]
            ru, rv = find(u), find(v)
            if ru != rv:
                union(u, v)   # union() makes ru = find(u) the surviving root
                members[ru].extend(members[rv])
                members[rv] = []

    cluster_of = {v: find(v) for v in nodes}
    return cluster_of, fill, dual_mass


def quotient_weight(edges, cluster_of, fill):
    """Grown-ball quotient weights: 0 inside a cluster, else uncovered remainder ``w_e - fill[e]``."""
    qw = []
    for idx, (u, v, w, _o) in enumerate(edges):
        if cluster_of[u] == cluster_of[v]:
            qw.append(0.0)
        else:
            qw.append(max(0.0, w - fill[idx]))
    return qw


def quotient_logical_distance(edges, qweights):
    """Shortest odd-observable path b1->b2 in the quotient graph via parity-doubled Dijkstra."""
    adjacency: dict = {}

    def link(a, b, w):
        adjacency.setdefault(a, []).append((b, w))

    for idx, (u, v, _w, o) in enumerate(edges):
        qw = qweights[idx]
        for p in (0, 1):
            link((u, p), (v, p ^ o), qw)
            link((v, p ^ o), (u, p), qw)

    source = (BOUNDARY, 0)
    target = (BOUNDARY, 1)
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == target:
            return d
        if d > dist.get(node, math.inf):
            continue
        for nb, w in adjacency.get(node, ()):  # noqa: E741
            nd = d + w
            if nd < dist.get(nb, math.inf):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return dist.get(target, math.inf)


class ClusterGapMetric:
    """Cluster-gap soft output for a single-observable window (SoftOutputMetric seam)."""

    name = "cluster_gap"

    def __init__(self, edges, n_det, matching):
        self.edges = edges
        self.n_det = int(n_det)
        self._matching = matching

    @classmethod
    def from_dem(cls, dem: "stim.DetectorErrorModel") -> "ClusterGapMetric":
        import numpy as np
        import pymatching

        edges = dem_to_graph(dem)
        n_det = dem.num_detectors
        # build the matching from the SAME edge list (via from_check_matrix) so w_min,
        # phi, and D_UF all live on one graph and the duality-gap correction is meaningful
        ncol = len(edges)
        check = np.zeros((n_det, ncol), dtype=np.uint8)
        obs = np.zeros((1, ncol), dtype=np.uint8)
        weights = np.empty(ncol, dtype=float)
        for j, (u, v, w, par) in enumerate(edges):
            check[u, j] = 1
            if v != BOUNDARY:
                check[v, j] = 1
            obs[0, j] = par
            weights[j] = w
        matching = pymatching.Matching.from_check_matrix(
            check, weights=weights, faults_matrix=obs)
        return cls(edges, n_det, matching)

    @classmethod
    def from_window_model(cls, model: "WindowErrorModel") -> "ClusterGapMetric":
        import numpy as np
        import pymatching

        weights = np.log((1.0 - np.asarray(model.priors)) / np.asarray(model.priors))
        edges = edges_from_matrices(model.check, model.obs, weights)
        matching = pymatching.Matching.from_check_matrix(
            model.check, weights=weights, faults_matrix=model.obs)
        return cls(edges, np.asarray(model.check).shape[0], matching)

    def _gap(self, syndrome_set, mode="simultaneous"):
        cluster_of, fill, dual_mass = union_find_clusters(
            self.edges, syndrome_set, self.n_det, mode=mode)
        qw = quotient_weight(self.edges, cluster_of, fill)
        gap = quotient_logical_distance(self.edges, qw)
        return gap, cluster_of, fill, qw, dual_mass

    def evaluate(self, syndrome, robust: bool = False) -> SoftOutput:
        """Soft output for one syndrome; ``robust=True`` subtracts the UF duality gap ``max(0, w_min - D_UF)``."""
        import numpy as np

        bits = np.asarray(syndrome, dtype=np.uint8).ravel()
        syndrome_set = {int(i) for i in np.nonzero(bits)[0]}
        phi, _cluster_of, _fill, _qw, dual_mass = self._gap(syndrome_set)
        prediction, w_min = self._matching.decode(bits, return_weight=True)
        pred = int(prediction[0]) if np.size(prediction) else 0
        gap = phi
        if robust:
            gap = phi - max(0.0, float(w_min) - dual_mass)
        return SoftOutput(logical_value=pred, gap=float(max(0.0, gap)),
                          w_min=float(w_min), w_comp=float(dual_mass))

    def trace(self, syndrome) -> dict:
        """Diagnostic dump for one syndrome: clusters, zeroed/frontier edges, g_cluster."""
        import numpy as np

        bits = np.asarray(syndrome, dtype=np.uint8).ravel()
        syndrome_set = {int(i) for i in np.nonzero(bits)[0]}
        gap, cluster_of, fill, qw, dual_mass = self._gap(syndrome_set)
        # group nodes by cluster (only clusters with >1 node or a syndrome vertex)
        clusters: dict = {}
        for node, root in cluster_of.items():
            clusters.setdefault(root, []).append(node)
        nontrivial = {r: ns for r, ns in clusters.items()
                      if len(ns) > 1 or any(n in syndrome_set for n in ns)}
        zeroed = [edges_i for edges_i, e in enumerate(self.edges)
                  if cluster_of[e[0]] == cluster_of[e[1]]]
        frontier = [edges_i for edges_i, e in enumerate(self.edges)
                    if cluster_of[e[0]] != cluster_of[e[1]] and fill[edges_i] > _EPS]
        return {
            "syndrome_set": syndrome_set,
            "clusters": nontrivial,
            "n_zeroed_edges": len(zeroed),
            "n_frontier_partial": len(frontier),
            "g_cluster": float(gap),
            "dual_mass": float(dual_mass),
            "edges": self.edges,
            "fill": fill,
            "qw": qw,
            "cluster_of": cluster_of,
        }
