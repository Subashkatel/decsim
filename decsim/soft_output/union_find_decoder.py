"""Uniform Union-Find decoding with confidence from the decoder's final balls.

The hard correction follows Delfosse--Nickerson's half-edge growth and
peeling construction. Confidence follows Meister--Pattison--Preskill's
cluster-gap quotient using the radii produced by that same decode.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Optional

from ..adapters.window_decode_results import (
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
    SoftOutput,
    SoftOutputSource,
)


BOUNDARY = -1
_EPSILON = 1e-12

UNION_FIND_CLUSTER_GAP_SOURCE = SoftOutputSource(
    method="cluster_gap",
    cluster_origin="union_find_decoder",
    growth_schedule="meister_uniform_fair",
    gap_units="graph_edges",
    correction="none",
    references=(
        "arXiv:1709.06218v3 Algorithm 1",
        "arXiv:2405.07433v2 Definition 9 / Algorithm 2",
    ),
)


@dataclass(frozen=True)
class UnionFindEdge:
    """One graphlike fault column in the unit decoding graph."""

    fault_index: int
    detector_a: int
    detector_b: int
    logical_parity: int


@dataclass(frozen=True)
class UnionFindGraph:
    """Immutable graph state cached independently of any syndrome."""

    detector_count: int
    fault_count: int
    edges: tuple[UnionFindEdge, ...]
    adjacency: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True)
class UnionFindOutcome:
    """Hard correction and actual-cluster confidence from one decode."""

    selected_faults: tuple[int, ...]
    completed_growth_edges: tuple[int, ...]
    cluster_root_by_detector: tuple[tuple[int, int], ...]
    radius_by_syndrome_center: tuple[tuple[int, float], ...]
    covered_edge_intervals: tuple[
        tuple[tuple[int, float, float], ...],
        ...,
    ]
    erasure_forest_edges: tuple[int, ...]
    cluster_gap: float


class _DisjointSet:
    """Per-decode cluster state for completed half-edge growth."""

    def __init__(self, detector_count: int, syndrome) -> None:
        boundary_node = detector_count
        node_count = detector_count + 1
        self.boundary_node = boundary_node
        self.parent = list(range(node_count))
        self.detectors = [
            ({node} if node < detector_count else set())
            for node in range(node_count)
        ]
        self.centers = [
            ({node} if node < detector_count and syndrome[node] else set())
            for node in range(node_count)
        ]
        self.parity = [
            (int(syndrome[node]) if node < detector_count else 0)
            for node in range(node_count)
        ]
        self.touches_boundary = [
            node == boundary_node for node in range(node_count)
        ]

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        survivor, absorbed = sorted((left_root, right_root))
        self.parent[absorbed] = survivor
        self.detectors[survivor].update(self.detectors[absorbed])
        self.centers[survivor].update(self.centers[absorbed])
        self.parity[survivor] ^= self.parity[absorbed]
        self.touches_boundary[survivor] = (
            self.touches_boundary[survivor]
            or self.touches_boundary[absorbed]
        )
        return survivor

    def is_valid(self, root: int) -> bool:
        root = self.find(root)
        return self.touches_boundary[root] or self.parity[root] == 0


def _graph_from_model(model, *, location: str) -> UnionFindGraph:
    import numpy as np

    from ..detector_error_model import validate_graphlike_matrices

    raw_check = np.asarray(model.check)
    raw_observables = np.asarray(model.obs)
    validate_graphlike_matrices(
        raw_check,
        raw_observables,
        location=location,
    )
    check = raw_check.astype(np.uint8, copy=False)
    observables = raw_observables.astype(np.uint8, copy=False)
    if observables.shape[0] != 1:
        raise ValueError(
            f"{location} cluster confidence requires exactly one logical "
            f"observable, got {observables.shape[0]}"
        )

    edges = []
    adjacency: dict[int, list[tuple[int, int]]] = {
        node: [] for node in range(check.shape[0])
    }
    adjacency[BOUNDARY] = []
    for fault_index in range(check.shape[1]):
        detectors = tuple(
            int(value) for value in np.nonzero(check[:, fault_index])[0]
        )
        if not detectors:
            continue
        detector_a = detectors[0]
        detector_b = detectors[1] if len(detectors) == 2 else BOUNDARY
        edge = UnionFindEdge(
            fault_index=fault_index,
            detector_a=detector_a,
            detector_b=detector_b,
            logical_parity=int(observables[:, fault_index].sum() % 2),
        )
        edge_index = len(edges)
        edges.append(edge)
        adjacency[detector_a].append((detector_b, edge_index))
        adjacency[detector_b].append((detector_a, edge_index))

    frozen_adjacency = tuple(
        (
            node,
            tuple(
                sorted(
                    neighbors,
                    key=lambda item: (
                        edges[item[1]].fault_index,
                        item[0],
                    ),
                )
            ),
        )
        for node, neighbors in sorted(adjacency.items())
    )
    return UnionFindGraph(
        detector_count=check.shape[0],
        fault_count=check.shape[1],
        edges=tuple(edges),
        adjacency=frozen_adjacency,
    )


def _graph_adjacency(graph: UnionFindGraph):
    return {node: neighbors for node, neighbors in graph.adjacency}


def _grow_one_fair_sweep(
    graph: UnionFindGraph,
    disjoint_set: _DisjointSet,
    radii: dict[int, float],
    edge_growth_units: list[int],
) -> Optional[int]:
    """Visit each still-odd snapshot cluster at most once."""
    detector_count = graph.detector_count
    boundary_node = detector_count

    def endpoint_node(detector: int) -> int:
        return boundary_node if detector == BOUNDARY else detector

    roots = {
        disjoint_set.find(node)
        for node in range(detector_count + 1)
    }
    invalid_roots = [
        root for root in roots if not disjoint_set.is_valid(root)
    ]
    if not invalid_roots:
        return None
    snapshots = sorted(
        (
            (
                len(disjoint_set.detectors[root]),
                min(disjoint_set.centers[root]),
                tuple(sorted(disjoint_set.centers[root])),
            )
            for root in invalid_roots
        ),
        key=lambda item: (item[0], item[1]),
    )
    grown_centers: set[int] = set()
    visit_count = 0

    for _size, anchor, _snapshot_centers in snapshots:
        root = disjoint_set.find(anchor)
        current_centers = set(disjoint_set.centers[root])
        if current_centers & grown_centers:
            continue
        if disjoint_set.is_valid(root):
            continue

        frontier = []
        for edge_index, edge in enumerate(graph.edges):
            left_root = disjoint_set.find(
                endpoint_node(edge.detector_a)
            )
            right_root = disjoint_set.find(
                endpoint_node(edge.detector_b)
            )
            if (
                left_root != right_root
                and root in (left_root, right_root)
                and edge_growth_units[edge_index] < 2
            ):
                frontier.append(edge_index)
        if not frontier:
            raise RuntimeError(
                "odd Union-Find cluster has no outward graph edge: "
                f"centers {tuple(sorted(current_centers))}"
            )

        for center in current_centers:
            radii[center] += 0.5
        grown_centers.update(current_centers)
        visit_count += 1

        completed_now = []
        for edge_index in frontier:
            edge_growth_units[edge_index] += 1
            if edge_growth_units[edge_index] == 2:
                completed_now.append(edge_index)
        for edge_index in sorted(
            completed_now,
            key=lambda index: graph.edges[index].fault_index,
        ):
            edge = graph.edges[edge_index]
            disjoint_set.union(
                endpoint_node(edge.detector_a),
                endpoint_node(edge.detector_b),
            )

    if visit_count == 0:
        raise RuntimeError(
            "odd Union-Find clusters remain but no cluster grew"
        )
    return visit_count


def _growth_outcome(graph: UnionFindGraph, syndrome):
    detector_count = graph.detector_count
    disjoint_set = _DisjointSet(detector_count, syndrome)
    edge_growth_units = [0] * len(graph.edges)
    radii = {
        detector: 0.0
        for detector, bit in enumerate(syndrome)
        if bit
    }
    maximum_visits = (
        4
        * max(1, detector_count + len(graph.edges))
        * max(1, len(radii))
    )
    visit_count = 0

    while True:
        visits_this_sweep = _grow_one_fair_sweep(
            graph,
            disjoint_set,
            radii,
            edge_growth_units,
        )
        if visits_this_sweep is None:
            break
        visit_count += visits_this_sweep
        if visit_count > maximum_visits:
            raise RuntimeError(
                "Union-Find growth exceeded its finite graph bound"
            )

    completed = tuple(
        edge_index
        for edge_index, units in enumerate(edge_growth_units)
        if units == 2
    )
    return disjoint_set, radii, completed


def _peel_completed_edges(
    graph: UnionFindGraph,
    syndrome,
    completed_edge_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    detector_count = graph.detector_count
    boundary_node = detector_count
    forest_set = _DisjointSet(
        detector_count,
        [0] * detector_count,
    )
    forest_edges = []

    def endpoint_node(detector: int) -> int:
        return boundary_node if detector == BOUNDARY else detector

    for edge_index in sorted(
        completed_edge_indices,
        key=lambda index: graph.edges[index].fault_index,
    ):
        edge = graph.edges[edge_index]
        left = endpoint_node(edge.detector_a)
        right = endpoint_node(edge.detector_b)
        if forest_set.find(left) == forest_set.find(right):
            continue
        forest_set.union(left, right)
        forest_edges.append(edge_index)

    adjacency = {
        node: [] for node in range(detector_count + 1)
    }
    for edge_index in forest_edges:
        edge = graph.edges[edge_index]
        left = endpoint_node(edge.detector_a)
        right = endpoint_node(edge.detector_b)
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    for neighbors in adjacency.values():
        neighbors.sort(
            key=lambda item: (
                graph.edges[item[1]].fault_index,
                item[0],
            )
        )

    selected_edges: set[int] = set()
    unseen = set(range(detector_count + 1))
    while unseen:
        component_start = min(unseen)
        component = set()
        pending = [component_start]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(neighbor for neighbor, _edge in adjacency[node])
        unseen.difference_update(component)
        if not any(
            node != boundary_node and syndrome[node]
            for node in component
        ):
            continue

        root = boundary_node if boundary_node in component else min(component)
        parents: dict[int, Optional[int]] = {root: None}
        parent_edges: dict[int, int] = {}
        order = []
        pending = [root]
        while pending:
            node = pending.pop()
            order.append(node)
            for neighbor, edge_index in reversed(adjacency[node]):
                if neighbor in parents:
                    continue
                parents[neighbor] = node
                parent_edges[neighbor] = edge_index
                pending.append(neighbor)
        residual = {
            node: (
                0 if node == boundary_node else int(syndrome[node])
            )
            for node in component
        }
        for node in reversed(order[1:]):
            if residual[node] == 0:
                continue
            edge_index = parent_edges[node]
            selected_edges.add(edge_index)
            parent = parents[node]
            assert parent is not None
            residual[parent] ^= 1
        if root != boundary_node and residual[root]:
            raise RuntimeError(
                f"Union-Find erasure retained odd root detector {root}"
            )

    return (
        tuple(sorted(selected_edges)),
        tuple(sorted(forest_edges)),
    )


def _endpoint_potentials(
    graph: UnionFindGraph,
    radii: dict[int, float],
):
    adjacency = _graph_adjacency(graph)
    potentials = {node: math.inf for node in adjacency}
    sources: dict[int, Optional[int]] = {
        node: None for node in adjacency
    }
    sequence = itertools.count()
    pending = []
    for center, radius in sorted(radii.items()):
        potentials[center] = -radius
        sources[center] = center
        heapq.heappush(
            pending,
            (-radius, center, next(sequence), center),
        )
    while pending:
        potential, source, _sequence_id, node = heapq.heappop(pending)
        if potential > potentials[node] + _EPSILON:
            continue
        if (
            math.isclose(
                potential,
                potentials[node],
                rel_tol=0.0,
                abs_tol=_EPSILON,
            )
            and sources[node] is not None
            and source > sources[node]
        ):
            continue
        for neighbor, _edge_index in adjacency[node]:
            candidate = potential + 1.0
            current = potentials[neighbor]
            current_source = sources[neighbor]
            if (
                candidate < current - _EPSILON
                or (
                    math.isclose(
                        candidate,
                        current,
                        rel_tol=0.0,
                        abs_tol=_EPSILON,
                    )
                    and (
                        current_source is None
                        or source < current_source
                    )
                )
            ):
                potentials[neighbor] = candidate
                sources[neighbor] = source
                heapq.heappush(
                    pending,
                    (candidate, source, next(sequence), neighbor),
                )
    return potentials, sources


def _covered_intervals(
    graph: UnionFindGraph,
    radii: dict[int, float],
    cluster_label_by_center: dict[int, int],
):
    potentials, sources = _endpoint_potentials(graph, radii)
    intervals_by_edge = []
    for edge in graph.edges:
        candidates = []
        left_potential = potentials[edge.detector_a]
        left_source = sources[edge.detector_a]
        if left_source is not None and left_potential <= _EPSILON:
            candidates.append(
                (
                    cluster_label_by_center[left_source],
                    0.0,
                    min(1.0, max(0.0, -left_potential)),
                )
            )
        right_potential = potentials[edge.detector_b]
        right_source = sources[edge.detector_b]
        if right_source is not None and right_potential <= _EPSILON:
            candidates.append(
                (
                    cluster_label_by_center[right_source],
                    max(0.0, min(1.0, 1.0 + right_potential)),
                    1.0,
                )
            )
        merged = []
        for label, lower, upper in sorted(
            candidates,
            key=lambda item: (item[1], item[2], item[0]),
        ):
            if (
                merged
                and label == merged[-1][0]
                and lower <= merged[-1][2] + _EPSILON
            ):
                merged[-1][2] = max(merged[-1][2], upper)
            else:
                merged.append([label, lower, upper])
        intervals_by_edge.append(
            tuple(
                (int(label), float(lower), float(upper))
                for label, lower, upper in merged
            )
        )
    return tuple(intervals_by_edge)


def _quotient_cluster_gap(
    graph: UnionFindGraph,
    intervals_by_edge,
) -> float:
    adjacency: dict[object, list[tuple[object, float, int]]] = {}

    def add_segment(left, right, weight: float, parity: int) -> None:
        adjacency.setdefault(left, []).append((right, weight, parity))
        adjacency.setdefault(right, []).append((left, weight, parity))

    for edge_index, edge in enumerate(graph.edges):
        intervals = intervals_by_edge[edge_index]
        breakpoints = {0.0, 1.0}
        for _label, lower, upper in intervals:
            breakpoints.add(lower)
            breakpoints.add(upper)
        coordinates = sorted(breakpoints)
        path_nodes: list[object] = [edge.detector_a]
        path_nodes.extend(
            ("union_find_edge", edge_index, split_index)
            for split_index in range(1, len(coordinates) - 1)
        )
        path_nodes.append(edge.detector_b)
        for segment_index, (lower, upper) in enumerate(
            zip(coordinates, coordinates[1:])
        ):
            midpoint = (lower + upper) / 2.0
            covered = any(
                interval_lower - _EPSILON
                <= midpoint
                <= interval_upper + _EPSILON
                for _label, interval_lower, interval_upper in intervals
            )
            add_segment(
                path_nodes[segment_index],
                path_nodes[segment_index + 1],
                0.0 if covered else upper - lower,
                edge.logical_parity if segment_index == 0 else 0,
            )

    best = math.inf
    sequence = itertools.count()
    for reference_node in adjacency:
        source_state = (reference_node, 0)
        target_state = (reference_node, 1)
        distances = {source_state: 0.0}
        pending = [(0.0, next(sequence), source_state)]
        while pending:
            distance, _sequence_id, state = heapq.heappop(pending)
            if distance > distances.get(state, math.inf) + _EPSILON:
                continue
            if distance >= best - _EPSILON:
                break
            if state == target_state:
                best = distance
                break
            graph_node, logical_parity = state
            for neighbor, weight, edge_parity in adjacency.get(
                graph_node,
                (),
            ):
                neighbor_state = (
                    neighbor,
                    logical_parity ^ edge_parity,
                )
                candidate = distance + weight
                if candidate + _EPSILON < distances.get(
                    neighbor_state,
                    math.inf,
                ):
                    distances[neighbor_state] = candidate
                    heapq.heappush(
                        pending,
                        (candidate, next(sequence), neighbor_state),
                    )
    return best


def _decode_graph(
    graph: UnionFindGraph,
    syndrome,
) -> UnionFindOutcome:
    import numpy as np

    raw_syndrome = np.asarray(syndrome)
    if (
        raw_syndrome.ndim != 1
        or raw_syndrome.size != graph.detector_count
    ):
        raise ValueError(
            "Union-Find syndrome must be a one-dimensional detector vector "
            f"of length {graph.detector_count}"
        )
    if not np.all((raw_syndrome == 0) | (raw_syndrome == 1)):
        raise ValueError("Union-Find syndrome must contain only binary values")
    syndrome = raw_syndrome.astype(np.uint8, copy=False)

    disjoint_set, radii, completed = _growth_outcome(graph, syndrome)
    selected_edges, forest = _peel_completed_edges(
        graph,
        syndrome,
        completed,
    )
    selected_faults = [0] * graph.fault_count
    for edge_index in selected_edges:
        selected_faults[graph.edges[edge_index].fault_index] = 1

    reproduced = np.zeros(graph.detector_count, dtype=np.uint8)
    for edge_index in selected_edges:
        edge = graph.edges[edge_index]
        reproduced[edge.detector_a] ^= 1
        if edge.detector_b != BOUNDARY:
            reproduced[edge.detector_b] ^= 1
    if not np.array_equal(reproduced, syndrome):
        unmatched = tuple(
            int(value)
            for value in np.nonzero(reproduced ^ syndrome)[0]
        )
        raise RuntimeError(
            "Union-Find peeling correction does not reproduce the syndrome; "
            f"unmatched detectors {unmatched}"
        )

    cluster_label_by_center = {}
    for center in radii:
        root = disjoint_set.find(center)
        cluster_label_by_center[center] = min(
            disjoint_set.centers[root]
        )
    intervals = _covered_intervals(
        graph,
        radii,
        cluster_label_by_center,
    )
    return UnionFindOutcome(
        selected_faults=tuple(selected_faults),
        completed_growth_edges=tuple(
            graph.edges[index].fault_index for index in completed
        ),
        cluster_root_by_detector=tuple(
            (
                detector,
                min(disjoint_set.detectors[disjoint_set.find(detector)])
                if disjoint_set.detectors[disjoint_set.find(detector)]
                else BOUNDARY,
            )
            for detector in range(graph.detector_count)
        ),
        radius_by_syndrome_center=tuple(sorted(radii.items())),
        covered_edge_intervals=intervals,
        erasure_forest_edges=tuple(
            graph.edges[index].fault_index for index in forest
        ),
        cluster_gap=_quotient_cluster_gap(graph, intervals),
    )


def decode_union_find_model(model, syndrome) -> UnionFindOutcome:
    """Decode one placed unit-geometry model and return inspectable evidence."""
    graph = _graph_from_model(
        model,
        location="Union-Find window model",
    )
    return _decode_graph(graph, syndrome)


class UnionFindDecoder:
    """Real uniform Union-Find decoder with actual-cluster confidence."""

    def __init__(self, latency_model) -> None:
        self.latency_model = latency_model
        self._graphs: dict = {}

    def run_manifest_config(self):
        return {
            "algorithm": "union_find",
            "growth_schedule": "meister_uniform_fair",
            "edge_geometry": "unit_graph_edges",
            "cluster_origin": "union_find_decoder",
            "correction": "completed_growth_edge_peeling",
            "confidence_method": "cluster_gap",
            "gap_units": "graph_edges",
            "graph_domain": "one_or_two_detectors",
            "confidence_observable_count": 1,
            "logical_search": "global_odd_parity_closed_walk",
        }

    def run_seed_children(self):
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "latency_model"),),
                self.latency_model,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        return self.latency_model.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, model)
        graph = self._graph_for_model(model, job.label)
        outcome = _decode_graph(graph, syndrome)
        result = result_from_selected_faults(
            job,
            model,
            outcome.selected_faults,
        )
        result.soft_output = SoftOutput(
            gap=outcome.cluster_gap,
            source=UNION_FIND_CLUSTER_GAP_SOURCE,
        )
        return result

    def _graph_for_model(self, model, job_label: str) -> UnionFindGraph:
        import weakref

        model_identity = id(model)
        entry = self._graphs.get(model_identity)
        graph = entry[1] if entry is not None and entry[0]() is model else None
        if graph is None:
            location = (
                f"{job_label} Union-Find window model"
                if job_label
                else "Union-Find window model"
            )
            graph = _graph_from_model(model, location=location)

            def discard_dead_model(reference) -> None:
                current = self._graphs.get(model_identity)
                if current is not None and current[0] is reference:
                    del self._graphs[model_identity]

            reference = weakref.ref(model, discard_dead_model)
            self._graphs[model_identity] = (reference, graph)
        return graph
