"""Uniform graphlike Union-Find growth and peeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


BOUNDARY = -1


@dataclass(frozen=True)
class UnionFindEdge:
    """One graphlike fault column in the unit decoding graph."""

    fault_index: int
    detector_a: int
    detector_b: int
    logical_observables: tuple[int, ...]


@dataclass(frozen=True)
class UnionFindGraph:
    """Immutable graph state shared by decodes of one placed fault model."""

    detector_count: int
    fault_count: int
    edges: tuple[UnionFindEdge, ...]
    adjacency: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True)
class UnionFindHardEvidence:
    """Immutable graph, growth, and peeling evidence from one hard decode."""

    graph: UnionFindGraph
    syndrome: tuple[int, ...]
    selected_faults: tuple[int, ...]
    completed_growth_faults: tuple[int, ...]
    cluster_label_by_syndrome_center: tuple[tuple[int, int], ...]
    radius_by_syndrome_center: tuple[tuple[int, float], ...]
    erasure_forest_faults: tuple[int, ...]


class _DisjointSet:
    """Per-decode cluster state for completed half-edge growth."""

    def __init__(self, detector_count: int, syndrome) -> None:
        boundary_node = detector_count
        node_count = detector_count + 1
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


def _graph_from_model(faults, *, location: str) -> UnionFindGraph:
    import numpy as np

    from ..detector_error_model import validate_graphlike_matrices

    raw_check = np.asarray(faults.check)
    raw_observables = np.asarray(faults.observables)
    validate_graphlike_matrices(
        raw_check,
        raw_observables,
        location=location,
    )
    check = raw_check.astype(np.uint8, copy=False)
    observables = raw_observables.astype(np.uint8, copy=False)

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
            logical_observables=tuple(
                int(value) for value in observables[:, fault_index]
            ),
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
            )
            for root in invalid_roots
        ),
        key=lambda item: (item[0], item[1]),
    )
    grown_centers: set[int] = set()
    visit_count = 0

    for _size, anchor in snapshots:
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


def _decode_graph(
    graph: UnionFindGraph,
    syndrome,
) -> UnionFindHardEvidence:
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
    syndrome_array = raw_syndrome.astype(np.uint8, copy=False)
    syndrome_bits = tuple(int(value) for value in syndrome_array)

    disjoint_set, radii, completed = _growth_outcome(
        graph,
        syndrome_array,
    )
    selected_edges, forest = _peel_completed_edges(
        graph,
        syndrome_array,
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
    if not np.array_equal(reproduced, syndrome_array):
        unmatched = tuple(
            int(value)
            for value in np.nonzero(reproduced ^ syndrome_array)[0]
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
    return UnionFindHardEvidence(
        graph=graph,
        syndrome=syndrome_bits,
        selected_faults=tuple(selected_faults),
        completed_growth_faults=tuple(
            graph.edges[index].fault_index for index in completed
        ),
        cluster_label_by_syndrome_center=tuple(
            sorted(cluster_label_by_center.items())
        ),
        radius_by_syndrome_center=tuple(sorted(radii.items())),
        erasure_forest_faults=tuple(
            graph.edges[index].fault_index for index in forest
        ),
    )


def decode_union_find_model(model, syndrome) -> UnionFindHardEvidence:
    """Decode one placed unit-geometry model and return hard evidence."""
    from ..detector_error_model import FaultRepresentation

    faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    graph = _graph_from_model(
        faults,
        location="Union-Find window model",
    )
    return _decode_graph(graph, syndrome)
