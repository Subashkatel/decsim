"""Prior-weighted graphlike Union-Find growth and peeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


BOUNDARY = -1


@dataclass(frozen=True)
class Open:
    """The uncovered interval between two growing edge fronts."""

    lower: float
    upper: float


@dataclass(frozen=True)
class Closed:
    """A fully covered edge and its represented contact coordinate."""

    contact: float


@dataclass(frozen=True)
class UnionFindEdge:
    """One graphlike residual fault column in the weighted graph."""

    fault_index: int
    detector_a: int
    detector_b: int
    logical_observables: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class UnionFindGraph:
    """Immutable graph state shared by decodes of one placed fault model."""

    detector_count: int
    fault_count: int
    edges: tuple[UnionFindEdge, ...]
    adjacency: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    baseline_faults: tuple[int, ...]
    baseline_syndrome: tuple[int, ...]
    logical_observables_by_fault: tuple[tuple[int, ...], ...] = ()
    logical_observable_count: int = 0


@dataclass(frozen=True)
class UnionFindHardEvidence:
    """Immutable weighted growth and peeling evidence from one hard decode."""

    graph: UnionFindGraph
    syndrome: tuple[int, ...]
    residual_syndrome: tuple[int, ...]
    baseline_faults: tuple[int, ...]
    selected_faults: tuple[int, ...]
    contact_faults: tuple[int, ...]
    edge_intervals: tuple[Open | Closed, ...]
    erasure_forest_faults: tuple[int, ...]
    logical_observables: tuple[int, ...]


class _DisjointSet:
    """Per-decode cluster parity and shared-boundary state."""

    def __init__(self, detector_count: int, syndrome) -> None:
        boundary_node = detector_count
        node_count = detector_count + 1
        self.parent = list(range(node_count))
        self.parity = [
            int(syndrome[node]) if node < detector_count else 0
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
        self.parity[survivor] ^= self.parity[absorbed]
        self.touches_boundary[survivor] = (
            self.touches_boundary[survivor]
            or self.touches_boundary[absorbed]
        )
        return survivor

    def is_active(self, root: int) -> bool:
        root = self.find(root)
        return self.parity[root] == 1 and not self.touches_boundary[root]


def _natural_log_odds(residual_probability: float) -> float:
    if residual_probability == 0.5:
        return 0.0
    if residual_probability >= 0.25:
        return math.log1p(
            (1.0 - 2.0 * residual_probability) / residual_probability
        )
    return math.log1p(-residual_probability) - math.log(residual_probability)


def _endpoint_node(detector: int, detector_count: int) -> int:
    return detector_count if detector == BOUNDARY else detector


def _graph_from_model(faults, *, location: str) -> UnionFindGraph:
    import numpy as np

    from ..detector_error_model import validate_graphlike_matrices

    raw_check = np.asarray(faults.check)
    raw_priors = np.asarray(faults.priors)
    raw_observables = np.asarray(faults.observables)
    validate_graphlike_matrices(raw_check, raw_observables, location=location)
    fault_count = raw_check.shape[1]
    if raw_priors.ndim != 1 or raw_priors.size != fault_count:
        raise ValueError(
            f"{location} priors must have one entry per fault column"
        )
    if not np.all(np.isfinite(raw_priors)):
        raise ValueError(f"{location} priors must be finite")
    if not np.all((raw_priors >= 0.0) & (raw_priors <= 1.0)):
        raise ValueError(f"{location} priors must lie in [0, 1]")

    check = raw_check.astype(np.uint8, copy=False)
    observables = raw_observables.astype(np.uint8, copy=False)
    priors = raw_priors.astype(float, copy=False)
    baseline = (priors > 0.5).astype(np.uint8)
    baseline_syndrome = (check @ baseline) % 2

    edges = []
    adjacency: dict[int, list[tuple[int, int]]] = {
        node: [] for node in range(check.shape[0])
    }
    adjacency[BOUNDARY] = []
    for fault_index in range(fault_count):
        probability = float(priors[fault_index])
        residual_probability = (
            1.0 - probability if baseline[fault_index] else probability
        )
        if residual_probability == 0.0:
            continue
        detectors = tuple(
            int(value) for value in np.nonzero(check[:, fault_index])[0]
        )
        if len(detectors) == 0:
            detector_a = BOUNDARY
            detector_b = BOUNDARY
        elif len(detectors) == 1:
            detector_a = detectors[0]
            detector_b = BOUNDARY
        else:
            detector_a, detector_b = detectors
        edge = UnionFindEdge(
            fault_index=fault_index,
            detector_a=detector_a,
            detector_b=detector_b,
            logical_observables=tuple(
                int(value) for value in observables[:, fault_index]
            ),
            weight=_natural_log_odds(residual_probability),
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
                    key=lambda item: (edges[item[1]].fault_index, item[0]),
                )
            ),
        )
        for node, neighbors in sorted(adjacency.items())
    )
    logical_columns = tuple(
        tuple(int(value) for value in observables[:, fault_index])
        for fault_index in range(fault_count)
    )
    return UnionFindGraph(
        detector_count=check.shape[0],
        fault_count=fault_count,
        edges=tuple(edges),
        adjacency=frozen_adjacency,
        baseline_faults=tuple(int(value) for value in baseline),
        baseline_syndrome=tuple(int(value) for value in baseline_syndrome),
        logical_observables_by_fault=logical_columns,
        logical_observable_count=observables.shape[0],
    )


def _closed_contact_batch(
    graph: UnionFindGraph,
    disjoint_set: _DisjointSet,
    edge_intervals: list[Open | Closed],
) -> tuple[int, ...]:
    detector_count = graph.detector_count
    frozen_roots = tuple(
        (
            disjoint_set.find(
                _endpoint_node(edge.detector_a, detector_count)
            ),
            disjoint_set.find(
                _endpoint_node(edge.detector_b, detector_count)
            ),
        )
        for edge in graph.edges
    )
    contacts = tuple(
        edge_index
        for edge_index, interval in enumerate(edge_intervals)
        if isinstance(interval, Closed)
        and frozen_roots[edge_index][0] != frozen_roots[edge_index][1]
    )
    return tuple(
        sorted(contacts, key=lambda index: graph.edges[index].fault_index)
    )


def _union_contact_batch(
    graph: UnionFindGraph,
    disjoint_set: _DisjointSet,
    contact_edge_indices: tuple[int, ...],
) -> None:
    detector_count = graph.detector_count
    for edge_index in contact_edge_indices:
        edge = graph.edges[edge_index]
        disjoint_set.union(
            _endpoint_node(edge.detector_a, detector_count),
            _endpoint_node(edge.detector_b, detector_count),
        )


def _advance_open_interval(
    interval: Open,
    weight: float,
    elapsed: float,
    left_active: bool,
    right_active: bool,
) -> Open:
    lower = interval.lower + elapsed if left_active else interval.lower
    upper = interval.upper - elapsed if right_active else interval.upper
    if left_active and lower <= interval.lower:
        raise RuntimeError("weighted Union-Find left edge front did not advance")
    if right_active and upper >= interval.upper:
        raise RuntimeError("weighted Union-Find right edge front did not advance")
    if not (0.0 <= lower < upper <= weight):
        raise RuntimeError(
            "weighted Union-Find edge update lost represented interval order"
        )
    return Open(lower, upper)


def _weighted_growth_outcome(graph: UnionFindGraph, syndrome):
    disjoint_set = _DisjointSet(graph.detector_count, syndrome)
    edge_intervals: list[Open | Closed] = [
        Closed(0.0) if edge.weight == 0.0 else Open(0.0, edge.weight)
        for edge in graph.edges
    ]
    contact_edges = list(
        _closed_contact_batch(graph, disjoint_set, edge_intervals)
    )
    _union_contact_batch(graph, disjoint_set, tuple(contact_edges))

    event_count = 0
    while True:
        detector_count = graph.detector_count
        frozen_roots = tuple(
            (
                disjoint_set.find(
                    _endpoint_node(edge.detector_a, detector_count)
                ),
                disjoint_set.find(
                    _endpoint_node(edge.detector_b, detector_count)
                ),
            )
            for edge in graph.edges
        )
        all_roots = {
            disjoint_set.find(node)
            for node in range(graph.detector_count + 1)
        }
        frozen_active = {
            root: disjoint_set.is_active(root) for root in all_roots
        }
        if not any(frozen_active.values()):
            break

        candidates = []
        for edge_index, (edge, interval) in enumerate(
            zip(graph.edges, edge_intervals)
        ):
            if not isinstance(interval, Open):
                continue
            left_root, right_root = frozen_roots[edge_index]
            if left_root == right_root:
                continue
            rate = int(frozen_active[left_root]) + int(
                frozen_active[right_root]
            )
            if rate:
                candidates.append(
                    ((interval.upper - interval.lower) / rate, edge_index)
                )
        if not candidates:
            active_roots = tuple(
                sorted(root for root, active in frozen_active.items() if active)
            )
            raise RuntimeError(
                "odd Union-Find cluster has no outward graph edge: "
                f"roots {active_roots}"
            )
        elapsed = min(candidate for candidate, _edge_index in candidates)
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError(
                "weighted Union-Find event has no positive represented growth"
            )
        selected = {
            edge_index
            for candidate, edge_index in candidates
            if candidate == elapsed
        }

        proposals: list[Open | Closed] = []
        for edge_index, (edge, interval) in enumerate(
            zip(graph.edges, edge_intervals)
        ):
            if not isinstance(interval, Open):
                proposals.append(interval)
                continue
            left_root, right_root = frozen_roots[edge_index]
            left_active = frozen_active[left_root]
            right_active = frozen_active[right_root]
            if left_root != right_root:
                if edge_index in selected:
                    contact = (
                        interval.lower + elapsed
                        if left_active
                        else interval.upper - elapsed
                    )
                    if not math.isfinite(contact) or not (
                        0.0 <= contact <= edge.weight
                    ):
                        raise RuntimeError(
                            "weighted Union-Find contact lies outside its edge"
                        )
                    proposals.append(Closed(contact))
                else:
                    proposals.append(
                        _advance_open_interval(
                            interval,
                            edge.weight,
                            elapsed,
                            left_active,
                            right_active,
                        )
                    )
                continue
            if not left_active:
                proposals.append(interval)
                continue
            half_remaining = (interval.upper - interval.lower) / 2.0
            if elapsed >= half_remaining:
                proposals.append(Closed(interval.lower + half_remaining))
            else:
                proposals.append(
                    _advance_open_interval(
                        interval,
                        edge.weight,
                        elapsed,
                        True,
                        True,
                    )
                )

        edge_intervals = proposals
        contact_batch = tuple(
            sorted(selected, key=lambda index: graph.edges[index].fault_index)
        )
        contact_edges.extend(contact_batch)
        _union_contact_batch(graph, disjoint_set, contact_batch)
        event_count += 1
        if event_count > len(graph.edges):
            raise RuntimeError(
                "weighted Union-Find growth exceeded its finite graph bound"
            )

    return disjoint_set, tuple(edge_intervals), tuple(contact_edges)


def _minimum_weight_contact_forest(
    graph: UnionFindGraph,
    contact_edge_indices: tuple[int, ...],
) -> tuple[int, ...]:
    forest_set = _DisjointSet(
        graph.detector_count,
        [0] * graph.detector_count,
    )
    forest_edges = []
    for edge_index in sorted(
        contact_edge_indices,
        key=lambda index: (
            graph.edges[index].weight,
            graph.edges[index].fault_index,
        ),
    ):
        edge = graph.edges[edge_index]
        left = _endpoint_node(edge.detector_a, graph.detector_count)
        right = _endpoint_node(edge.detector_b, graph.detector_count)
        if forest_set.find(left) == forest_set.find(right):
            continue
        forest_set.union(left, right)
        forest_edges.append(edge_index)
    return tuple(forest_edges)


def _peel_forest(
    graph: UnionFindGraph,
    syndrome,
    forest_edges: tuple[int, ...],
) -> tuple[int, ...]:
    detector_count = graph.detector_count
    boundary_node = detector_count
    adjacency = {node: [] for node in range(detector_count + 1)}
    for edge_index in forest_edges:
        edge = graph.edges[edge_index]
        left = _endpoint_node(edge.detector_a, detector_count)
        right = _endpoint_node(edge.detector_b, detector_count)
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    for neighbors in adjacency.values():
        neighbors.sort(
            key=lambda item: (graph.edges[item[1]].fault_index, item[0])
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
            node != boundary_node and syndrome[node] for node in component
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
            node: 0 if node == boundary_node else int(syndrome[node])
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
    return tuple(sorted(selected_edges))


def _decode_graph(graph: UnionFindGraph, syndrome) -> UnionFindHardEvidence:
    import numpy as np

    raw_syndrome = np.asarray(syndrome)
    if raw_syndrome.ndim != 1 or raw_syndrome.size != graph.detector_count:
        raise ValueError(
            "Union-Find syndrome must be a one-dimensional detector vector "
            f"of length {graph.detector_count}"
        )
    if not np.all((raw_syndrome == 0) | (raw_syndrome == 1)):
        raise ValueError("Union-Find syndrome must contain only binary values")
    syndrome_array = raw_syndrome.astype(np.uint8, copy=False)
    baseline_syndrome = np.asarray(graph.baseline_syndrome, dtype=np.uint8)
    residual_syndrome = syndrome_array ^ baseline_syndrome

    _disjoint_set, intervals, contacts = _weighted_growth_outcome(
        graph, residual_syndrome
    )
    forest = _minimum_weight_contact_forest(graph, contacts)
    selected_residual_edges = _peel_forest(
        graph, residual_syndrome, forest
    )
    selected_faults = list(graph.baseline_faults)
    for edge_index in selected_residual_edges:
        selected_faults[graph.edges[edge_index].fault_index] ^= 1

    reproduced = baseline_syndrome.copy()
    for edge_index in selected_residual_edges:
        edge = graph.edges[edge_index]
        if edge.detector_a != BOUNDARY:
            reproduced[edge.detector_a] ^= 1
        if edge.detector_b != BOUNDARY:
            reproduced[edge.detector_b] ^= 1
    if not np.array_equal(reproduced, syndrome_array):
        unmatched = tuple(
            int(value) for value in np.nonzero(reproduced ^ syndrome_array)[0]
        )
        raise RuntimeError(
            "Union-Find peeling correction does not reproduce the syndrome; "
            f"unmatched detectors {unmatched}"
        )

    logical_observables = tuple(
        sum(
            selected_faults[fault_index]
            * graph.logical_observables_by_fault[fault_index][logical_index]
            for fault_index in range(graph.fault_count)
        )
        % 2
        for logical_index in range(graph.logical_observable_count)
    )
    return UnionFindHardEvidence(
        graph=graph,
        syndrome=tuple(int(value) for value in syndrome_array),
        residual_syndrome=tuple(int(value) for value in residual_syndrome),
        baseline_faults=graph.baseline_faults,
        selected_faults=tuple(selected_faults),
        contact_faults=tuple(
            graph.edges[index].fault_index for index in contacts
        ),
        edge_intervals=intervals,
        erasure_forest_faults=tuple(
            graph.edges[index].fault_index for index in forest
        ),
        logical_observables=logical_observables,
    )


def decode_union_find_model(model, syndrome) -> UnionFindHardEvidence:
    """Decode one placed weighted graphlike model and return hard evidence."""
    from ..detector_error_model import FaultRepresentation

    faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    graph = _graph_from_model(faults, location="Union-Find window model")
    return _decode_graph(graph, syndrome)
