"""Prior-weighted graphlike Union-Find: weighted growth and peeling.

Delfosse and Nickerson 1709.06218 (tmp/uf-decoder-research/papers):
every odd cluster grows by one half-edge per round (Algorithm 1, step
4), clusters that meet fuse, and the peeling decoder reads the
correction off a spanning forest of the grown erasure (Algorithm 2).
The weighted variant is Huang, Newman and Brown 2004.04693 (same
folder): an edge of log-odds weight w has the integer length
round(w / weight_step), never below one tick, so a likelier fault is
crossed sooner; that quantization is _quantize_weight_ticks. Growth
here is event driven: each round jumps to the next tick at which some
front closes an edge, and a front advances one half-tick per tick.
"""

import dataclasses
import math
from numbers import Real
from typing import Optional, Union

import numpy

BOUNDARY = -1


@dataclasses.dataclass(frozen=True)
class Open:
    """The uncovered interval between two growing edge fronts."""

    lower_tick: int
    upper_tick: int


@dataclasses.dataclass(frozen=True)
class Closed:
    """A fully covered edge."""


@dataclasses.dataclass(frozen=True)
class UnionFindEdge:
    """One graphlike residual fault column in the weighted graph."""

    fault_index: int
    detector_a: int
    detector_b: int
    logical_observables: tuple[int, ...]
    length_half_ticks: int


@dataclasses.dataclass(frozen=True)
class UnionFindGraph:
    """Immutable graph state shared by decodes of one placed fault model."""

    detector_count: int
    fault_count: int
    edges: tuple[UnionFindEdge, ...]
    baseline_faults: tuple[int, ...]
    baseline_syndrome: tuple[int, ...]
    logical_observables_by_fault: tuple[tuple[int, ...], ...] = ()
    logical_observable_count: int = 0


@dataclasses.dataclass(frozen=True)
class UnionFindHardEvidence:
    """Immutable weighted growth and peeling evidence from one hard decode."""

    graph: UnionFindGraph
    syndrome: tuple[int, ...]
    residual_syndrome: tuple[int, ...]
    selected_faults: tuple[int, ...]
    contact_faults: tuple[int, ...]
    edge_intervals: tuple[Union[Open, Closed], ...]
    erasure_forest_faults: tuple[int, ...]
    logical_observables: tuple[int, ...]
    # detectors the best-effort correction leaves unexplained: an odd
    # cluster that ran out of edges before reaching another defect or the
    # boundary
    unmatched_detectors: tuple[int, ...] = ()


def normalized_weight_step(weight_step) -> float:
    """The weight step as a positive finite float; anything else is refused."""
    if isinstance(weight_step, bool) or not isinstance(weight_step, Real):
        raise TypeError("Union-Find weight_step must be a real number")
    normalized = float(weight_step)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("Union-Find weight_step must be finite and positive")
    return normalized


def graph_from_model(
    faults, *, location: str, weight_step: float
) -> UnionFindGraph:
    """The weighted graph of one placed graphlike model.

    Faults likelier than one half are the baseline correction; every
    other column becomes an edge whose length is its residual log-odds
    in weight ticks.
    """
    check = faults.check
    raw_priors = numpy.asarray(faults.priors)
    fault_count = check.shape[1]
    _check_priors(raw_priors, fault_count, location)
    # observables are few rows; dense per-fault columns are cheap to read
    observables = faults.observables.toarray()
    observables = observables.astype(numpy.uint8, copy=False)
    priors = raw_priors.astype(float, copy=False)
    likely = priors > 0.5
    baseline = likely.astype(numpy.uint8)
    baseline_syndrome = _parity_product(check, baseline)
    edges = []
    for fault_index in range(fault_count):
        edge = _edge_of(
            check, priors, baseline, observables, fault_index, weight_step
        )
        if edge is not None:
            edges.append(edge)
    logical_columns = _logical_columns(observables, fault_count)
    baseline_faults = _int_tuple(baseline)
    baseline_syndrome = _int_tuple(baseline_syndrome)
    return UnionFindGraph(
        detector_count=check.shape[0],
        fault_count=fault_count,
        edges=tuple(edges),
        baseline_faults=baseline_faults,
        baseline_syndrome=baseline_syndrome,
        logical_observables_by_fault=logical_columns,
        logical_observable_count=observables.shape[0],
    )


def decode_graph(graph: UnionFindGraph, syndrome) -> UnionFindHardEvidence:
    """Decode one syndrome on the graph: grow, take the forest, peel, report."""
    syndrome_array = _checked_syndrome(graph, syndrome)
    baseline_syndrome = numpy.asarray(
        graph.baseline_syndrome, dtype=numpy.uint8
    )
    syndrome_bits = _int_tuple(syndrome_array)
    residual_syndrome = syndrome_array ^ baseline_syndrome
    residual_bits = _int_tuple(residual_syndrome)
    _disjoint_set, intervals, contacts = _weighted_growth_outcome(
        graph, residual_syndrome
    )
    forest = _minimum_weight_contact_forest(graph, contacts)
    selected_residual_edges = _peel_forest(graph, residual_syndrome, forest)
    selected_faults = _selected_faults(graph, selected_residual_edges)
    reproduced = _reproduced_syndrome(
        graph, baseline_syndrome, selected_residual_edges
    )
    mismatch = reproduced ^ syndrome_array
    unmatched = numpy.nonzero(mismatch)
    unmatched_detectors = _int_tuple(unmatched[0])
    logical_observables = _logical_observables(graph, selected_faults)
    contact_faults = _fault_indices(graph, contacts)
    erasure_forest_faults = _fault_indices(graph, forest)
    return UnionFindHardEvidence(
        graph=graph,
        syndrome=syndrome_bits,
        residual_syndrome=residual_bits,
        selected_faults=tuple(selected_faults),
        contact_faults=contact_faults,
        edge_intervals=intervals,
        erasure_forest_faults=erasure_forest_faults,
        logical_observables=logical_observables,
        unmatched_detectors=unmatched_detectors,
    )


class _DisjointSet:
    """Per-decode cluster parity and shared-boundary state."""

    def __init__(self, detector_count: int, syndrome) -> None:
        node_count = detector_count + 1
        self.parent = list(range(node_count))
        parity = []
        for node in range(detector_count):
            parity.append(int(syndrome[node]))
        parity.append(0)  # the boundary node carries no defect
        self.parity = parity
        touches_boundary = [False] * detector_count
        touches_boundary.append(True)
        self.touches_boundary = touches_boundary

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
            self.touches_boundary[survivor] or self.touches_boundary[absorbed]
        )
        return survivor

    def is_active(self, root: int) -> bool:
        """An odd cluster that does not touch the boundary keeps growing."""
        root = self.find(root)
        if self.touches_boundary[root]:
            return False
        return self.parity[root] == 1


def _natural_log_odds(residual_probability: float) -> float:
    if residual_probability == 0.5:
        return 0.0
    if residual_probability >= 0.25:
        ratio = (1.0 - 2.0 * residual_probability) / residual_probability
        return math.log1p(ratio)
    log_survival = math.log1p(-residual_probability)
    log_probability = math.log(residual_probability)
    return log_survival - log_probability


def _quantize_weight_ticks(weight: float, weight_step: float) -> int:
    """The edge length in weight ticks: round(weight / weight_step), at least 1.

    Exact in rational arithmetic, so equal weights get equal lengths on
    every platform (Huang, Newman and Brown 2004.04693: integer edge
    lengths from the log-odds).
    """
    if weight == 0.0:
        return 0
    weight_numerator, weight_denominator = weight.as_integer_ratio()
    step_numerator, step_denominator = weight_step.as_integer_ratio()
    numerator = weight_numerator * step_denominator
    denominator = weight_denominator * step_numerator
    whole, remainder = divmod(numerator, denominator)
    rounds_up = 2 * remainder >= denominator
    ticks = whole + int(rounds_up)
    return max(1, ticks)


def _endpoint_node(detector: int, detector_count: int) -> int:
    if detector == BOUNDARY:
        return detector_count
    return detector


def _check_priors(raw_priors, fault_count: int, location: str) -> None:
    if raw_priors.ndim != 1 or raw_priors.size != fault_count:
        raise ValueError(
            f"{location} priors must have one entry per fault column"
        )
    is_finite = numpy.isfinite(raw_priors)
    if not numpy.all(is_finite):
        raise ValueError(f"{location} priors must be finite")
    at_least_zero = raw_priors >= 0.0
    at_most_one = raw_priors <= 1.0
    inside = at_least_zero & at_most_one
    if not numpy.all(inside):
        raise ValueError(f"{location} priors must lie in [0, 1]")


def _parity_product(matrix, vector):
    """The matrix times the vector over GF(2), as a flat array."""
    matrix_integers = matrix.astype(numpy.int64)
    vector_integers = vector.astype(numpy.int64)
    product = matrix_integers @ vector_integers
    product = numpy.asarray(product)
    flat = product.ravel()
    return flat % 2


def _edge_of(
    check, priors, baseline, observables, fault_index: int, weight_step: float
) -> Optional[UnionFindEdge]:
    """The edge of one fault column; None when its residual is certain."""
    probability = float(priors[fault_index])
    residual_probability = probability
    if baseline[fault_index]:
        residual_probability = 1.0 - probability
    if residual_probability == 0.0:
        return None
    detector_a, detector_b = _endpoints(check, fault_index)
    log_odds = _natural_log_odds(residual_probability)
    weight_ticks = _quantize_weight_ticks(log_odds, weight_step)
    column = observables[:, fault_index]
    logical_observables = _int_tuple(column)
    length_half_ticks = 2 * weight_ticks
    return UnionFindEdge(
        fault_index=fault_index,
        detector_a=detector_a,
        detector_b=detector_b,
        logical_observables=logical_observables,
        length_half_ticks=length_half_ticks,
    )


def _endpoints(check, fault_index: int) -> tuple:
    """The two detectors of a column; a missing one is the boundary."""
    start = check.indptr[fault_index]
    end = check.indptr[fault_index + 1]
    detectors = _int_tuple(check.indices[start:end])
    if len(detectors) == 0:
        return BOUNDARY, BOUNDARY
    if len(detectors) == 1:
        return detectors[0], BOUNDARY
    detector_a, detector_b = detectors
    return detector_a, detector_b


def _logical_columns(observables, fault_count: int) -> tuple:
    """Each fault's logical-observable column as a tuple of bits."""
    columns = []
    for fault_index in range(fault_count):
        column = observables[:, fault_index]
        bits = _int_tuple(column)
        columns.append(bits)
    return tuple(columns)


def _int_tuple(values) -> tuple:
    integers = []
    for value in values:
        integers.append(int(value))
    return tuple(integers)


def _fault_indices(graph: UnionFindGraph, edge_indices) -> tuple:
    fault_indices = []
    for edge_index in edge_indices:
        fault_indices.append(graph.edges[edge_index].fault_index)
    return tuple(fault_indices)


def _in_fault_order(graph: UnionFindGraph, edge_indices) -> tuple:
    keyed = []
    for edge_index in edge_indices:
        keyed.append((graph.edges[edge_index].fault_index, edge_index))
    keyed.sort()
    ordered = []
    for _fault_index, edge_index in keyed:
        ordered.append(edge_index)
    return tuple(ordered)


def _frozen_roots(graph: UnionFindGraph, disjoint_set: _DisjointSet) -> tuple:
    """(root of a, root of b) per edge, as the clusters stand now."""
    detector_count = graph.detector_count
    roots = []
    for edge in graph.edges:
        node_a = _endpoint_node(edge.detector_a, detector_count)
        node_b = _endpoint_node(edge.detector_b, detector_count)
        root_a = disjoint_set.find(node_a)
        root_b = disjoint_set.find(node_b)
        roots.append((root_a, root_b))
    return tuple(roots)


def _active_roots(graph: UnionFindGraph, disjoint_set: _DisjointSet) -> dict:
    """Whether each cluster still grows, by its root."""
    all_roots = set()
    node_count = graph.detector_count + 1
    for node in range(node_count):
        root = disjoint_set.find(node)
        all_roots.add(root)
    active = {}
    for root in all_roots:
        active[root] = disjoint_set.is_active(root)
    return active


def _closed_contact_batch(
    graph: UnionFindGraph, disjoint_set: _DisjointSet, edge_intervals: list
) -> tuple:
    """Closed edges whose ends are still in different clusters."""
    frozen_roots = _frozen_roots(graph, disjoint_set)
    contacts = []
    for edge_index, interval in enumerate(edge_intervals):
        if not isinstance(interval, Closed):
            continue
        left_root, right_root = frozen_roots[edge_index]
        if left_root == right_root:
            continue
        contacts.append(edge_index)
    return _in_fault_order(graph, contacts)


def _union_contact_batch(
    graph: UnionFindGraph, disjoint_set: _DisjointSet, contact_edge_indices
) -> None:
    detector_count = graph.detector_count
    for edge_index in contact_edge_indices:
        edge = graph.edges[edge_index]
        node_a = _endpoint_node(edge.detector_a, detector_count)
        node_b = _endpoint_node(edge.detector_b, detector_count)
        disjoint_set.union(node_a, node_b)


def _advance_open_interval(
    interval: Open,
    length_half_ticks: int,
    elapsed_ticks: int,
    left_active: bool,
    right_active: bool,
) -> Open:
    lower_tick = interval.lower_tick
    if left_active:
        lower_tick += elapsed_ticks
    upper_tick = interval.upper_tick
    if right_active:
        upper_tick -= elapsed_ticks
    if not 0 <= lower_tick < upper_tick <= length_half_ticks:
        raise RuntimeError(
            "weighted Union-Find edge update lost represented interval order"
        )
    return Open(lower_tick, upper_tick)


def _initial_intervals(graph: UnionFindGraph) -> list:
    intervals = []
    for edge in graph.edges:
        interval = Open(0, edge.length_half_ticks)
        if edge.length_half_ticks == 0:
            interval = Closed()
        intervals.append(interval)
    return intervals


def _closing_candidates(
    edge_intervals: list, frozen_roots: tuple, frozen_active: dict
) -> list:
    """(ticks until the edge closes, edge index) for every growing edge."""
    candidates = []
    for edge_index, interval in enumerate(edge_intervals):
        if not isinstance(interval, Open):
            continue
        left_root, right_root = frozen_roots[edge_index]
        if left_root == right_root:
            continue
        rate = int(frozen_active[left_root]) + int(frozen_active[right_root])
        if rate == 0:
            continue
        remaining_ticks = interval.upper_tick - interval.lower_tick
        padded_ticks = remaining_ticks + rate - 1
        ticks_until_close = padded_ticks // rate
        candidates.append((ticks_until_close, edge_index))
    return candidates


def _next_event_ticks(candidates: list) -> int:
    ticks = []
    for ticks_until_close, _edge_index in candidates:
        ticks.append(ticks_until_close)
    elapsed = min(ticks)
    if elapsed <= 0:
        raise RuntimeError(
            "weighted Union-Find event has no positive represented growth"
        )
    return elapsed


def _edges_closing_at(candidates: list, elapsed: int) -> set:
    selected = set()
    for ticks_until_close, edge_index in candidates:
        if ticks_until_close == elapsed:
            selected.add(edge_index)
    return selected


def _advanced_interval(
    edge: UnionFindEdge,
    edge_index: int,
    interval,
    frozen_roots: tuple,
    frozen_active: dict,
    elapsed: int,
    selected: set,
):
    """The edge's interval after every active front grew by elapsed ticks."""
    if not isinstance(interval, Open):
        return interval
    left_root, right_root = frozen_roots[edge_index]
    left_active = frozen_active[left_root]
    right_active = frozen_active[right_root]
    if left_root != right_root:
        if edge_index in selected:
            return Closed()
        return _advance_open_interval(
            interval, edge.length_half_ticks, elapsed, left_active, right_active
        )
    # both ends in one cluster: its front grows inward from both sides
    if not left_active:
        return interval
    remaining_ticks = interval.upper_tick - interval.lower_tick
    if 2 * elapsed >= remaining_ticks:
        return Closed()
    return _advance_open_interval(
        interval, edge.length_half_ticks, elapsed, True, True
    )


def _advanced_intervals(
    graph: UnionFindGraph,
    edge_intervals: list,
    frozen_roots: tuple,
    frozen_active: dict,
    elapsed: int,
    selected: set,
) -> list:
    proposals = []
    for edge_index, interval in enumerate(edge_intervals):
        edge = graph.edges[edge_index]
        proposal = _advanced_interval(
            edge,
            edge_index,
            interval,
            frozen_roots,
            frozen_active,
            elapsed,
            selected,
        )
        proposals.append(proposal)
    return proposals


def _weighted_growth_outcome(graph: UnionFindGraph, syndrome) -> tuple:
    """Grow every odd cluster until none is left or none can grow.

    Returns the clusters, the edge intervals and the contact edges in
    the order they closed.
    """
    disjoint_set = _DisjointSet(graph.detector_count, syndrome)
    edge_intervals = _initial_intervals(graph)
    first_contacts = _closed_contact_batch(graph, disjoint_set, edge_intervals)
    contact_edges = list(first_contacts)
    _union_contact_batch(graph, disjoint_set, first_contacts)
    event_count = 0
    while True:
        frozen_roots = _frozen_roots(graph, disjoint_set)
        frozen_active = _active_roots(graph, disjoint_set)
        growing = frozen_active.values()
        if not any(growing):
            break
        candidates = _closing_candidates(
            edge_intervals, frozen_roots, frozen_active
        )
        if not candidates:
            # every remaining odd cluster has no outward edge left: the
            # syndrome is not satisfiable inside this window; stop growing
            # and peel what there is (PECOS and ldpc do the same: best
            # effort)
            break
        elapsed = _next_event_ticks(candidates)
        selected = _edges_closing_at(candidates, elapsed)
        edge_intervals = _advanced_intervals(
            graph,
            edge_intervals,
            frozen_roots,
            frozen_active,
            elapsed,
            selected,
        )
        contact_batch = _in_fault_order(graph, selected)
        contact_edges.extend(contact_batch)
        _union_contact_batch(graph, disjoint_set, contact_batch)
        event_count += 1
        if event_count > len(graph.edges):
            raise RuntimeError(
                "weighted Union-Find growth exceeded its finite graph bound"
            )
    return disjoint_set, tuple(edge_intervals), tuple(contact_edges)


def _by_length_then_fault(graph: UnionFindGraph, edge_indices) -> list:
    keyed = []
    for edge_index in edge_indices:
        edge = graph.edges[edge_index]
        keyed.append((edge.length_half_ticks, edge.fault_index, edge_index))
    keyed.sort()
    ordered = []
    for _length, _fault_index, edge_index in keyed:
        ordered.append(edge_index)
    return ordered


def _minimum_weight_contact_forest(
    graph: UnionFindGraph, contact_edge_indices: tuple
) -> tuple:
    """A spanning forest of the contacts, lightest edges first (Kruskal)."""
    no_defects = [0] * graph.detector_count
    forest_set = _DisjointSet(graph.detector_count, no_defects)
    forest_edges = []
    for edge_index in _by_length_then_fault(graph, contact_edge_indices):
        edge = graph.edges[edge_index]
        left = _endpoint_node(edge.detector_a, graph.detector_count)
        right = _endpoint_node(edge.detector_b, graph.detector_count)
        left_root = forest_set.find(left)
        right_root = forest_set.find(right)
        if left_root == right_root:
            continue
        forest_set.union(left, right)
        forest_edges.append(edge_index)
    return tuple(forest_edges)


def _sort_neighbors(graph: UnionFindGraph, neighbors: list) -> None:
    """Neighbors by their edge's fault index, then by node, in place."""
    keyed = []
    for neighbor, edge_index in neighbors:
        fault_index = graph.edges[edge_index].fault_index
        keyed.append((fault_index, neighbor, edge_index))
    keyed.sort()
    neighbors.clear()
    for _fault_index, neighbor, edge_index in keyed:
        neighbors.append((neighbor, edge_index))


def _forest_adjacency(graph: UnionFindGraph, forest_edges: tuple) -> dict:
    detector_count = graph.detector_count
    node_count = detector_count + 1
    adjacency = {}
    for node in range(node_count):
        adjacency[node] = []
    for edge_index in forest_edges:
        edge = graph.edges[edge_index]
        left = _endpoint_node(edge.detector_a, detector_count)
        right = _endpoint_node(edge.detector_b, detector_count)
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    for neighbors in adjacency.values():
        _sort_neighbors(graph, neighbors)
    return adjacency


def _component_of(adjacency: dict, start: int) -> set:
    component = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node in component:
            continue
        component.add(node)
        for neighbor, _edge_index in adjacency[node]:
            pending.append(neighbor)
    return component


def _has_defect(component: set, syndrome, boundary_node: int) -> bool:
    for node in component:
        if node == boundary_node:
            continue
        if syndrome[node]:
            return True
    return False


def _defect_of(node: int, syndrome, boundary_node: int) -> int:
    if node == boundary_node:
        return 0
    return int(syndrome[node])


def _adopt_children(
    node: int, neighbors: list, parents: dict, parent_edges: dict, pending: list
) -> None:
    for neighbor, edge_index in reversed(neighbors):
        if neighbor in parents:
            continue
        parents[neighbor] = node
        parent_edges[neighbor] = edge_index
        pending.append(neighbor)


def _tree_order(
    adjacency: dict, root: int, parents: dict, parent_edges: dict
) -> list:
    """The tree's nodes from the root outward; parents filled on the way."""
    order = []
    pending = [root]
    while pending:
        node = pending.pop()
        order.append(node)
        _adopt_children(node, adjacency[node], parents, parent_edges, pending)
    return order


def _peel_component(
    adjacency: dict, component: set, root: int, syndrome, boundary_node: int
) -> set:
    """Peel one tree leaf to root: a defect leaf selects its parent edge.

    An odd component without the boundary keeps its root defect
    unmatched; the caller reports it through unmatched_detectors.
    """
    parents = {root: None}
    parent_edges = {}
    order = _tree_order(adjacency, root, parents, parent_edges)
    residual = {}
    for node in component:
        residual[node] = _defect_of(node, syndrome, boundary_node)
    selected = set()
    for node in reversed(order[1:]):
        if residual[node] == 0:
            continue
        edge_index = parent_edges[node]
        selected.add(edge_index)
        parent = parents[node]
        assert parent is not None, "every peeled node has a parent"
        residual[parent] ^= 1
    return selected


def _peel_forest(graph: UnionFindGraph, syndrome, forest_edges: tuple) -> tuple:
    """The forest edges that pair the defects (peeling, Algorithm 2)."""
    adjacency = _forest_adjacency(graph, forest_edges)
    boundary_node = graph.detector_count
    selected_edges = set()
    node_count = graph.detector_count + 1
    unseen = set(range(node_count))
    while unseen:
        component_start = min(unseen)
        component = _component_of(adjacency, component_start)
        unseen.difference_update(component)
        if not _has_defect(component, syndrome, boundary_node):
            continue
        root = min(component)
        if boundary_node in component:
            root = boundary_node
        peeled = _peel_component(
            adjacency, component, root, syndrome, boundary_node
        )
        selected_edges.update(peeled)
    return tuple(sorted(selected_edges))


def _checked_syndrome(graph: UnionFindGraph, syndrome):
    raw_syndrome = numpy.asarray(syndrome)
    if raw_syndrome.ndim != 1 or raw_syndrome.size != graph.detector_count:
        raise ValueError(
            "Union-Find syndrome must be a one-dimensional detector vector "
            f"of length {graph.detector_count}"
        )
    is_zero = raw_syndrome == 0
    is_one = raw_syndrome == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise ValueError("Union-Find syndrome must contain only binary values")
    return raw_syndrome.astype(numpy.uint8, copy=False)


def _selected_faults(graph: UnionFindGraph, edge_indices: tuple) -> list:
    selected = list(graph.baseline_faults)
    for edge_index in edge_indices:
        fault_index = graph.edges[edge_index].fault_index
        selected[fault_index] ^= 1
    return selected


def _reproduced_syndrome(
    graph: UnionFindGraph, baseline_syndrome, edge_indices
):
    reproduced = baseline_syndrome.copy()
    for edge_index in edge_indices:
        edge = graph.edges[edge_index]
        if edge.detector_a != BOUNDARY:
            reproduced[edge.detector_a] ^= 1
        if edge.detector_b != BOUNDARY:
            reproduced[edge.detector_b] ^= 1
    return reproduced


def _logical_observables(graph: UnionFindGraph, selected_faults: list) -> tuple:
    parities = [0] * graph.logical_observable_count
    for fault_index in range(graph.fault_count):
        if not selected_faults[fault_index]:
            continue
        column = graph.logical_observables_by_fault[fault_index]
        for logical_index, flips in enumerate(column):
            parities[logical_index] ^= flips
    return tuple(parities)
