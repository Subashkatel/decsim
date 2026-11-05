"""The cluster record of one union-find decode: its clusters and its doubt.

It reads the growth a hard decode returned (UnionFindHardEvidence) and
reports what a chip that grows the clusters can count on the way, and
which clusters the cluster gap's shortest odd walk passes through: the
doubtful clusters. The walk is Meister et al. 2405.07433 Definition 9
as union_find/cluster_gap.c runs it, which also marks the edges it
crosses. A cluster is the detectors the growth joined by fully covered
edges. The boundary joins no two clusters, as in the decode itself,
and only a cluster that holds a detection event counts.
"""

import dataclasses
from typing import Union

import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records


@dataclasses.dataclass(frozen=True)
class ClusterRecord:
    """The counts of one growth, and where its shortest odd walk runs.

    walk_half_ticks is the gap in the growth's half ticks, infinite when
    no odd walk exists. walk_detection_events are the detection events
    of the doubtful clusters, as window detector indices, ascending.
    """

    cluster_count: int
    largest_cluster_events: int
    largest_cluster_edges: int
    growth_step_count: int
    walk_half_ticks: Union[int, float]
    walk_cluster_count: int
    walk_event_count: int
    is_walk_through_boundary: bool
    walk_detection_events: tuple


def cluster_record(
    evidence: evidence_records.UnionFindHardEvidence,
) -> ClusterRecord:
    """The counts of one growth and the doubtful clusters of its walk."""
    graph = evidence.graph
    roots = detector_roots(graph, evidence.edge_intervals)
    events_by_root = cluster_events(roots, evidence.residual_syndrome)
    largest_root = _largest_root(events_by_root)
    largest_events = 0
    largest_edges = 0
    if largest_root is not None:
        largest_events = len(events_by_root[largest_root])
        largest_edges = _closed_edge_count(graph, evidence, roots, largest_root)
    walk = compiled_decoder.cluster_gap_walk(graph, evidence.edge_intervals)
    walk_half_ticks, walk_edges = walk
    walk_roots = _walk_roots(graph, walk_edges, roots, events_by_root)
    walk_events = _events_of(walk_roots, events_by_root)
    is_through_boundary = _crosses_boundary(graph, walk_edges)
    return ClusterRecord(
        cluster_count=len(events_by_root),
        largest_cluster_events=largest_events,
        largest_cluster_edges=largest_edges,
        growth_step_count=len(evidence.growth_steps),
        walk_half_ticks=walk_half_ticks,
        walk_cluster_count=len(walk_roots),
        walk_event_count=len(walk_events),
        is_walk_through_boundary=is_through_boundary,
        walk_detection_events=walk_events,
    )


def detector_roots(
    graph: evidence_records.UnionFindGraph, edge_intervals: tuple
) -> list:
    """Each detector's cluster, as the lowest detector of that cluster."""
    parents = list(range(graph.detector_count))
    edges = zip(graph.edges, edge_intervals, strict=True)
    for edge, interval in edges:
        if not isinstance(interval, evidence_records.Closed):
            continue
        if _touches_boundary(edge):
            continue
        _join(parents, edge.detector_a, edge.detector_b)
    roots = []
    for detector in range(graph.detector_count):
        root = _root(parents, detector)
        roots.append(root)
    return roots


def cluster_events(roots: list, residual_syndrome: tuple) -> dict:
    """The detection events of each cluster that holds one, keyed by root."""
    events_by_root = {}
    for detector, bit in enumerate(residual_syndrome):
        if not bit:
            continue
        root = roots[detector]
        events = events_by_root.setdefault(root, [])
        events.append(detector)
    return events_by_root


def _join(parents: list, first: int, second: int) -> None:
    first_root = _root(parents, first)
    second_root = _root(parents, second)
    lower = min(first_root, second_root)
    higher = max(first_root, second_root)
    parents[higher] = lower


def _root(parents: list, detector: int) -> int:
    """The detector's root, halving the path on the way."""
    while parents[detector] != detector:
        grandparent = parents[parents[detector]]
        parents[detector] = grandparent
        detector = grandparent
    return detector


def _largest_root(events_by_root: dict):
    """The cluster with the most detection events; the lowest root on a tie."""
    largest = None
    for root in sorted(events_by_root):
        if largest is None:
            largest = root
            continue
        if len(events_by_root[root]) > len(events_by_root[largest]):
            largest = root
    return largest


def _closed_edge_count(graph, evidence, roots: list, cluster_root: int) -> int:
    """The fully covered edges with an end in the cluster, boundary ones too."""
    count = 0
    edges = zip(graph.edges, evidence.edge_intervals, strict=True)
    for edge, interval in edges:
        if not isinstance(interval, evidence_records.Closed):
            continue
        if _has_end_in(edge, roots, cluster_root):
            count += 1
    return count


def _has_end_in(edge, roots: list, cluster_root: int) -> bool:
    for detector in _detectors_of(edge):
        if roots[detector] == cluster_root:
            return True
    return False


def _walk_roots(
    graph, walk_edges: tuple, roots: list, events_by_root: dict
) -> tuple:
    """The clusters holding an event that the walk's edges reach, ascending."""
    found = set()
    for edge_index in walk_edges:
        edge = graph.edges[edge_index]
        detectors = _detectors_of(edge)
        edge_roots = [roots[detector] for detector in detectors]
        found.update(edge_roots)
    held = found.intersection(events_by_root)
    return tuple(sorted(held))


def _detectors_of(edge) -> tuple:
    """The edge's ends that are detectors, the boundary left out."""
    ends = (edge.detector_a, edge.detector_b)
    return tuple(end for end in ends if end != window_decoder.BOUNDARY)


def _events_of(cluster_roots: tuple, events_by_root: dict) -> tuple:
    events = []
    for root in cluster_roots:
        events.extend(events_by_root[root])
    return tuple(sorted(events))


def _crosses_boundary(graph, walk_edges: tuple) -> bool:
    for edge_index in walk_edges:
        edge = graph.edges[edge_index]
        if _touches_boundary(edge):
            return True
    return False


def _touches_boundary(edge) -> bool:
    if edge.detector_a == window_decoder.BOUNDARY:
        return True
    return edge.detector_b == window_decoder.BOUNDARY
