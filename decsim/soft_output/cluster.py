"""Cluster-gap confidence from one Union-Find hard decode."""

from __future__ import annotations

import heapq
import itertools
import math

from ..detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from ..message import (
    DecodeJob,
    DecodeResult,
    SoftOutput,
    SoftOutputSource,
)
from ..union_find_decoder.decoder import UnionFindDecoder
from ..union_find_decoder.window_decoder import (
    UnionFindGraph,
    UnionFindHardEvidence,
)


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


def _graph_adjacency(graph: UnionFindGraph):
    return {node: neighbors for node, neighbors in graph.adjacency}


def _endpoint_potentials(
    graph: UnionFindGraph,
    radii: dict[int, float],
):
    adjacency = _graph_adjacency(graph)
    potentials = {node: math.inf for node in adjacency}
    sources: dict[int, int | None] = {
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
                edge.logical_observables[0] if segment_index == 0 else 0,
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


def _cluster_gap(hard_evidence: UnionFindHardEvidence) -> float:
    radii = dict(hard_evidence.radius_by_syndrome_center)
    cluster_labels = dict(
        hard_evidence.cluster_label_by_syndrome_center
    )
    intervals = _covered_intervals(
        hard_evidence.graph,
        radii,
        cluster_labels,
    )
    return _quotient_cluster_gap(hard_evidence.graph, intervals)


class UnionFindClusterGapDecoder:
    """Attach one-logical cluster-gap confidence to a hard Union-Find result.

    SCOPE:
    - Confidence requires exactly one nonzero logical-observable row.
    - The gap is in normalized graph-edge units.
    - Multiplying by ``log((1-p)/p)`` is justified only for the uniform
      repetition-code setting of arXiv:2405.07433v2, Theorem 10.
    - Surface-code cluster gap is confidence, not a calibrated failure
      probability.
    - Confidence uses the immutable final growth evidence from the same hard
      decode and does not rerun or reconstruct Union-Find.
    """

    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, base: UnionFindDecoder) -> None:
        if not isinstance(base, UnionFindDecoder):
            raise TypeError(
                "UnionFindClusterGapDecoder requires a UnionFindDecoder base"
            )
        self.base = base

    def run_seed_children(self):
        """Keep confidence transparent to the hard decoder's seed paths."""
        return self.base.run_seed_children()

    def latency(self, job: DecodeJob) -> int:
        """Confidence adds no simulated service latency."""
        return self.base.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Decode once, compute cluster gap, then publish confidence."""
        model = job.dem
        if model is None:
            return self.base.decode(job)

        import numpy as np

        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        observables = np.asarray(faults.observables)
        if observables.shape[0] != 1:
            raise ValueError(
                "Union-Find cluster confidence requires exactly one logical "
                f"observable, got {observables.shape[0]}"
            )
        if not np.any(observables[0]):
            raise ValueError(
                "Union-Find cluster confidence requires one nonzero logical "
                "observable row"
            )

        decoded_window = self.base.decode_with_growth_evidence(job)
        gap = _cluster_gap(decoded_window.hard_evidence)
        if not math.isfinite(gap):
            raise ValueError(
                "Union-Find cluster confidence could not reach an odd "
                "logical cycle"
            )
        soft_output = SoftOutput(
            gap=gap,
            source=UNION_FIND_CLUSTER_GAP_SOURCE,
        )
        decoded_window.hard_result.soft_output = soft_output
        return decoded_window.hard_result
