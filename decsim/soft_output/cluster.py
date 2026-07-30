"""Cluster-gap confidence from one weighted Union-Find hard decode."""

from __future__ import annotations

import heapq
import itertools
import math

from ..detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from ..message import DecodeJob, DecodeResult, SoftOutput, SoftOutputSource
from ..union_find_decoder.decoder import UnionFindDecoder
from ..union_find_decoder.window_decoder import (
    Closed,
    Open,
    UnionFindGraph,
    UnionFindHardEvidence,
)


UNION_FIND_CLUSTER_GAP_SOURCE = SoftOutputSource(
    method="cluster_gap",
    cluster_origin="union_find_decoder",
    growth_schedule="weighted_global_fair",
    gap_units="decibels",
    correction="none",
    references=(
        "arXiv:2004.04693 Section II",
        "arXiv:2405.07433v2 Definition 9 / Algorithm 2",
        "arXiv:2510.25222v1 Section II-C",
    ),
)


def _quotient_cluster_gap(
    graph: UnionFindGraph,
    edge_intervals: tuple[Open | Closed, ...],
) -> float:
    """Return the shortest odd-logical quotient walk in natural-log units."""
    adjacency: dict[object, list[tuple[object, float, int]]] = {}

    def add_segment(left, right, weight: float, parity: int) -> None:
        adjacency.setdefault(left, []).append((right, weight, parity))
        adjacency.setdefault(right, []).append((left, weight, parity))

    for edge_index, (edge, interval) in enumerate(
        zip(graph.edges, edge_intervals)
    ):
        if isinstance(interval, Closed):
            coordinates = (0.0, edge.weight)
        else:
            coordinates = tuple(
                sorted({0.0, interval.lower, interval.upper, edge.weight})
            )
        path_nodes: list[object] = [edge.detector_a]
        path_nodes.extend(
            ("union_find_edge", edge_index, split_index)
            for split_index in range(1, len(coordinates) - 1)
        )
        path_nodes.append(edge.detector_b)
        for segment_index, (lower, upper) in enumerate(
            zip(coordinates, coordinates[1:])
        ):
            covered = isinstance(interval, Closed) or (
                upper <= interval.lower or lower >= interval.upper
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
            if distance != distances.get(state, math.inf):
                continue
            if distance >= best:
                break
            if state == target_state:
                best = distance
                break
            graph_node, logical_parity = state
            for neighbor, weight, edge_parity in adjacency.get(graph_node, ()):
                neighbor_state = (
                    neighbor,
                    logical_parity ^ edge_parity,
                )
                candidate = distance + weight
                if candidate < distances.get(neighbor_state, math.inf):
                    distances[neighbor_state] = candidate
                    heapq.heappush(
                        pending,
                        (candidate, next(sequence), neighbor_state),
                    )
    return best


def _cluster_gap(hard_evidence: UnionFindHardEvidence) -> float:
    natural_gap = _quotient_cluster_gap(
        hard_evidence.graph,
        hard_evidence.edge_intervals,
    )
    return natural_gap * 10.0 / math.log(10.0)


class UnionFindClusterGapDecoder:
    """Attach one-logical cluster-gap confidence to a hard Union-Find result.

    SCOPE:
    - Confidence requires exactly one nonzero logical-observable row.
    - The public gap is in decibels; hard growth uses natural-log-odds lengths.
    - The exact likelihood-ratio interpretation applies only to the uniform
      repetition-code setting of arXiv:2405.07433v2, Theorem 10.
    - Surface-code cluster gap is confidence, not a calibrated failure
      probability or a general Union-Find likelihood bound.
    - Confidence consumes immutable intervals from the same hard decode.
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
        decoded_window.hard_result.soft_output = SoftOutput(
            gap=_cluster_gap(decoded_window.hard_evidence),
            source=UNION_FIND_CLUSTER_GAP_SOURCE,
        )
        return decoded_window.hard_result
