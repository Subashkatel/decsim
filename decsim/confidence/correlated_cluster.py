"""The correlated cluster gap: the cluster gap after the decode's own pairs.

Stim splits a fault that flips more than two detectors, a Y error or a
hook error, into graphlike pieces, and union-find prices each piece on
its own. Correlated matching (Fowler arXiv:1310.0863; PyMatching 2.4.0)
decodes once, shortens the partners of every piece the first answer
used, and decodes again. This row takes the first half of that and no
second decode: the partners of the pieces union-find's own correction
selected are shortened to their conditional lengths
(window_decoder.split_fault_partners), the growth the decode left is
kept on every edge, so a shortened edge's uncovered part is its new
length less what the growth covered, and the cluster gap (Meister et al.
2405.07433 Definition 9, cluster.py) is walked again on those lengths.

No paper describes this combination; it is decsim's own. Lengths are
only lowered, so the gap is never above the cluster gap of the same
decode, and the correction is the decode's, unchanged. Reading the
correction as faults that happened is a plug-in: the correction is a
function of the syndrome, not new evidence.
"""

import dataclasses
import time
from typing import Optional, Union

import numpy

import decsim.confidence.cluster as cluster
import decsim.config as config
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.ports as ports
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

CORRELATED_CLUSTER_GAP_SOURCE = decoding_records.SoftOutputSource(
    method="correlated_cluster_gap"
)


class CorrelatedClusterGap:
    """The signal row: the cluster gap with the correction's pairs shortened."""

    source = CORRELATED_CLUSTER_GAP_SOURCE
    fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
    decoder_evidence_requirement = decoding_records.CORRELATED_CLUSTER_EVIDENCE
    # the window is decoded once; the gap is read off that decode
    forced_logical_classes = ()
    evidence_refusal = (
        "the correlated cluster gap reads a union-find growth and the "
        "pairs of its split faults; use a union_find weak decoder with "
        "split_fault_partners on, or the cluster_gap confidence"
    )

    def __init__(
        self,
        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
        walk_microseconds: Optional[float] = None,
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)
        # what the reweighting and the walk cost on the weak tier's
        # clock: a card's declared number, or None to measure the call
        self.walk_microseconds = walk_microseconds

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """Its settings: walk_microseconds prices the walk, None measures it."""

        walk_microseconds: Optional[float] = None
        # the word the reports name this row by
        name = "correlated_cluster_gap"

        def __post_init__(self) -> None:
            if self.walk_microseconds is None:
                return
            config.check_microseconds(
                "walk_microseconds", self.walk_microseconds
            )

        def build(
            self,
            weak_algorithm: ports.DecoderSettings,
            threshold_nats: Optional[float],
        ) -> "CorrelatedClusterGap":
            """The row at the weak decoder's weight step, or the default one.

            A weak row with no weight step grows no clusters, and the
            build refuses that pairing.
            """
            del threshold_nats
            walk_microseconds = self.walk_microseconds
            weight_step = getattr(weak_algorithm, "weight_step", None)
            if weight_step is None:
                return CorrelatedClusterGap(walk_microseconds=walk_microseconds)
            return CorrelatedClusterGap(
                weight_step=weight_step, walk_microseconds=walk_microseconds
            )

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The gap in natural-log weight, and the ticks the walk cost.

        A decode that carried no growth gives None and zero; the policy
        escalates a window with no soft output.
        """
        evidence = solves[0].cluster_evidence
        if evidence is None:
            return decoding_records.SoftOutputComputation(None, 0)
        cluster.require_one_logical_row(evidence.graph)
        gap, ticks = self._walk(evidence)
        soft_output = decoding_records.SoftOutput(gap=gap, source=self.source)
        return decoding_records.SoftOutputComputation(soft_output, ticks)

    def _walk(self, evidence) -> tuple:
        """The gap and the ticks the walk cost, declared or measured."""
        started_ns = time.perf_counter_ns()
        gap_half_ticks = correlated_gap_half_ticks(evidence)
        finished_ns = time.perf_counter_ns()
        gap = cluster.gap_half_ticks_to_natural_log_weight(
            gap_half_ticks, self.weight_step
        )
        if self.walk_microseconds is not None:
            ticks = config.microseconds_to_ticks(self.walk_microseconds)
            return gap, ticks
        elapsed_ns = finished_ns - started_ns
        elapsed_microseconds = elapsed_ns / 1000.0
        ticks = config.microseconds_to_ticks(elapsed_microseconds)
        return gap, ticks


def correlated_gap_half_ticks(
    evidence: evidence_records.UnionFindHardEvidence,
) -> Union[int, float]:
    """The walk on the shortened lengths, in half ticks; inf for none."""
    graph = evidence.graph
    _endpoint_a, _endpoint_b, graph_lengths = graph.edge_arrays
    lengths = graph_lengths.copy()
    intervals = list(evidence.edge_intervals)
    selected = numpy.nonzero(evidence.selected_faults)
    for fault in selected[0]:
        for edge_index, half_ticks in graph.split_fault_partners[fault]:
            _shorten(lengths, intervals, edge_index, half_ticks)
    kept_intervals = tuple(intervals)
    return compiled_decoder.cluster_gap_at_lengths(
        graph, lengths, kept_intervals
    )


def _shorten(
    lengths: numpy.ndarray, intervals: list, edge_index: int, half_ticks: int
) -> None:
    """One edge at the shorter of its length and half_ticks, growth kept."""
    length = int(lengths[edge_index])
    if half_ticks >= length:
        return
    lengths[edge_index] = half_ticks
    interval = intervals[edge_index]
    intervals[edge_index] = _kept_growth(interval, length, half_ticks)


def _kept_growth(interval, length: int, shorter: int):
    """The interval on the shorter edge, each front where it was.

    The front from the edge's first end stays at its tick, the front
    from the second end keeps its distance from that end, and an edge
    the two fronts now meet on is closed.
    """
    if isinstance(interval, evidence_records.Closed):
        return interval
    grown_from_second_end = length - interval.upper_tick
    upper_tick = shorter - grown_from_second_end
    if upper_tick <= interval.lower_tick:
        return evidence_records.Closed()
    return evidence_records.Open(interval.lower_tick, upper_tick)
