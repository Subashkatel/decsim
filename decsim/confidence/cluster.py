"""The cluster gap: the confidence of one weighted Union-Find window decode.

Meister et al. 2405.07433 Definition 9 and Algorithm 2 (lines 518-536):
the gap is the shortest closed walk of odd logical parity through the
edge intervals the hard decode grew (Delfosse and Nickerson 1709.06218),
in half ticks of growth, reported as natural-log weight, the unit the
switching threshold is held in. It reads the growth the decode returned
(DecodeResult.cluster_evidence), so no second decode runs. The walk runs
in C (union_find/cluster_gap.c), and its time is charged on the unit
that produced the growth (Toshio et al. 2510.25222 lines 152-160).

The likelihood-ratio reading holds only in the repetition-code setting
of Meister's Theorem 10; on a surface code the gap is a confidence, not
a calibrated failure probability. A growth with no edge across the one
logical row admits no odd walk, so its gap is infinite.
"""

import dataclasses
import fractions
import math
import time
from typing import Optional, Union

import decsim.config as config
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.ports as ports
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records

CLUSTER_GAP_SOURCE = decoding_records.SoftOutputSource(method="cluster_gap")


class ClusterGap:
    """The signal row: the gap of one cluster-based decode's own growth."""

    source = CLUSTER_GAP_SOURCE
    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.CLUSTER_GROWTH_EVIDENCE
    # the window is decoded once; the gap is read off that decode
    forced_logical_classes = ()
    evidence_refusal = (
        "the cluster gap walks the radii a growth left behind, and "
        "PyMatching's API reports no regions, blossoms or radii "
        "(pymatching 2.4.0 Matching), while belief propagation and "
        "search decoders grow no clusters at all; use a union_find weak "
        "decoder, or the complementary_gap confidence"
    )

    def __init__(
        self,
        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
        walk_microseconds: Optional[float] = None,
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)
        # what the walk costs on the weak tier's clock: a card's declared
        # number, or None to measure the call as a measured decoder is
        self.walk_microseconds = walk_microseconds

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """Its settings: walk_microseconds prices the walk, None measures it."""

        walk_microseconds: Optional[float] = None
        # the word the reports name this row by
        name = "cluster_gap"

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
        ) -> "ClusterGap":
            """The row at the weak decoder's weight step, or the default one.

            A weak row with no weight step grows no clusters, and the
            build refuses that pairing.
            """
            del threshold_nats
            walk_microseconds = self.walk_microseconds
            weight_step = getattr(weak_algorithm, "weight_step", None)
            if weight_step is None:
                return ClusterGap(walk_microseconds=walk_microseconds)
            return ClusterGap(
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
        graph = evidence.graph
        require_one_logical_row(graph)
        gap, ticks = self._walk(evidence)
        soft_output = decoding_records.SoftOutput(gap=gap, source=self.source)
        return decoding_records.SoftOutputComputation(soft_output, ticks)

    def _walk(self, evidence) -> tuple:
        """The gap and the ticks the walk cost, declared or measured."""
        if self.walk_microseconds is not None:
            gap = _cluster_gap(evidence, self.weight_step)
            ticks = config.microseconds_to_ticks(self.walk_microseconds)
            return gap, ticks
        started_ns = time.perf_counter_ns()
        gap = _cluster_gap(evidence, self.weight_step)
        finished_ns = time.perf_counter_ns()
        elapsed_ns = finished_ns - started_ns
        elapsed_microseconds = elapsed_ns / 1000.0
        ticks = config.microseconds_to_ticks(elapsed_microseconds)
        return gap, ticks


def require_one_logical_row(graph) -> None:
    """The gap is defined for exactly one logical-observable row."""
    row_count = graph.logical_observable_count
    if row_count != 1:
        raise ValueError(
            "Union-Find cluster confidence requires exactly one logical "
            f"observable, got {row_count}"
        )


def gap_half_ticks_to_natural_log_weight(
    gap_half_ticks: Union[int, float], weight_step: float
) -> float:
    """Half ticks of the growth as natural-log weight, exactly, rounded once.

    Every signal reports in the threshold's unit, so one threshold means
    the same thing whichever signal a run names.
    """
    if gap_half_ticks == math.inf:
        return math.inf
    half_ticks = fractions.Fraction(gap_half_ticks, 2)
    step = fractions.Fraction.from_float(weight_step)
    exact_gap_nats = half_ticks * step
    try:
        return float(exact_gap_nats)
    except OverflowError:
        return math.inf


def _cluster_gap(
    hard_evidence: evidence_records.UnionFindHardEvidence, weight_step: float
) -> float:
    gap_half_ticks = compiled_decoder.cluster_gap(
        hard_evidence.graph, hard_evidence.edge_intervals
    )
    return gap_half_ticks_to_natural_log_weight(gap_half_ticks, weight_step)
