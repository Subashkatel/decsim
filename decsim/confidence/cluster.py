"""The cluster gap: the confidence of one weighted Union-Find window decode.

Meister et al. 2405.07433 Definition 9 and Algorithm 2 (lines 518-536):
the weighted edge intervals the hard decode grew are quotiented into a
graph whose shortest closed walk of odd logical parity is the gap, in
the growth's half-tick units, reported as natural-log weight, which is
the unit the switching threshold is held in. The hard decode is Delfosse
and Nickerson 1709.06218 (union_find/window_decoder.py), and this row
reads the growth that decode already returned on its result
(DecodeResult.cluster_evidence), so the confidence is the decoder's own
and no second decode is run. The signal needs one ordinary decode of the
window, not a forced-class pair, so its forced_logical_classes is empty.

The walk is not free. Algorithm 2 is a Dijkstra over the decode's own
edge intervals, one search per node of the quotient graph, which is the
larger half of a switching window's host time when it runs in Python;
it runs in C beside the decode that produced the growth
(decoders/union_find/cluster_gap.c) and this row reports what it cost:
measured on the host clock the way a measured decoder is, or the
declared number of a tier that is a card. The evidence and its reader
are the same hardware, so the time is charged on the unit that produced
the growth (decision D8; Toshio et al. 2510.25222 lines 152-160 compute
the soft output on the weak decoder).

The exact likelihood-ratio reading of the gap holds only in the uniform
repetition-code setting of Meister's Theorem 10; on a surface code it is
a confidence, not a calibrated failure probability. The gap needs one
nonzero logical-observable row, and thresholds are calibrated per
weight step.
"""

import math
import time
from fractions import Fraction
from typing import Optional, Union

import decsim.config as config
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records


def union_find_cluster_gap_source(
    weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
) -> decoding_records.SoftOutputSource:
    """The source of a cluster gap at one absolute natural-log weight step."""
    normalized_step = evidence_records.normalized_weight_step(weight_step)
    return decoding_records.SoftOutputSource(
        method="cluster_gap",
        cluster_origin="union_find_decoder",
        growth_schedule="weighted_global_fair",
        gap_units="log_likelihood_weight",
        correction="none",
        weight_step_natural_log=normalized_step,
        references=("cluster-gap method",),
    )


class ClusterGap:
    """The signal row: the gap of one cluster-based decode's own growth."""

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.CLUSTER_GROWTH_EVIDENCE
    # the window is decoded once; the gap is read off that decode
    forced_logical_classes = ()
    evidence_refusal = (
        "the cluster gap walks the radii a growth left behind, and "
        "PyMatching's API reports no regions, blossoms or radii "
        "(pymatching 2.4.0 Matching), while belief propagation and "
        "search decoders grow no clusters at all; use "
        "weak_decoder.kind union_find, or escalation.confidence "
        "complementary_gap"
    )

    def __init__(
        self,
        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
        walk_microseconds: Optional[float] = None,
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)
        self.source = union_find_cluster_gap_source(self.weight_step)
        # what the walk costs on the weak tier's clock: a card's declared
        # number, or None to measure the call as a measured decoder is
        self.walk_microseconds = walk_microseconds

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The gap of the window's one decode, and what the walk cost.

        The gap is in natural-log weight, and None when the decode
        carried no growth: a job without a window model runs no growth,
        and the escalation policy escalates a window without a soft
        output (escalation/policies.py). A window with no growth walks
        nothing and charges nothing.
        """
        evidence = solves[0].cluster_evidence
        if evidence is None:
            return decoding_records.SoftOutputComputation(None, 0)
        graph = evidence.graph
        _require_one_logical_row(graph)
        _require_weight_step(graph, self.weight_step)
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


def _require_one_logical_row(graph) -> None:
    """The gap is defined for exactly one nonzero logical-observable row."""
    row_count = graph.logical_observable_count
    if row_count != 1:
        raise ValueError(
            "Union-Find cluster confidence requires exactly one logical "
            f"observable, got {row_count}"
        )
    for edge in graph.edges:
        if edge.logical_observables[0]:
            return
    raise ValueError(
        "Union-Find cluster confidence requires one nonzero logical "
        "observable row"
    )


def _require_weight_step(graph, weight_step: float) -> None:
    """The gap is read off the ticks the growth actually used."""
    if graph.weight_step == weight_step:
        return
    raise RuntimeError(
        "the cluster gap reads the decode's own ticks: the decoder grew "
        f"at weight step {graph.weight_step} and the signal reports "
        f"{weight_step}"
    )


def _cluster_gap(
    hard_evidence: evidence_records.UnionFindHardEvidence, weight_step: float
) -> float:
    gap_half_ticks = compiled_decoder.cluster_gap(
        hard_evidence.graph, hard_evidence.edge_intervals
    )
    return _gap_half_ticks_to_natural_log_weight(gap_half_ticks, weight_step)


def _gap_half_ticks_to_natural_log_weight(
    gap_half_ticks: Union[int, float], weight_step: float
) -> float:
    """Half ticks of the growth as natural-log weight, exactly, rounded once.

    Every signal reports its gap in the units the switching threshold is
    held in (escalation.gap_threshold_db is converted once, at the yaml
    boundary, by escalation/settings.py decibels_to_nats), so a threshold
    means the same thing whichever signal a run names.
    """
    if gap_half_ticks == math.inf:
        return math.inf
    half_ticks = Fraction(gap_half_ticks, 2)
    step = Fraction.from_float(weight_step)
    exact_gap_nats = half_ticks * step
    try:
        return float(exact_gap_nats)
    except OverflowError:
        return math.inf
