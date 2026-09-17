"""The extra-cluster gap: a Union-Find decode's confidence, by growing on.

Kishi et al. 2602.03336 Algorithm 1 (lines 478 to 491 of the text):
after the decode, every cluster and the boundary grow by half a weight
step a tick, clusters that meet fuse, and after every tick the unit
asks whether the two boundaries have joined. The gap is the growth
spent when they do, and undefined, here infinite, when the growth limit
is reached first. Theorem 2 (532-548): whenever the cluster gap is at or
under the limit this gap is defined and no larger, so no window the
cluster gap would escalate is missed; Theorem 1 (510-530): whenever it
is defined it is at most the cluster gap. The limit is the escalation
threshold itself (345-347), which is why a threshold source with no
number when the machine is built is refused.

The growth is the decoder's own grow and merge loop run on
(union_find.c, union_find_extra_growth), which is the paper's point
(145-149: the soft output "can directly reuse the cluster growth module
of the decoder"). decsim's graph has one boundary node, so "b1 meets
b2" is an edge closing a walk of odd logical parity, the reading of
Meister's quotient that cluster_gap.c also takes. The cost is the
loop's own cycle count on the unit that decoded (cycle_count.py,
extra_growth_cycles) when the weak row has one, a card's number when
the yaml prices it, and the host clock otherwise, as the cluster gap
(decision D8).

The unit grows whole ticks, so the gap it reports is a whole number of
weight steps and can exceed the continuous value by up to half a step;
a cluster gap exactly at the threshold may therefore read as kept.
"""

import math
import time
from typing import Optional

import decsim.confidence.cluster as cluster
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.escalation.settings as escalation_settings
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records


def union_find_extra_cluster_gap_source(
    weight_step: float,
) -> decoding_records.SoftOutputSource:
    """The source of an extra-cluster gap at one natural-log weight step."""
    normalized_step = evidence_records.normalized_weight_step(weight_step)
    return decoding_records.SoftOutputSource(
        method="extra_cluster_gap",
        cluster_origin="union_find_decoder",
        growth_schedule="weighted_global_fair",
        gap_units="log_likelihood_weight",
        correction="none",
        weight_step_natural_log=normalized_step,
        references=("extra-cluster gap without cluster graph",),
    )


def growth_limit_ticks(growth_limit_nats: float, weight_step: float) -> int:
    """The ticks that reach the limit: a tick grows a cluster half a step.

    Two fronts close a weight step a tick between them, so the gap after
    t ticks is t weight steps and the limit is ceil(limit / step) ticks;
    the ceiling keeps Theorem 2, since every cluster gap at or under
    the limit joins within that many ticks.
    """
    if not math.isfinite(growth_limit_nats) or growth_limit_nats < 0:
        raise ValueError(
            "the extra-cluster gap's growth limit must be a finite "
            f"nonnegative natural-log weight, got {growth_limit_nats!r}"
        )
    ticks = growth_limit_nats / weight_step
    return math.ceil(ticks)


class ExtraClusterGap:
    """The signal row: the growth spent before the boundaries join."""

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.CLUSTER_GROWTH_EVIDENCE
    # the window is decoded once; the growth goes on from that decode
    forced_logical_classes = ()
    evidence_refusal = (
        "the extra-cluster gap grows the clusters a decode left on, and "
        "PyMatching's API reports no regions, blossoms or radii "
        "(pymatching 2.4.0 Matching), while belief propagation and "
        "search decoders grow no clusters at all; use "
        "weak_decoder.kind union_find, or escalation.confidence "
        "complementary_gap"
    )

    def __init__(
        self,
        growth_limit_nats: float,
        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
        walk_microseconds: Optional[float] = None,
        cycle_count: Optional[cycle_count_module.CycleCount] = None,
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)
        self.source = union_find_extra_cluster_gap_source(self.weight_step)
        self.growth_limit_ticks = growth_limit_ticks(
            growth_limit_nats, self.weight_step
        )
        # what the growth costs on the weak unit: the card's number, the
        # unit's own cycle count, or the host clock when neither is given
        self.walk_microseconds = walk_microseconds
        self.cycle_count = cycle_count

    @classmethod
    def from_settings(
        cls,
        escalation: escalation_settings.EscalationSettings,
        weak_decoder: decoder_settings.DecoderSettings,
    ) -> "ExtraClusterGap":
        """The row grown to the threshold, on the weak row's unit."""
        threshold_nats = escalation.gap_threshold_nats
        if threshold_nats is None:
            raise ValueError(
                "escalation.confidence extra_cluster_gap grows to the "
                "threshold, so it needs the threshold as one number when "
                "the machine is built (Kishi 2602.03336 Sec. III, the "
                "early-stopping threshold is the switching threshold): "
                "give gap_threshold_db, or threshold_source table; "
                "threshold_source online has no fixed number"
            )
        return cls(
            threshold_nats,
            weight_step=weak_decoder.weight_step,
            walk_microseconds=escalation.confidence_walk_microseconds,
            cycle_count=weak_decoder.cycle_count,
        )

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The window's gap by growing on, and what the growth cost.

        The gap is in natural-log weight, infinite when the limit came
        before the join, and None when the decode carried no growth, in
        which case the escalation policy escalates the window
        (escalation/policies.py) and nothing is charged.
        """
        evidence = solves[0].cluster_evidence
        if evidence is None:
            return decoding_records.SoftOutputComputation(None, 0)
        graph = evidence.graph
        cluster.require_one_logical_row(graph)
        cluster.require_weight_step(graph, self.weight_step)
        gap, ticks = self._grow_on(evidence)
        soft_output = decoding_records.SoftOutput(gap=gap, source=self.source)
        return decoding_records.SoftOutputComputation(soft_output, ticks)

    def _grow_on(self, evidence) -> tuple:
        """The gap and the ticks the growth cost: card, count or measured."""
        started_ns = time.perf_counter_ns()
        outcome = compiled_decoder.extra_growth(
            evidence.graph,
            evidence.edge_intervals,
            evidence.residual_syndrome,
            self.growth_limit_ticks,
        )
        finished_ns = time.perf_counter_ns()
        gap = self._gap_of(outcome)
        if self.walk_microseconds is not None:
            ticks = config.microseconds_to_ticks(self.walk_microseconds)
            return gap, ticks
        if self.cycle_count is not None:
            ticks = self.cycle_count.extra_growth_ticks(outcome.growth_steps)
            return gap, ticks
        elapsed_ns = finished_ns - started_ns
        elapsed_microseconds = elapsed_ns / 1000.0
        ticks = config.microseconds_to_ticks(elapsed_microseconds)
        return gap, ticks

    def _gap_of(self, outcome: compiled_decoder.ExtraGrowthOutcome) -> float:
        """The growth spent as natural-log weight: a tick is one step."""
        if outcome.joined_at_tick is None:
            return math.inf
        half_ticks = 2 * outcome.joined_at_tick
        return cluster.gap_half_ticks_to_natural_log_weight(
            half_ticks, self.weight_step
        )
