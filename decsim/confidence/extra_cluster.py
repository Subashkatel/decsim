"""The extra-cluster gap: a Union-Find decode's confidence, by growing on.

Kishi et al. 2602.03336 Algorithm 1 (lines 478-491): after the decode,
every cluster and the boundary grow half a weight step a tick, clusters
that meet fuse, and the gap is the growth spent when the two boundaries
join; it is infinite when the growth limit comes first. Theorem 2
(532-548): whenever the cluster gap is at or under the limit, this gap
is defined and no larger, so no window the cluster gap would escalate is
missed; Theorem 1 (510-530): when defined, it is at most the cluster
gap. The limit is the escalation threshold (345-347), so the build
refuses a threshold source with no number.

The growth is the decoder's own grow and merge loop run on
(union_find.c, union_find_extra_growth), the paper's point (145-149).
decsim's graph has one boundary node, so the two boundaries meeting is
an edge closing a walk of odd logical parity, as in cluster_gap.c.

The unit grows whole ticks, so the gap is a whole number of weight steps
and can exceed the continuous value by up to half a step; a cluster gap
exactly at the threshold may read as kept.
"""

import dataclasses
import math
import time
from typing import Optional

import decsim.confidence.cluster as cluster
import decsim.config as config
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.ports as ports
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records

# union_find_extra_growth takes the limit as an int64_t (union_find.h),
# and ctypes wraps a larger Python int without a word, which would turn
# a huge threshold into a small or negative limit and keep every window
_LARGEST_TICK_COUNT = 2**63 - 1


EXTRA_CLUSTER_GAP_SOURCE = decoding_records.SoftOutputSource(
    method="extra_cluster_gap"
)


def growth_limit_ticks(growth_limit_nats: float, weight_step: float) -> int:
    """The ticks that reach the limit: a tick grows a cluster half a step.

    Two fronts close a weight step a tick between them, so the limit is
    ceil(limit / step) ticks; the ceiling keeps Theorem 2.
    """
    ticks = growth_limit_nats / weight_step
    limit_ticks = math.ceil(ticks)
    if limit_ticks > _LARGEST_TICK_COUNT:
        raise ValueError(
            f"the extra-cluster gap's growth limit of {limit_ticks} ticks "
            "passes the unit's 64-bit tick counter; lower the threshold "
            "or raise the weak decoder's weight_step"
        )
    return limit_ticks


class ExtraClusterGap:
    """The signal row: the growth spent before the boundaries join."""

    source = EXTRA_CLUSTER_GAP_SOURCE
    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.CLUSTER_GROWTH_EVIDENCE
    # the window is decoded once; the growth goes on from that decode
    forced_logical_classes = ()
    evidence_refusal = (
        "the extra-cluster gap grows the clusters a decode left on, and "
        "PyMatching's API reports no regions, blossoms or radii "
        "(pymatching 2.4.0 Matching), while belief propagation and "
        "search decoders grow no clusters at all; use a union_find weak "
        "decoder, or the complementary_gap confidence"
    )

    def __init__(
        self,
        growth_limit_nats: float,
        timing: cycle_count_module.Timing,
        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
        walk_microseconds: Optional[float] = None,
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)
        self.growth_limit_ticks = growth_limit_ticks(
            growth_limit_nats, self.weight_step
        )
        # what the growth costs on the weak unit: the card's number when
        # given, else the weak row's timing
        self.walk_microseconds = walk_microseconds
        self.timing = timing

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """Its settings: walk_microseconds prices the growth, or None.

        None charges the weak row's own timing: its loop's cycle count
        on the unit that decoded (cycle_count.py), or the host's
        measured time.
        """

        walk_microseconds: Optional[float] = None
        # the word the reports name this row by
        name = "extra_cluster_gap"

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
        ) -> "ExtraClusterGap":
            """The row grown to the threshold, on the weak row's unit.

            A weak row with no weight step grows no clusters, and the
            build refuses that pairing.
            """
            walk_microseconds = self.walk_microseconds
            weight_step = getattr(weak_algorithm, "weight_step", None)
            if weight_step is None:
                host_time = cycle_count_module.HostMeasuredTime()
                return ExtraClusterGap(
                    threshold_nats,
                    host_time,
                    walk_microseconds=walk_microseconds,
                )
            return ExtraClusterGap(
                threshold_nats,
                weak_algorithm.timing,
                weight_step=weight_step,
                walk_microseconds=walk_microseconds,
            )

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The gap in natural-log weight, and the ticks the growth cost.

        The gap is infinite when the limit came before the join; a
        decode that carried no growth gives None and zero.
        """
        evidence = solves[0].cluster_evidence
        if evidence is None:
            return decoding_records.SoftOutputComputation(None, 0)
        graph = evidence.graph
        cluster.require_one_logical_row(graph)
        gap, ticks = self._grow_on(evidence)
        soft_output = decoding_records.SoftOutput(gap=gap, source=self.source)
        return decoding_records.SoftOutputComputation(soft_output, ticks)

    def _grow_on(self, evidence) -> tuple:
        """The gap and the ticks the growth cost: the card, or the timing."""
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
        elapsed_ns = finished_ns - started_ns
        steps = outcome.growth_steps
        ticks = self.timing.extra_growth_ticks(steps, elapsed_ns)
        return gap, ticks

    def _gap_of(self, outcome: compiled_decoder.ExtraGrowthOutcome) -> float:
        """The growth spent as natural-log weight: a tick is one step."""
        if outcome.joined_at_tick is None:
            return math.inf
        half_ticks = 2 * outcome.joined_at_tick
        return cluster.gap_half_ticks_to_natural_log_weight(
            half_ticks, self.weight_step
        )
