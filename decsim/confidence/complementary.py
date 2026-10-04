"""The complementary gap: the confidence of one window's two forced solves.

g_comp = |w(class 1) - w(class 0)| (Toshio et al. 2510.25222 Sec. III A,
lines 482-494, the method of Gidney et al. 2312.04522): the window is
decoded twice, each solve pinned to one logical class, and a small gap
is a decoder unsure which class it is in. The solves are the weak
decoder's own (DecodeJob.forced_logical_class); this row subtracts.
"""

import dataclasses
import math
from typing import Optional

import decsim.config as config
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

COMPLEMENTARY_GAP_SOURCE = decoding_records.SoftOutputSource(
    method="complementary_gap"
)
# the two logical classes a window's solves are pinned to, in the order
# the window side submits them
FORCED_LOGICAL_CLASSES = (0, 1)


@dataclasses.dataclass(frozen=True)
class ComplementaryGap:
    """The signal row: the gap between one window's two forced solves.

    The subtraction costs nothing unless walk_microseconds prices it on
    the weak unit.
    """

    walk_microseconds: Optional[float] = None
    source = COMPLEMENTARY_GAP_SOURCE
    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.FORCED_CLASS_SOLVES
    forced_logical_classes = FORCED_LOGICAL_CLASSES
    evidence_refusal = (
        "a decoder that does not find the minimum-weight correction "
        "inside a fixed logical class reports a weight that cannot be "
        "compared across classes, and a virtual detector wrecks a "
        "cluster-growing decoder's locality (Lee et al. "
        "arXiv:2510.05795 Sec. 2.1.1); use the cluster_gap confidence, "
        "or a matching weak decoder"
    )

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """Its settings: walk_microseconds prices the subtraction, or None."""

        walk_microseconds: Optional[float] = None
        # the word the reports name this row by
        name = "complementary_gap"

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
        ) -> "ComplementaryGap":
            """The row priced by the card; the weak row says nothing to it."""
            del weak_algorithm
            del threshold_nats
            return ComplementaryGap(walk_microseconds=self.walk_microseconds)

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The gap of one window's forced solves, and the ticks it is priced at.

        The gap is None when a weight is missing: a model that pins no
        observable has no forced solve, and the policy escalates the
        window.
        """
        soft_output = self._gap_of(solves)
        ticks = self._walk_ticks()
        return decoding_records.SoftOutputComputation(soft_output, ticks)

    def _walk_ticks(self) -> int:
        """The ticks the card prices this row's computation at."""
        if self.walk_microseconds is None:
            return 0
        return config.microseconds_to_ticks(self.walk_microseconds)

    def _gap_of(self, solves: tuple) -> Optional[decoding_records.SoftOutput]:
        """|w(class 1) - w(class 0)|, or None when a weight is missing.

        A class no correction reaches weighs +inf, so the gap to it is
        +inf and the decoder is certain of the other class; with no
        class reachable there is no ratio and no gap.
        """
        weights = []
        for solve in solves:
            if solve.forced_class_weight is None:
                return None
            weights.append(solve.forced_class_weight)
        decoded_class_weight = min(weights)
        if decoded_class_weight == math.inf:
            return None
        complementary_class_weight = max(weights)
        gap = complementary_class_weight - decoded_class_weight
        return decoding_records.SoftOutput(
            gap=gap,
            source=COMPLEMENTARY_GAP_SOURCE,
            decoded_class_weight=decoded_class_weight,
            complementary_class_weight=complementary_class_weight,
        )
