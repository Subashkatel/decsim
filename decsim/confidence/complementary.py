"""The complementary gap: the confidence of one window's two forced solves.

g_comp = |w(class 1) - w(class 0)| (Toshio et al. 2510.25222 Sec. III A,
lines 482-494, the method of Gidney et al. 2312.04522): the window is
decoded twice, each solve pinned to one logical class, and the two
minimum weights are subtracted. A small gap is a decoder unsure which
class it is in. The two solves are the weak decoder's own
(DecodeJob.forced_logical_class, decsim/ports.py Decoder), so this row
names the signal, says what it needs a decoder to answer, and does the
subtraction.
"""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records

COMPLEMENTARY_GAP_SOURCE = decoding_records.SoftOutputSource(
    method="complementary_gap",
    cluster_origin="mwpm_opposite_logical",
    growth_schedule="minimum_weight_matching",
    gap_units="log_likelihood_weight",
    correction="opposite_logical_constraint",
    weight_step_natural_log=None,
    references=("complementary-gap method",),
)
# the two logical classes a window's solves are pinned to, in the order
# the window side submits them
FORCED_LOGICAL_CLASSES = (0, 1)


@dataclasses.dataclass(frozen=True)
class ComplementaryGap:
    """The signal row: the gap between one window's two forced solves.

    walk_microseconds is what this row's own computation costs on the
    weak unit; None is free, because the computation is one subtraction
    of two numbers the decodes already reported (decision D8).
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
        "arXiv:2510.05795 Sec. 2.1.1); use escalation.confidence "
        "cluster_gap, or a matching weak decoder"
    )

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The gap between the weights of one window's forced solves.

        The gap is None when a weight is missing: a window whose model
        pins no observable has no forced solve, and the escalation
        policy then escalates it (escalation/policies.py). The
        computation is one subtraction of two numbers the decodes
        already reported, so it charges no time unless the yaml prices
        it (decision D8).
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
        """|w(class 1) - w(class 0)|, or None when a weight is missing."""
        weights = []
        for solve in solves:
            if solve.forced_class_weight is None:
                return None
            weights.append(solve.forced_class_weight)
        decoded_class_weight = min(weights)
        complementary_class_weight = max(weights)
        gap = complementary_class_weight - decoded_class_weight
        return decoding_records.SoftOutput(
            gap=gap,
            source=COMPLEMENTARY_GAP_SOURCE,
            decoded_class_weight=decoded_class_weight,
            complementary_class_weight=complementary_class_weight,
        )
