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
    """The signal row: the gap between one window's two forced solves."""

    source = COMPLEMENTARY_GAP_SOURCE
    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    forced_logical_classes = FORCED_LOGICAL_CLASSES

    def soft_output_for(
        self, forced_class_weights
    ) -> Optional[decoding_records.SoftOutput]:
        """The gap of one window's forced-class weights.

        None when a weight is missing: a window whose model pins no
        observable has no forced solve, and the escalation policy then
        escalates it (escalation/policies.py).
        """
        for weight in forced_class_weights:
            if weight is None:
                return None
        decoded_class_weight = min(forced_class_weights)
        complementary_class_weight = max(forced_class_weights)
        gap = complementary_class_weight - decoded_class_weight
        return decoding_records.SoftOutput(
            gap=gap,
            source=COMPLEMENTARY_GAP_SOURCE,
            decoded_class_weight=decoded_class_weight,
            complementary_class_weight=complementary_class_weight,
        )
