"""One tier's event-detection logic: it forms the rounds that tier reads.

Under controller.detection_events_formed_at decoder the rounds reach a
tier raw and that tier converts them, which is where two of the three
decoder-side papers put the work: LILLIPUT "generates error detection
events by comparing the stabilizer measurement outcomes from two
consecutive QEC cycles ... This step is accomplished by the Event
Detection Logic block shown in Figure 4", inside the decoder (2108.06569
lines 499-510), and Yang et al. fix "The total latency of the
preprocessing stage for syndrome calculation ... at 20 ns (5 FPGA clock
cycles)" and count it inside their decoder subtotal (2605.04892 lines
1274-1275, Table I lines 1049-1052).

Each tier has its own: the weak tier reads Buffer 0 and the strong tier
reads Buffer 1, two copies of the same raw rounds in two stores, each
with its own logic in front of its own decoder core. So a round two
windows of one tier read is formed once and charged once, and a round
both tiers read is charged twice, once on each tier's engine clock. The
value is the former's (detector_error_model/detection_event_formation.py
RememberedDetectionEvents); the rounds this tier is charged for are the
tier's own.
"""

import dataclasses
from typing import Optional

import decsim.decoders.staged_decoder as staged_decoder
import decsim.detector_error_model.detection_event_formation as formation
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records


class TierFormation:
    """The rounds one decoder tier forms, and what it is charged for them.

    A job's rounds are formed where they land on the tier, which is the
    first demand for them; the rounds it is charged for are the ones no
    earlier job of this tier had already formed, frozen on the job at
    the first ask so the stage that prices the formation and the
    dispatcher that predicts the unit's compute read one number.
    """

    def __init__(self, former: ports.DetectionEventFormer) -> None:
        self.former = former
        self.formed_round_keys: set = set()

    def form(self, payloads: list) -> list:
        """One job's rounds, their detection events in place of outcomes."""
        formed = []
        for fragment in payloads:
            converted = self._formed_fragment(fragment)
            formed.append(converted)
        return formed

    def rounds_to_form(self, job: decoding_records.DecodeJob) -> tuple:
        """The round keys this job forms on this tier, frozen at first ask."""
        frozen = job.detection_event_rounds
        if frozen is not None:
            return frozen
        fresh = []
        for key in _round_keys_of(job):
            if key in self.formed_round_keys:
                continue
            fresh.append(key)
        self.formed_round_keys.update(fresh)
        frozen = tuple(fresh)
        job.detection_event_rounds = frozen
        return frozen

    def _formed_fragment(
        self, fragment: round_records.RetainedSyndromeFragment
    ) -> round_records.RetainedSyndromeFragment:
        """One round's fragment carrying its events at their own width."""
        if fragment.bits is None:
            return fragment
        events = self.former.form_round(
            fragment.operation_id, fragment.round_index, fragment.bits
        )
        replaced = dataclasses.replace(fragment, bits=events)
        return formation.sized_by_its_bits(replaced)


@dataclasses.dataclass(frozen=True)
class DetectionEventFormationStage(staged_decoder.DecoderStage):
    """The tier's event-detection logic, priced in front of its core.

    Yang et al. 2605.04892 lines 1274-1275 fix the preprocessing stage
    for syndrome calculation at 20 ns, 5 FPGA clock cycles, and their
    Table I (lines 1049-1052) counts "Syndrome calculation 20" inside
    "Subtotal (decoder) 148": the cost is the decoder's own, per round
    it forms. The stage prices the rounds this job forms on this tier
    and no others, so a window that overlaps an earlier one pays for the
    rounds the earlier window did not bring.
    """

    formation: Optional[TierFormation] = None

    def cycles_for(self, job: decoding_records.DecodeJob) -> int:
        """The stage's cycles: per round this job forms on this tier."""
        rounds = self.formation.rounds_to_form(job)
        return self.cycles_per_round * len(rounds)

    def formed_round_keys(self, job: decoding_records.DecodeJob) -> tuple:
        """The rounds this stage forms for the job, for the trace."""
        return self.formation.rounds_to_form(job)


def _round_keys_of(job: decoding_records.DecodeJob) -> tuple:
    """The (operation, round) identities one job reads, in round order.

    The payloads before the input lands in a unit's memory, the landed
    input afterwards; a job that carries neither reads no rounds.
    """
    keys = []
    for payload in job.payloads:
        key = (payload.operation_id, payload.round_index)
        if key not in keys:
            keys.append(key)
    if keys:
        return tuple(keys)
    decoder_input = job.decoder_input
    if decoder_input is None:
        return ()
    for round_input in decoder_input.rounds:
        keys.append((round_input.operation_id, round_input.round_index))
    return tuple(keys)
