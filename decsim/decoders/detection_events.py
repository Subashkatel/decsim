"""One tier's event-detection logic: it forms the rounds that tier reads.

When detection_events.formed_at seats the former at a decoder unit, the
rounds reach that tier raw and it converts them, where two of the three
decoder-side papers put the work: LILLIPUT's Event Detection Logic block
sits inside the decoder (2108.06569 lines 499-510), and Yang et al.
count their 20 ns preprocessing stage inside their decoder subtotal
(2605.04892 lines 1273-1275, Table I lines 1049-1052).

Each tier is its own seat with its own history, so a round two windows
of one tier read is formed and charged once, and a round both tiers read
is formed and charged by each
(detector_error_model/detection_event_formation.py).
"""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.decoders.staged_decoder as staged_decoder
import decsim.ports as ports
import decsim.records.decoding as decoding_records


class TierFormation:
    """The rounds one decoder tier forms, and what it is charged for them.

    A job's rounds are formed where they land on the tier. It is charged
    for the ones no earlier job of this tier formed, frozen on the job
    at the first ask, so the stage that prices the formation and the
    dispatcher that predicts the unit's compute read one number. A job
    cancelled before its decode starts gives that claim back. Every job
    claims its rounds when staged, while they are still held for it, so
    the seat may forget a claim once its round retires. A source with no
    recipes leaves the rounds as they landed, charged all the same.
    """

    def __init__(
        self, placement: ports.DetectionEventPlacement, seat: str
    ) -> None:
        self.placement = placement
        self.seat = seat

    def form(self, payloads: list, rounds_before: tuple = ()) -> list:
        """One job's rounds, their detection events in place of outcomes.

        rounds_before are the raw rounds before the first, which the tier
        holds for that round's detectors and does not return.
        """
        fragments = tuple(payloads)
        formed = self.placement.form_at(self.seat, fragments, rounds_before)
        return list(formed)

    def cycles_for(self, job: decoding_records.DecodeJob) -> int:
        """The seat's cycles for the rounds this job forms on this tier."""
        rounds = self.rounds_to_form(job)
        round_count = len(rounds)
        return self.placement.cycles_at(self.seat, round_count)

    def rounds_to_form(self, job: decoding_records.DecodeJob) -> tuple:
        """The round keys this job forms on this tier, frozen at first ask."""
        frozen = job.detection_event_rounds
        if frozen is not None:
            return frozen
        round_keys = _round_keys_of(job)
        frozen = self.placement.claim_rounds(self.seat, round_keys)
        job.detection_event_rounds = frozen
        return frozen

    def release(self, job: decoding_records.DecodeJob) -> None:
        """A cancelled job gives the rounds it claimed back to this tier.

        The formation stage charges a job when its decode starts, so a
        job cancelled before that was never charged and the next job
        that reads those rounds pays; a started job keeps its claim.
        """
        if job.service_started:
            return
        claimed = job.detection_event_rounds
        if claimed is None:
            return
        self.placement.return_claim(self.seat, claimed)
        job.detection_event_rounds = None


@dataclasses.dataclass(frozen=True)
class DetectionEventFormationStage(staged_decoder.DecoderStage):
    """The tier's event-detection logic, priced in front of its core.

    A pipelined stage: the seat's fixed latency once and its rate for
    every round after the first, on the former's own clock
    (detector_error_model/settings.py). It prices only the rounds this
    job forms on this tier, so a window overlapping an earlier one pays
    for the rounds the earlier one did not bring.
    """

    formation: Optional[TierFormation] = None

    def cycles_for(self, job: decoding_records.DecodeJob) -> int:
        """Latency once, then the rate for each round after the first."""
        return self.formation.cycles_for(job)

    def priced_on(self, unit_clock: config.Clock) -> config.Clock:
        """The former's clock; the unit's when the former charges nothing."""
        clock = self.formation.placement.clock
        if clock is None:
            return unit_clock
        return clock

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
