"""One tier's event-detection logic: it forms the rounds that tier reads.

When detection_events.formed_at seats the former at a decoder unit, the
rounds reach that tier raw and it converts them, which is where two of
the three decoder-side papers put the work: LILLIPUT "generates error
detection events by comparing the stabilizer measurement outcomes from
two consecutive QEC cycles ... This step is accomplished by the Event
Detection Logic block shown in Figure 4", inside the decoder (2108.06569
lines 499-510), and Yang et al. keep "All variables ... in FPGA
registers, enabling fully pipelined operation" and fix "The total
latency of the preprocessing stage for syndrome calculation ... at 20 ns
(5 FPGA clock cycles)", counted inside their decoder subtotal
(2605.04892 lines 1273-1275, Table I lines 1049-1052).

Each tier is its own seat (weak_decoder, strong_decoder) with its own
history, so a round two windows of one tier read is formed once and
charged once, and a round both tiers read is formed and charged by
each. The values are the seat's
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

    A job's rounds are formed where they land on the tier, which is the
    first demand for them; the rounds it is charged for are the ones no
    earlier job of this tier had already formed, frozen on the job at
    the first ask so the stage that prices the formation and the
    dispatcher that predicts the unit's compute read one number. A job
    that is cancelled before its decode starts gives that claim back,
    since no stage of it ever ran. The claims live at the seat
    (claim_rounds), which forgets one once its round retires: every job
    claims its rounds when it is staged, while they are still held for
    it, so none asks for a round after it retires. A source with no
    recipes leaves the rounds as they landed, and the tier is charged
    for them all the same.
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

        A job is charged for its rounds by the formation stage, which
        runs when its decode starts; a job cancelled or withdrawn before
        that was never charged, so its claim goes back and the next job
        that reads those rounds pays for forming them. A job whose
        decode had started keeps its claim: its stage charged it.
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

    A pipelined stage, so its cycles are the seat's fixed latency once
    and its rate for every round after the first, on the former's own
    clock (detection_events, detector_error_model/settings.py). The
    stage prices the rounds this job forms on this tier and no others,
    so a window that overlaps an earlier one pays for the rounds the
    earlier window did not bring.
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
