"""Consensus: the share of an ensemble's members voting with its majority.

Harmony (Shutty, Newman and Villalonga 2401.12434 Sec. V) observes that
low consensus among an ensemble flags a high-risk decode, and uses "the
fraction of decoders that vote with the majority" as a confidence
score: its layered decoding sends a window to a second, more accurate
pass "if and only if the first pass ensemble fails to yield a consensus
prediction". The ensemble row (decoders/ensembles.py) puts that
fraction on its answer; this row reports it as the window's soft
output, a share between 0 and 1, not a gap in nats, so an edge decides
on it with the agreement verdict (escalation/verdicts.py).
"""

import dataclasses
from typing import Optional

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

CONSENSUS_SOURCE = decoding_records.SoftOutputSource(method="consensus")


class Consensus:
    """The signal row that reads an ensemble's agreement."""

    source = CONSENSUS_SOURCE
    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.CONSENSUS_EVIDENCE
    # the window is decoded once, by every member of the ensemble
    forced_logical_classes = ()
    evidence_refusal = (
        "consensus reads an ensemble's agreement, and only the ensemble "
        "row (decoders/ensembles.py) reports one"
    )

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The consensus row, which takes no setting."""

        name = "consensus"

        def build(
            self,
            weak_algorithm: ports.DecoderSettings,
            threshold_nats: Optional[float],
        ) -> "Consensus":
            """The signal, whatever the threshold: the verdict reads it."""
            del weak_algorithm, threshold_nats
            return Consensus()

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The ensemble's agreement as the soft output, at no cost."""
        agreement = solves[0].agreement
        if agreement is None:
            return decoding_records.SoftOutputComputation(None, 0)
        soft_output = decoding_records.SoftOutput(
            gap=agreement, source=CONSENSUS_SOURCE
        )
        return decoding_records.SoftOutputComputation(soft_output, 0)
