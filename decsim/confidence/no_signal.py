"""No confidence: an edge whose verdict reads none of the decode's gaps.

A verdict that decides on the backend's reported status, or keeps or
escalates every window (escalation/verdicts.py status, always, never),
reads no soft output, so the source pool need not produce any evidence:
a belief-propagation pool escalates the windows its backend did not
converge on to an OSD pool, as BP+OSD runs OSD only then (Roffe et al.
2005.07016). The row asks for nothing and reports no gap; a verdict that
reads the threshold is refused it at build (build/escalation.py).
"""

import dataclasses
from typing import Optional

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

NO_SIGNAL_SOURCE = decoding_records.SoftOutputSource(method="none")


class NoSignal:
    """The signal row that reads nothing and reports no gap."""

    source = NO_SIGNAL_SOURCE
    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = frozenset()
    # the window is decoded once, and nothing is read off the decode
    forced_logical_classes = ()
    evidence_refusal = ""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The none row, which takes no setting."""

        # the word the reports name this row by
        name = "none"

        def build(
            self,
            weak_algorithm: ports.DecoderSettings,
            threshold_nats: Optional[float],
        ) -> "NoSignal":
            """The signal, whatever the decoder and the threshold."""
            del weak_algorithm, threshold_nats
            return NoSignal()

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """No soft output, at no cost."""
        del solves
        return decoding_records.SoftOutputComputation(None, 0)
