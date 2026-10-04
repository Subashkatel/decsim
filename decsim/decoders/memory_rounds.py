"""The decoder side's end for a timing-only round that lands here.

A feedback-memory round carries no syndrome a decoder reads: it is an
idle patch's round, travelling so that its buffer slot, the link and the
decoder's stream stage are charged for it (controller/idle_rounds.py).
The stage is real in both stream decoders: LILLIPUT decodes a block of
rounds as the stream fills it (2108.06569) and Yang et al. run one stage
per round of the readout stream (2605.04892). Nothing is deposited in a
unit's memory; this end counts the round, logs it on the decoder side's
line and tells the window side.
"""

from typing import Any

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.log_sources as log_sources


class MemoryRoundArrivals:
    """The primary decoder's receiving end for timing-only memory rounds."""

    windows = ports.Port(ports.WindowInput)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        # operation id -> the timing-only rounds of it that landed here
        self.landed_by_operation: dict = {}

    def receive_memory_round(
        self,
        source_operation_id: Any,  # an opaque identity
    ) -> None:
        """Take one timing-only round: count it, then tell the windows."""
        landed = self.landed_by_operation.get(source_operation_id, 0)
        landed += 1
        self.landed_by_operation[source_operation_id] = landed
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"memory round for op {source_operation_id} "
            f"(idle buffer rounds: {landed})",
        )
        self.windows.accept_feedback_memory_round(source_operation_id)
