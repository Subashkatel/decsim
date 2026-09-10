"""The decoder side's end for a timing-only round that lands here.

A feedback-memory round carries no syndrome a decoder reads: it is an
idle patch's round travelling so that Buffer 0's slot, the link and the
decoder's own stream stage are charged for it
(controller/idle_rounds.py). The stage is real in both stream decoders
read for it, which take every round of the stream as it arrives, idle or
not: LILLIPUT decodes on a block of rounds as the stream fills it
(2108.06569) and Yang et al. run one stage per round of the readout
stream (2605.04892). So this hop has a far end after all, and it is
here: nothing is deposited in a unit's memory, and what the end does is
count the round it was handed, narrate it on the decoder side's line and
tell the window side, which keeps its own count for window readiness.
"""

import decsim.records.log_sources as log_sources


class MemoryRoundArrivals:
    """The decoders' end of weak_buffer_to_weak_decoder for a memory round."""

    def __init__(self, engine, windows) -> None:
        self.engine = engine
        self.windows = windows
        # operation id -> the timing-only rounds of it that landed here
        self.landed_by_operation: dict = {}

    def receive_memory_round(self, source_operation_id) -> None:
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
