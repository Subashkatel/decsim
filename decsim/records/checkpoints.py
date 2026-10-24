"""A stream as if its final readout came after a round: a checkpoint.

Tan et al. measure a run's logical error rate at several lengths from
one run: at a length they copy the state, treat the window there as the
last, read the copy out and decode it, and let the run go on unchanged
(2209.09219 lines 1003-1037). Fowler reads a round out perfectly and
undoes it (1110.5133 lines 322-331). The source that holds the state
closes such a copy with a noiseless readout; this record carries what
the copy read out, and the readers end the decoding there.
"""

import dataclasses
from typing import Any

import decsim.records.circuits as circuit_records


@dataclasses.dataclass(frozen=True)
class CheckpointSnapshot:
    """One stream after one round, closed by a noiseless readout.

    readout_events are the readout's detection events in detector
    order, the detectors the readout adds to its round. observable_flips
    is each observable's flip over the stream up to that readout: the
    truth a decoding that ended there is held to. At the stream's own
    last round (is_final) both are the real readout's. program is the
    stream's, whose fragments give the readers the window models of a
    stream that ends at that round.
    """

    stream_id: Any  # an opaque identity
    round_index: int
    readout_events: tuple[int, ...]
    observable_flips: tuple[int, ...]
    is_final: bool
    program: circuit_records.RepeatedStimCircuit
