"""The listeners of one run, by name.

The Machine builds each listener the observation section asks for,
connects it to the sources it hears, and keeps it here so the front,
the gate and the experiments read a run's numbers from its listeners
and never from a component. The log writer is always there; the rest
are None when the section did not ask.
"""

import dataclasses
from typing import Optional

import decsim.observe.log_writers as log_writers
import decsim.observe.metrics as metrics


@dataclasses.dataclass(frozen=True)
class Observation:
    """What one run's listeners heard."""

    log: log_writers.LogWriter
    decode_backlog: Optional[metrics.DecodeBacklog]
    decoder_utilization: Optional[metrics.DecoderUtilization]
    decoder_memory_occupancy: Optional[metrics.DecoderMemoryOccupancy]
