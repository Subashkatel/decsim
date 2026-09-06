"""The listeners of one run, by name.

The Machine builds each listener the observation section asks for,
connects it to the sources it hears, and keeps it here so the front,
the gate and the experiments read a run's numbers from its listeners
and never from a component. The log writer, the window ledger, the
runtime stamps, the queue depth, the controller counters and the
command events are always there; the ones a study asks for are None
when the section did not ask.
"""

import dataclasses
from typing import Optional

import decsim.observe.command_events as command_events_module
import decsim.observe.controller_counters as controller_counters_module
import decsim.observe.decode_records as decode_records_module
import decsim.observe.log_writers as log_writers
import decsim.observe.metrics as metrics
import decsim.observe.queue_depth as queue_depth_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.window_ledger as window_ledger_module


@dataclasses.dataclass(frozen=True)
class Observation:
    """What one run's listeners heard."""

    log: log_writers.LogWriter
    windows: window_ledger_module.WindowLedger
    decode_records: Optional[decode_records_module.DecodeRecordLedger]
    runtime_stamps: runtime_stamps_module.RuntimeStamps
    queue_depth: queue_depth_module.QueueDepthLog
    controller_counters: controller_counters_module.ControllerCounters
    command_events: command_events_module.CommandEvents
    decode_backlog: Optional[metrics.DecodeBacklog]
    decoder_utilization: Optional[metrics.DecoderUtilization]
    decoder_memory_occupancy: Optional[metrics.DecoderMemoryOccupancy]
