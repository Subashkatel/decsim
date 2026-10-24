"""The listeners of one run, by name.

Experiments and the gate read a run's numbers from its listeners, never
from a component. A listener a study did not ask for is None.
"""

import dataclasses
from typing import Optional

import decsim.observe.answer_checks as answer_checks_module
import decsim.observe.checkpoints as checkpoints_module
import decsim.observe.command_events as command_events_module
import decsim.observe.controller_counters as controller_counters_module
import decsim.observe.data_movement as data_movement_module
import decsim.observe.decode_records as decode_records_module
import decsim.observe.frame_corrections as frame_corrections_module
import decsim.observe.link_traffic as link_traffic
import decsim.observe.log_writers as log_writers
import decsim.observe.metrics as metrics
import decsim.observe.referee_audit as referee_audit_module
import decsim.observe.region_of_interest as region_of_interest_module
import decsim.observe.result_ledger as result_ledger_module
import decsim.observe.round_events as round_events_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.sampled_shots as sampled_shots_module
import decsim.observe.stage_records as stage_records_module
import decsim.observe.trace_writer as trace_writer_module
import decsim.observe.window_ledger as window_ledger_module
import decsim.observe.window_outcomes as window_outcomes_module
import decsim.statistics.registry as statistics_registry


@dataclasses.dataclass(frozen=True)
class Observation:
    """What one run's listeners heard, and what its parts measured.

    statistics is the run's registry of the statistics each part
    declares (decsim/statistics), read as rows or by part path.
    """

    log: log_writers.LogWriter
    windows: window_ledger_module.WindowLedger
    results: result_ledger_module.ResultLedger
    traffic: link_traffic.TrafficLedger
    frame_corrections: frame_corrections_module.FrameCorrections
    trace_writer: Optional[trace_writer_module.TraceWriter]
    data_movement: Optional[data_movement_module.DataMovement]
    decode_records: Optional[decode_records_module.DecodeRecordLedger]
    runtime_stamps: runtime_stamps_module.RuntimeStamps
    controller_counters: controller_counters_module.ControllerCounters
    command_events: command_events_module.CommandEvents
    stages: stage_records_module.StageLedger
    decode_backlog: Optional[metrics.DecodeBacklog]
    statistics: statistics_registry.Registry
    round_events: round_events_module.RoundEventRecorder
    referee_audit: referee_audit_module.RefereeAudit
    sampled_shots: sampled_shots_module.SampledShots
    window_outcomes: Optional[window_outcomes_module.WindowOutcomes]
    confidence: Optional[decode_records_module.ConfidenceLedger]
    # None unless the run leaves a warm-up out of its statistics
    region_of_interest: Optional[region_of_interest_module.RegionOfInterest] = (
        None
    )
    # None unless the run closes checkpoints of its streams
    checkpoints: Optional[checkpoints_module.Checkpoints] = None
    # None unless the run checks each window's two answers
    answer_checks: Optional[answer_checks_module.AnswerChecks] = None
