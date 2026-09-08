"""The QEC cycle clock: one syndrome round per cycle on every live patch.

The QPU runs syndrome extraction on every live patch every cycle, whether
or not an operation is using the patch: every measure qubit is read out
each cycle (Google, Suppressing quantum errors by scaling a surface code
logical qubit, 2207.06431; Google, Quantum error correction below the
surface code threshold, 2408.13687), and the extraction does not pause
for a patch no instruction is using, so that patch emits an idle round;
what those rounds cost the decoder is the idle policy's question, with
its sources in controller/policies.py. Operations start on a cycle
boundary and occupy whole cycles; a command that arrives on a boundary
starts on that boundary (QubiC, 2404.15260 Sec. IV: a pulse timestamp is
the time after which the pulse plays). This module owns that cadence
only; program dependencies, windows and decoder state live elsewhere.
"""

import dataclasses
from typing import Any, Callable, Optional

import decsim.engine
import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source

# Patches and operation ids are opaque identities chosen by the workload;
# Any stands for them in every signature below.


@dataclasses.dataclass(frozen=True)
class QPUCommandEvent:
    """When a command arrived at the QPU, and when it started."""

    # "ARRIVED" or "STARTED"; the event ledger reads these words.
    kind: str
    tick: int
    command: program_records.RunOperationBody


class QPUDevice:
    """Runs issued operation bodies on one QEC cycle clock.

    Every cycle emits one syndrome round per running operation and per
    idle patch. The receivers may arrive later through connect_*. Trace
    sources: command_event(QPUCommandEvent) when a command arrives and
    when it starts; round_emitted(readout) for every readout a round
    hands to the controller.
    """

    def __init__(
        self,
        engine: decsim.engine.Engine,
        syndrome_source: ports.SyndromeSource,
        cycle_ticks: int,
        readout_receiver: Optional[ports.ReadoutReceiver] = None,
        completion_receiver: Optional[
            Callable[[program_records.Operation], None]
        ] = None,
        idle_receiver: Optional[Callable[[Any, Any, int], None]] = None,
    ):
        self.engine = engine
        self.syndrome_source = syndrome_source
        self.cycle_ticks = cycle_ticks
        self.receivers = _Receivers(
            readout_receiver, completion_receiver, idle_receiver
        )
        self.trace = _TraceSources()
        self._live = _LiveOperations()

    def connect_readout_receiver(self, receiver: ports.ReadoutReceiver) -> None:
        """Wire the component that accepts every readout."""
        self.receivers.readout = receiver

    def connect_completion_receiver(
        self, receiver: Callable[[program_records.Operation], None]
    ) -> None:
        """Wire the callback for a completed operation body."""
        self.receivers.completion = receiver

    def connect_idle_receiver(
        self, receiver: Callable[[Any, Any, int], None]
    ) -> None:
        """Wire the callback for an idle patch's round."""
        self.receivers.idle = receiver

    def issue(self, command: program_records.RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""
        if command.round_ticks != self.cycle_ticks:
            raise ValueError("operation cadence must equal the QPU cycle")
        is_instant = command.round_count == 0
        emits_without_finalizing = (
            command.emits_detector_data and not command.finalizes_stream_round
        )
        if is_instant and emits_without_finalizing:
            raise ValueError(
                "zero-duration detector emitters must finalize a stream round"
            )
        event = QPUCommandEvent("ARRIVED", self.engine.now, command)
        self.trace.command_event.fire(event)
        self._live.commands_waiting.append(command)
        boundary = self.next_boundary()
        self._schedule_boundary(boundary)

    def finish(self) -> None:
        """The program is complete: idle patches stop after this cycle."""
        self._live.is_finished = True

    def next_boundary(self) -> int:
        """The cycle boundary at or after now, where issued operations start."""
        now = self.engine.now
        return self.boundary_at_or_after(now)

    def boundary_at_or_after(self, tick: int) -> int:
        """The first cycle boundary not earlier than the tick."""
        if tick < 0:
            raise ValueError("QPU boundary query tick must be nonnegative")
        if tick % self.cycle_ticks == 0:
            return tick
        whole_cycle_count = tick // self.cycle_ticks
        return (whole_cycle_count + 1) * self.cycle_ticks

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> None:
        """Produce and deliver one idle round of a live stream."""
        payloads = self.syndrome_source.idle_round_payloads(
            operation, stream_id, global_round, patch
        )
        self._deliver(payloads, operation)

    def emit_feedback_memory_round(
        self, operation_id: Any, patch: Any, round_index: int
    ) -> None:
        """Deliver the timing-only round of an idle patch."""
        payload = round_records.QPUReadout(
            ("idle", operation_id, patch), patch, round_index
        )
        route = round_records.SyndromePacketRoute.feedback_memory_round(
            operation_id
        )
        self.receivers.readout.accept_qpu_readout(payload, route)

    def _schedule_boundary(self, boundary: int) -> None:
        if boundary in self._live.scheduled_boundaries:
            return
        self._live.scheduled_boundaries.add(boundary)
        delay = boundary - self.engine.now
        cycle_number = boundary // self.cycle_ticks
        self.engine.schedule(
            delay, self._cross_boundary, label=f"qpu-cycle({cycle_number})"
        )

    def _cross_boundary(self) -> None:
        """One cycle boundary: the ended cycle's rounds, then the starts."""
        now = self.engine.now
        self._live.scheduled_boundaries.discard(now)
        if now > self._live.last_emitted_boundary:
            self._live.last_emitted_boundary = now
            self._emit_idle_rounds()
            self._emit_operation_rounds()
        self._start_waiting_commands()
        if self._live.is_finished:
            self._live.idle_by_patch.clear()
        has_running = bool(self._live.running_by_operation_id)
        has_idle = bool(self._live.idle_by_patch)
        has_waiting = bool(self._live.commands_waiting)
        is_live = has_running or has_idle
        if is_live or has_waiting:
            next_boundary = now + self.cycle_ticks
            self._schedule_boundary(next_boundary)

    def _emit_idle_rounds(self) -> None:
        idle_patches = self._live.idle_by_patch.items()
        for patch, idle in list(idle_patches):
            idle.emitted_round_count += 1
            self.receivers.idle(
                idle.operation_id, patch, idle.emitted_round_count
            )

    def _emit_operation_rounds(self) -> None:
        running_operations = self._live.running_by_operation_id.items()
        for operation_id, running in list(running_operations):
            running.emitted_round_count += 1
            command = running.command
            operation = command.operation
            if command.emits_detector_data:
                self.engine.log(
                    log_sources.QPU,
                    f"{operation.name} fires round "
                    f"{running.emitted_round_count}/{command.round_count}",
                )
                payloads = self.syndrome_source.round_payloads(
                    operation, running.emitted_round_count
                )
                self._deliver(payloads, operation)
            if running.emitted_round_count == command.round_count:
                del self._live.running_by_operation_id[operation_id]
                self._finish_command(command)

    def _start_waiting_commands(self) -> None:
        waiting = self._live.commands_waiting
        self._live.commands_waiting = []
        for command in waiting:
            self._start_command(command)

    def _start_command(self, command: program_records.RunOperationBody) -> None:
        operation = command.operation
        event = QPUCommandEvent("STARTED", self.engine.now, command)
        self.trace.command_event.fire(event)
        for patch in program_records.patches_of(operation):
            self._live.idle_by_patch.pop(patch, None)
        if command.round_count == 0:
            self._run_instant_command(command)
            return
        if command.emits_detector_data:
            self.syndrome_source.begin_operation(
                operation, command.round_count, command.source_round_count
            )
        running = _RunningOperation(command, 0)
        self._live.running_by_operation_id[operation.id] = running

    def _run_instant_command(
        self, command: program_records.RunOperationBody
    ) -> None:
        """A zero-round body only finalizes a stream round, then completes."""
        operation = command.operation
        if command.emits_detector_data:
            payloads = self.syndrome_source.finalize_stream_round(
                operation, command.source_round_count
            )
            self._deliver(payloads, operation)
        self._finish_command(command)

    def _finish_command(
        self, command: program_records.RunOperationBody
    ) -> None:
        operation = command.operation
        for patch in program_records.patches_of(operation):
            idle = _IdlePatch(operation.id, 0)
            self._live.idle_by_patch.setdefault(patch, idle)
        self.receivers.completion(operation)

    def _deliver(
        self,
        payloads: list[round_records.QPUReadout],
        operation: program_records.Operation,
    ) -> None:
        """Stamp every payload with its fragment slot and hand it on."""
        if not payloads:
            raise ValueError(
                "a detector-emitting round must emit at least one readout"
            )
        fragment_index = operation.syndrome_fragment_index
        declared_count = operation.syndrome_fragment_count
        if fragment_index is not None and len(payloads) != 1:
            raise ValueError(
                "an explicit syndrome fragment slot must emit one payload"
            )
        fragment_count = declared_count
        if declared_count is None:
            fragment_count = len(payloads)
        elif fragment_index is None and declared_count != len(payloads):
            raise ValueError(
                "declared syndrome fragment count must match emitted readouts"
            )
        for local_index, payload in enumerate(payloads):
            index = local_index
            if fragment_index is not None:
                index = fragment_index
            readout = dataclasses.replace(
                payload, fragment_count=fragment_count, fragment_index=index
            )
            self.trace.round_emitted.fire(readout)
            self.receivers.readout.accept_qpu_readout(
                readout, round_records.WINDOW_INPUT_ROUTE
            )


@dataclasses.dataclass
class _RunningOperation:
    """An operation body on the QPU and how many rounds it has emitted."""

    command: program_records.RunOperationBody
    emitted_round_count: int


@dataclasses.dataclass
class _IdlePatch:
    """A patch between operations and how many idle rounds it has emitted."""

    operation_id: object
    emitted_round_count: int


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the q p u device reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    command_event: trace_source.TraceSource = trace_source.new_source()
    round_emitted: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass
class _LiveOperations:
    """What the clock is running this cycle, and what it has emitted.

    running_by_operation_id and idle_by_patch are what a cycle emits a
    round for; commands_waiting are the bodies whose start the clock has
    not reached; scheduled_boundaries and last_emitted_boundary keep a
    cycle boundary from being scheduled or emitted twice; is_finished
    closes the clock once the workload is done. gem5 groups a
    component's many members the same way
    (tmp/resources/gem5/src/base/stats/group.hh:60-92).
    """

    running_by_operation_id: dict = dataclasses.field(default_factory=dict)
    idle_by_patch: dict = dataclasses.field(default_factory=dict)
    commands_waiting: list = dataclasses.field(default_factory=list)
    scheduled_boundaries: set = dataclasses.field(default_factory=set)
    last_emitted_boundary: int = 0
    is_finished: bool = False


@dataclasses.dataclass
class _Receivers:
    """The three components a cycle hands its output to.

    readout takes every syndrome round, completion hears an operation's
    body end, idle hears a round of a patch that is running nothing.
    Each may arrive after the clock is built, through connect_*, because
    the controller and the runtime are built after the QPU.
    """

    readout: Optional[ports.ReadoutReceiver] = None
    completion: Optional[Callable] = None
    idle: Optional[Callable] = None
