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
from typing import Any

import decsim.config as config
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
    idle patch; the source that produces its bits and the three ends
    that output reaches are ports. A readout leaves at the tick its
    syndrome source names
    (SyndromeSource.readout_departure_tick), never before an earlier
    readout of one of its patches: it waits behind that one, as gem5's
    packet queue with forceOrder schedules a packet after the last one
    to the same address rather than ahead of it
    (gem5 src/mem/packet_queue.cc:134-147), and a tick
    before the readout is refused, as that queue asserts
    (packet_queue.cc:114). Trace sources: command_event(QPUCommandEvent)
    when a command arrives and when it starts; round_emitted(readout)
    for every readout a round produces, at the boundary it is read out;
    and round_event(RoundEvent) with kind EMITTED when a readout leaves
    for the controller, on the readout path's own ledger. The event is
    the emitter's own: gem5's SimObject reports its statistics from the
    object the event happened in
    (gem5 src/base/stats/group.hh:60-92).
    """

    # the device model that produces each round's bits, shared with the
    # detection event former and the window models
    syndrome_source = ports.Port(ports.SyndromeSource)
    readout_receiver = ports.Port(ports.ReadoutReceiver)
    # the end that owns an operation's life: it hears the body's last
    # round on this clock
    runtime = ports.Port(ports.OperationRuntime)
    # a patch nobody is operating on still emits a round every cycle,
    # and what that round costs is the idle policy's question
    idle_rounds = ports.Port(ports.IdleRoundReceiver)

    def __init__(
        self,
        engine: decsim.engine.Engine,
        clock: config.Clock,
        code: ports.CodeModel,
    ):
        self.engine = engine
        self.clock = clock
        self.code = code
        self.trace = _TraceSources()
        self._live = _LiveOperations()

    def issue(self, command: program_records.RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""
        if command.round_ticks != self.clock.period_ticks:
            raise RuntimeError("operation cadence must equal the QPU cycle")
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
        return self.clock.edge(0, tick)

    def are_patches_idle(self, operation_id: Any, patches: tuple) -> bool:
        """Every group member is idle after the same completed operation."""
        for patch in patches:
            idle = self._live.idle_by_patch.get(patch)
            if idle is None:
                return False
            if idle.operation_id != operation_id:
                return False
        return True

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
    ) -> None:
        """Produce and deliver one idle round of a live stream."""
        payloads = self.syndrome_source.idle_round_payloads(
            operation,
            stream_id,
            global_round,
            is_final=is_final,
            round_period_ticks=self.clock.period_ticks,
        )
        self._deliver(payloads, operation)

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """The source's own answer: it alone saw the rounds it executed."""
        self.syndrome_source.validate_stream_length(
            stream_operation, stream_round_count
        )

    def emit_feedback_memory_round(
        self, operation_id: Any, patch: Any, round_index: int
    ) -> None:
        """Deliver the timing-only round of an idle patch.

        It carries no values but is as wide as the patch's syndrome: the
        extraction reads every measure qubit every cycle whether or not an
        instruction uses the patch (Google 2207.06431 lines 118-125, "All
        stabilisers are measured in this manner concurrently"), so every
        wire and memory the round crosses can price it.
        """
        size_bits = self.code.syndrome_bits_per_round(1)
        payload = round_records.QPUReadout(
            ("idle", operation_id, patch),
            (patch,),
            round_index,
            size_bits=size_bits,
        )
        route = round_records.SyndromePacketRoute.feedback_memory_round(
            operation_id
        )
        self._send(payload, route)

    def _schedule_boundary(self, boundary: int) -> None:
        if boundary in self._live.scheduled_boundaries:
            return
        self._live.scheduled_boundaries.add(boundary)
        delay = boundary - self.engine.now
        cycle_number = boundary // self.clock.period_ticks
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
            next_boundary = self.clock.edge(1, now)
            self._schedule_boundary(next_boundary)

    def _emit_idle_rounds(self) -> None:
        idle_patches = self._live.idle_by_patch.items()
        for patch, idle in list(idle_patches):
            idle.emitted_round_count += 1
            self.idle_rounds.emit_idle_round(
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
        event = QPUCommandEvent("STARTED", self.engine.now, command)
        self.trace.command_event.fire(event)
        command = self.idle_rounds.start_command(command)
        operation = command.operation
        for patch in program_records.patches_of(operation):
            self._live.idle_by_patch.pop(patch, None)
        if command.round_count == 0:
            self._run_instant_command(command)
            return
        if command.emits_detector_data:
            self.syndrome_source.begin_operation(
                operation,
                command.round_count,
                command.source_round_count,
                round_period_ticks=self.clock.period_ticks,
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
        self.runtime.body_done(operation)

    def _deliver(
        self,
        payloads: list[round_records.QPUReadout],
        operation: program_records.Operation,
    ) -> None:
        """Stamp every payload with its fragment slot and hand it on."""
        if not payloads:
            raise RuntimeError(
                "a detector-emitting round must emit at least one readout"
            )
        fragment_count, first_index = _fragment_slots(operation, len(payloads))
        for local_index, payload in enumerate(payloads):
            index = first_index + local_index
            readout = dataclasses.replace(
                payload, fragment_count=fragment_count, fragment_index=index
            )
            self.trace.round_emitted.fire(readout)
            self._send(readout, round_records.WINDOW_INPUT_ROUTE)

    def _send(self, readout: round_records.QPUReadout, route) -> None:
        """Hand the readout on at its departure tick, behind its patches'.

        A readout that leaves now, with no readout of its patches still
        waiting, is handed on at once, so a source that names the
        boundary sends in the order the cycle produced its readouts.
        """
        now = self.engine.now
        stated_tick = self.syndrome_source.readout_departure_tick(readout, now)
        if stated_tick < now:
            raise RuntimeError(
                f"the syndrome source sends readout {readout.round_index} "
                f"of operation {readout.operation_id!r} at tick "
                f"{stated_tick}, before tick {now} it was read out at"
            )
        waiting_until = self._last_departure_of(readout.patch_ids)
        if stated_tick == now and waiting_until < now:
            self._hand_to_the_controller(readout, route)
            return
        departure_tick = max(stated_tick, waiting_until)
        for patch in readout.patch_ids:
            self._live.departure_tick_by_patch[patch] = departure_tick
        delay = departure_tick - now
        self.engine.schedule(
            delay,
            lambda: self._hand_to_the_controller(readout, route),
            label=f"qpu-readout-departs({readout.round_index})",
        )

    def _last_departure_of(self, patches: tuple) -> int:
        """The latest tick a waiting readout of these patches leaves at.

        -1 when none of them has ever waited.
        """
        departure_ticks = [-1]
        for patch in patches:
            tick = self._live.departure_tick_by_patch.get(patch, -1)
            departure_ticks.append(tick)
        return max(departure_ticks)

    def _hand_to_the_controller(self, readout, route) -> None:
        """The readout leaves the QPU: report the instant, then hand it on."""
        emitted = round_records.RoundEvent.of(
            "EMITTED",
            self.engine.now,
            readout.operation_id,
            readout.round_index,
            route,
            readout.patch_ids,
        )
        self.trace.round_event.fire(emitted)
        self.readout_receiver.accept_qpu_readout(readout, route)


def _fragment_slots(operation, payload_count: int) -> tuple[int, int]:
    first_index = operation.syndrome_fragment_index
    fragment_count = operation.syndrome_fragment_count
    if fragment_count is None:
        fragment_count = payload_count
    if first_index is None:
        if fragment_count != payload_count:
            raise ValueError(
                "declared syndrome fragment count must match emitted readouts"
            )
        return fragment_count, 0
    after_last_index = first_index + payload_count
    if after_last_index > fragment_count:
        raise ValueError(
            "readout group exceeds the declared syndrome fragment count"
        )
    return fragment_count, first_index


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
    """Every event the QPU device reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    command_event: trace_source.TraceSource = trace_source.new_source()
    round_emitted: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass
class _LiveOperations:
    """What the clock is running this cycle, and what it has emitted.

    running_by_operation_id and idle_by_patch are what a cycle emits a
    round for; commands_waiting are the bodies whose start the clock has
    not reached; scheduled_boundaries and last_emitted_boundary keep a
    cycle boundary from being scheduled or emitted twice; is_finished
    closes the clock once the workload is done; departure_tick_by_patch
    is the tick the latest readout of a patch that had to wait leaves
    at, so a later one queues behind it. gem5 groups a
    component's many members the same way
    (gem5 src/base/stats/group.hh:60-92).
    """

    running_by_operation_id: dict = dataclasses.field(default_factory=dict)
    idle_by_patch: dict = dataclasses.field(default_factory=dict)
    commands_waiting: list = dataclasses.field(default_factory=list)
    scheduled_boundaries: set = dataclasses.field(default_factory=set)
    last_emitted_boundary: int = 0
    is_finished: bool = False
    departure_tick_by_patch: dict = dataclasses.field(default_factory=dict)
