"""Execute controller commands and emit QPU readout events on one QEC cycle clock.

The QPU runs syndrome extraction on every live patch every cycle, whether or
not an operation is using the patch (Google 2207.06431 / 2408.13687: every
measure qubit is read out each cycle; SWIPER device_manager.py
``_generate_syndrome_round``: active instructions emit one round per patch and
every other active patch emits an idle round). Operations start on a cycle
boundary and consume whole cycles. This module owns that cadence only; program
dependencies, windowing and decoder state live elsewhere.
"""
from __future__ import annotations
from dataclasses import replace

from ..message import (
    RunOperationBody, QPUReadout, SyndromePacketRoute, WINDOW_INPUT_ROUTE,
)


class QPUDevice:
    """Run issued operations on the configured physical device model."""

    def __init__(self, engine, model, cycle_ticks: int, readout_receiver=None,
                 completion_receiver=None, idle_receiver=None):
        self.engine = engine
        self.model = model
        self.cycle_ticks = cycle_ticks
        self.readout_receiver = readout_receiver
        self.completion_receiver = completion_receiver
        self.idle_receiver = idle_receiver
        self.cycle = 0                 # cycles completed
        self._pending = []             # commands waiting for the next boundary
        self._running = {}             # op.id -> [command, rounds emitted]
        self._idle = {}                # patch -> [operation_id, idle rounds emitted]
        self._finished = False
        self._scheduled = set()        # boundary ticks already scheduled
        self._emitted_boundary = 0     # last boundary whose rounds were emitted

    def connect_readout_receiver(self, receiver) -> None:
        self.readout_receiver = receiver

    def connect_completion_receiver(self, receiver) -> None:
        self.completion_receiver = receiver

    def connect_idle_receiver(self, receiver) -> None:
        self.idle_receiver = receiver

    # ------------------------------------------------------------ commands

    def issue(self, command: RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""
        if command.round_ticks != self.cycle_ticks:
            raise ValueError("operation cadence must equal the QPU cycle")
        if command.round_count == 0 and command.emits_detector_data \
                and not command.finalizes_stream_round:
            raise ValueError("zero-duration detector emitters must finalize a stream round")
        self._pending.append(command)
        self._arm()

    def finish(self) -> None:
        """The program is complete: idle patches stop after the current cycle."""
        self._finished = True

    # --------------------------------------------------------------- clock

    def next_boundary(self) -> int:
        """The cycle boundary at or after now, where issued operations start."""
        now = self.engine.now
        return now if now % self.cycle_ticks == 0 else \
            (now // self.cycle_ticks + 1) * self.cycle_ticks

    def _arm(self, boundary=None) -> None:
        if boundary is None:
            boundary = self.next_boundary()
        if boundary in self._scheduled:
            return
        self._scheduled.add(boundary)
        self.engine.schedule(boundary - self.engine.now, self._tick,
                             label=f"qpu-cycle({boundary // self.cycle_ticks})")

    def _tick(self) -> None:
        """One cycle boundary: rounds of the cycle just ended, then starts."""
        now = self.engine.now
        self._scheduled.discard(now)
        if now > self._emitted_boundary:
            self._emitted_boundary = now
            self.cycle = now // self.cycle_ticks
            self._emit_idle_rounds()
            self._emit_operation_rounds()
        self._start_pending()
        if self._finished:
            self._idle.clear()
        if self._running or self._idle or self._pending:
            self._arm(now + self.cycle_ticks)

    def _emit_idle_rounds(self) -> None:
        for patch, state in list(self._idle.items()):
            state[1] += 1
            self.idle_receiver(state[0], patch, state[1])

    def _emit_operation_rounds(self) -> None:
        for op_id, state in list(self._running.items()):
            command, done = state
            state[1] = done + 1
            op = command.operation
            if command.emits_detector_data:
                self.engine.log("QPU", f"{op.name} fires round {state[1]}/{command.round_count}")
                self._emit(self.model.round_payloads(op, state[1]), op)
            if state[1] == command.round_count:
                del self._running[op_id]
                self._body_done(command)

    def _start_pending(self) -> None:
        pending, self._pending = self._pending, []
        for command in pending:
            op = command.operation
            for patch in _patches(op):
                self._idle.pop(patch, None)
            if command.round_count == 0:
                if command.emits_detector_data:
                    self._emit(self.model.finalize_stream_round(op, command.source_round_count), op)
                self._body_done(command)
                continue
            if command.emits_detector_data:
                self.model.begin_operation(op, command.round_count, command.source_round_count)
            self._running[op.id] = [command, 0]

    def _body_done(self, command: RunOperationBody) -> None:
        self._release_patches(command)
        self.completion_receiver(command.operation)

    def _release_patches(self, command: RunOperationBody) -> None:
        for patch in _patches(command.operation):
            self._idle.setdefault(patch, [command.operation.id, 0])

    # ---------------------------------------------------------------- emit

    def _emit(self, payloads, operation) -> None:
        if not payloads:
            raise ValueError("a detector-emitting round must emit at least one readout")
        if operation.syndrome_fragment_index is not None and len(payloads) != 1:
            raise ValueError("an explicit syndrome fragment slot must emit one payload")
        count = (operation.syndrome_fragment_count
                 if operation.syndrome_fragment_count is not None
                 else len(payloads))
        if (
            operation.syndrome_fragment_index is None
            and operation.syndrome_fragment_count is not None
            and operation.syndrome_fragment_count != len(payloads)
        ):
            raise ValueError(
                "declared syndrome fragment count must match emitted readouts")
        for local_index, payload in enumerate(payloads):
            index = operation.syndrome_fragment_index if operation.syndrome_fragment_index is not None else local_index
            self.readout_receiver.accept_qpu_readout(
                replace(payload, n_fragments=count, fragment_index=index),
                WINDOW_INPUT_ROUTE,
            )

    def emit_idle_stream_round(self, operation, stream_id,
                               global_round: int, patch) -> None:
        """Produce and transmit one physical idle round for a live stream."""
        self._emit(self.model.idle_round_payloads(
            operation, stream_id, global_round, patch), operation)

    def emit_feedback_memory_round(self, operation_id, patch,
                                   round_index: int) -> None:
        """Produce the timing-only physical round of an idle patch."""
        payload = QPUReadout(("idle", operation_id, patch), patch, round_index)
        self.readout_receiver.accept_qpu_readout(
            payload, SyndromePacketRoute.feedback_memory_round(operation_id))


def _patches(operation) -> tuple:
    if operation.patches:
        return tuple(operation.patches)
    if operation.qubits:
        return (operation.qubits[0],)
    return (0,)
