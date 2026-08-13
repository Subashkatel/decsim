"""Execute controller commands and emit QPU readout events.

This module schedules the cadence of physical rounds. It does not own program dependencies,
windowing, or decoder state.
"""
from __future__ import annotations
from dataclasses import replace

from .message import (
    RunOperationBody, QPUReadout, SyndromePacketRoute, WINDOW_INPUT_ROUTE,
)


class QPUDevice:
    """Run issued operations on the configured physical device model."""

    def __init__(self, engine, model, readout_receiver=None, completion_receiver=None):
        self.engine = engine
        self.model = model
        self.readout_receiver = readout_receiver
        self.completion_receiver = completion_receiver

    def connect_readout_receiver(self, receiver) -> None:
        self.readout_receiver = receiver

    def connect_completion_receiver(self, receiver) -> None:
        self.completion_receiver = receiver

    def issue(self, command: RunOperationBody) -> None:
        if self.completion_receiver is None:
            raise RuntimeError("QPU completion receiver is not connected")
        if type(command) is not RunOperationBody:
            raise TypeError("QPUDevice accepts only RunOperationBody commands")
        op = command.operation
        if not command.emits_detector_data:
            self.engine.schedule(command.round_count * command.round_ticks,
                lambda: self.completion_receiver(op), label=f"body-done({op.name})")
            return
        if command.round_count == 0:
            if not command.finalizes_stream_round:
                raise ValueError("zero-duration detector emitters must finalize a stream round")
            self._emit(self.model.finalize_stream_round(op, command.source_round_count), op)
            self.completion_receiver(op)
            return
        self.model.begin_operation(op, command.round_count, command.source_round_count)
        self.engine.schedule(command.round_ticks,
            lambda: self._round(command, 1), label=f"round1({op.name})")

    def _round(self, command: RunOperationBody, round_index: int) -> None:
        op = command.operation
        payloads = self.model.round_payloads(op, round_index)
        self.engine.log("QPU", f"{op.name} fires round {round_index}/{command.round_count}")
        self._emit(payloads, op)
        if round_index < command.round_count:
            self.engine.schedule(command.round_ticks,
                lambda: self._round(command, round_index + 1),
                label=f"round{round_index + 1}({op.name})")
        else:
            self.completion_receiver(op)

    def _emit(self, payloads, operation) -> None:
        if type(payloads) not in (list, tuple):
            raise TypeError("QPU model payloads must be an exact list or tuple")
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
            if type(payload) is not QPUReadout:
                raise TypeError("QPU model must emit exact QPUReadout values")
            index = operation.syndrome_fragment_index if operation.syndrome_fragment_index is not None else local_index
            if self.readout_receiver is None:
                raise RuntimeError("QPU readout receiver is not connected")
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
        """Produce the timing-only physical hold round used by feedback waits."""
        payload = QPUReadout(("idle", operation_id, patch), patch, round_index)
        if self.readout_receiver is None:
            raise RuntimeError("QPU readout receiver is not connected")
        self.readout_receiver.accept_qpu_readout(
            payload, SyndromePacketRoute.feedback_memory_round(operation_id))
