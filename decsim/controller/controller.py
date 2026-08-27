"""The controller: admitted operations become QPU commands, QPU readouts
become controller-side binary handed to syndrome packing, and conditional
releases from the Pauli frame travel OC then CQ back to the QPU. The QEC cycle itself is the
QPU's; execution admission is the ExecutionRuntime's; stream bookkeeping and
protected regions are FeedbackStreams'."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Callable

from ..links.links import LinkPath, TrafficAttribution
from ..message import (Decision, Operation, QPUReadout, RunOperationBody,
                       SyndromePacketRoute, SyndromePayload, normalize_binary_bits)


class Controller:
    def __init__(self, engine, *, qpu, window_manager, syndrome_packing=None,
                 binary_availability_ticks: int = 0, links=None,
                 round_ticks: int, code_geometry, resolved_operations,
                 resolved_patches, idle_policy, feedback_streams):
        self.engine = engine
        self.qpu = qpu
        self.window_manager = window_manager
        self.syndrome_packing = syndrome_packing
        self.binary_availability_ticks = binary_availability_ticks
        self.links = links
        self.runtime = None
        self.round_ticks = round_ticks
        self._code_geometry = code_geometry
        self.idle_policy = idle_policy
        self.streams = feedback_streams
        self._resolved_operations = MappingProxyType({
            operation.operation_id: operation for operation in resolved_operations})
        self._resolved_patches = MappingProxyType({
            patch.patch_identity: patch for patch in resolved_patches})
        self.idle_rounds_emitted = 0

    def round_ticks_for(self, operation: Operation) -> int:
        """The resolved QEC cycle length of one operation, in ticks."""
        return self._resolved_operations[operation.id].round_ticks

    def round_count_for(self, operation: Operation) -> int:
        """The resolved round count of one operation."""
        return self._resolved_operations[operation.id].round_count

    def connect_runtime(self, runtime) -> None:
        """Bind the execution runtime that owns readiness and completion."""
        self.runtime = runtime

    def load_program(self, program) -> None:
        """Load one immutable program: streams first, then every operation
        is registered with the window manager, then the runtime starts roots."""
        self.streams.load(program)
        for operation in program.operations:
            self.window_manager.register_op(operation)
        self.runtime.load_program(program)

    # ---- operations

    def can_start(self, operation: Operation) -> bool:
        """False while a protected feedback stream holds the operation for its cycle boundary."""
        return not self.streams.blocks_start(operation)

    def issue_operation(self, operation: Operation, idle_rounds: int) -> int:
        """Command the QPU; return the cycle boundary the operation starts on."""
        self.streams.begin(operation)
        if idle_rounds:
            self.window_manager.prepend_idle_rounds(operation.id, idle_rounds)
        self._log_start(operation)
        binding = self.streams.binding_for(operation.id)
        if binding is None:
            issued_operation = operation
            source_operation_id = operation.id
        else:
            issued_operation = replace(operation, stream_id=binding.stream_id,
                                       stream_offset=binding.stream_offset)
            source_operation_id = binding.stream_id
        source_round_count = self._resolved_operations[source_operation_id].round_count
        self.qpu.issue(RunOperationBody(
            operation=issued_operation,
            round_ticks=self.round_ticks_for(operation),
            round_count=self.round_count_for(operation),
            source_round_count=source_round_count,
            emits_detector_data=operation.emits_detector_data,
            finalizes_stream_round=operation.finalizes_stream_round,
        ))
        return self.qpu.next_boundary()

    def _log_start(self, operation: Operation) -> None:
        kind = "Clifford" if operation.clifford else "non-Clifford"
        release_note = ""
        if operation.blocked_by is not None:
            release_note = f" [unblocked by op#{operation.blocked_by}]"
        self.engine.log("Controller", f"START {operation.name}  ({kind}, qubits "
                                      f"{operation.qubits}){release_note}")

    def stream_binding_for(self, operation_id):
        """(stream_id, stream_offset) an operation was bound to, or None."""
        return self.streams.binding_for(operation_id)

    def _body_done(self, operation: Operation) -> None:
        self.runtime.body_done(operation)

    def before_successor_release(self, operation: Operation) -> None:
        """A body finished: ask its protected regions to close on the boundary."""
        self.streams.request_closes(operation)

    def after_successor_release(self, operation: Operation) -> None:
        """Successors released: close feedback boundaries, seal finished streams, stop the QPU when done."""
        self.streams.close_feedback_boundary(
            operation, self.runtime.waiting_blocked_successor(operation.id))
        if self.runtime.workload_complete:
            self.streams.seal_finished_streams()
            self.qpu.finish()

    # ---- idle rounds

    def emit_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """One idle cycle of a patch nobody is operating on: syndrome extraction
        never stops, so the round is produced, transmitted and accounted; the
        idle policy decides how it travels and whether it costs decode work.
        Patches on a live protected stream emit through that stream."""
        if self.streams.is_live_protected_patch(patch):
            return
        operation = self.runtime.operations[op_id]
        self.idle_policy.relay(self, operation, patch, round_index)
        self.runtime.record_idle_round(patch)
        self.idle_rounds_emitted += 1

    def emit_memory_round(self, operation: Operation, patch, round_index: int) -> None:
        """An idle round travels as an ordinary feedback-memory round of the operation."""
        self.qpu.emit_feedback_memory_round(operation.id, patch, round_index)

    def extend_live_stream(self, operation: Operation, patch) -> bool:
        """An idle round becomes the next round of the operation's live stream, if it has one."""
        return self.streams.extend_live_stream(operation, patch)

    def submit_idle_decode_if_due(self, operation: Operation, patch,
                                  round_index: int) -> None:
        """Every commit region of idle rounds costs one load-only decode job."""
        patch_record = self._resolved_patches[patch]
        geometry = patch_record.code_geometry
        if round_index % geometry.commit_round_count == 0:
            self.window_manager.accept_idle_decode_demand(
                rounds=geometry.commit_round_count + geometry.buffer_round_count,
                code=geometry.code_name,
                spatial_nodes=patch_record.spatial_node_count,
                label=f"mem({operation.name},r{round_index})")

    # ---- readouts and instructions

    def accept_qpu_readout(self, readout: QPUReadout, route: SyndromePacketRoute) -> None:
        """Turn one QPU readout into controller binary and hand it to packing."""
        payload = SyndromePayload(
            operation_id=readout.operation_id,
            patch_id=readout.patch_id,
            round_index=readout.round_index,
            bits=normalize_binary_bits(readout.bits),
            code=readout.code,
            n_fragments=readout.n_fragments,
            fragment_index=readout.fragment_index,
            size_bits=readout.size_bits,
        )
        self.syndrome_packing.relay_qpu_readout(
            payload, route, processing_ticks=self.binary_availability_ticks)

    def relay_instruction(self, decision: Decision,
                          deliver: Callable[[Decision], None]) -> None:
        """Carry a Pauli frame decision over OC to the controller, then over CQ to the QPU."""
        if self.links is None:
            deliver(decision)
            return
        attribution = TrafficAttribution(
            operation_id=decision.target_operation_id, patch_ids=(),
            window_id=None, round_lo=None, round_hi=None)

        def at_controller():
            cq_delay = self._instruction_delay(LinkPath.CQ, attribution)
            self.engine.schedule(cq_delay, lambda: deliver(decision), label="controller->qpu")

        oc_delay = self._instruction_delay(LinkPath.OC, attribution)
        self.engine.schedule(oc_delay, at_controller, label="pauli frame->controller")

    def _instruction_delay(self, path: LinkPath, attribution: TrafficAttribution) -> int:
        reservation = self.links.reserve(path, payload_bits=None, now_ticks=self.engine.now,
                                         attribution=attribution)
        return reservation.total_delay_ticks
