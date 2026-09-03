"""The controller: admitted operations become QPU commands, QPU readouts
become controller-side binary handed to syndrome packing, and conditional
releases from the Pauli frame travel OC then CQ back to the QPU. The QEC cycle itself is the
QPU's; execution admission is the ExecutionRuntime's; stream bookkeeping and
protected regions are FeedbackStreams'."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable

from ..qpu.cycle_clock import patches_of
from ..message import (Decision, LinkPath, Operation, QPUReadout, RunOperationBody,
                       SyndromePacketRoute, SyndromePayload, TransferAttribution,
                       normalize_binary_bits)


@dataclass(frozen=True)
class ControllerOutputEvent:
    """One auditable transition on the controller's digital-to-QPU path.

    ``payload`` is the actual immutable decision or QPU command crossing the
    boundary, so tests and ledgers can prove that timing was not modeled with
    a detached token while different data reached the QPU.
    """

    kind: str
    tick: int
    operation_id: object
    payload: object


class Controller:
    def __init__(self, engine, *, qpu, window_manager, syndrome_packing=None,
                 measurement_signal_to_classical_bits_ticks: int = 0,
                 instruction_or_decision_to_analog_control_pulse_ticks: int = 0,
                 links=None, resolved_operations, resolved_patches, idle_policy,
                 feedback_streams):
        self.engine = engine
        self.qpu = qpu
        self.window_manager = window_manager
        self.syndrome_packing = syndrome_packing
        if measurement_signal_to_classical_bits_ticks < 0:
            raise ValueError("measurement_signal_to_classical_bits_ticks must be nonnegative")
        if instruction_or_decision_to_analog_control_pulse_ticks < 0:
            raise ValueError(
                "instruction_or_decision_to_analog_control_pulse_ticks must be nonnegative")
        self.measurement_signal_to_classical_bits_ticks = \
            measurement_signal_to_classical_bits_ticks
        self.instruction_or_decision_to_analog_control_pulse_ticks = \
            instruction_or_decision_to_analog_control_pulse_ticks
        self.links = links
        self.runtime = None
        self.idle_policy = idle_policy
        self.streams = feedback_streams
        self._resolved_operations = MappingProxyType({
            operation.operation_id: operation for operation in resolved_operations})
        self._resolved_patches = MappingProxyType({
            patch.patch_identity: patch for patch in resolved_patches})
        self.idle_rounds_emitted = 0
        # idle rounds per patch not yet covered by a load-only decode job,
        # and the index of the latest one (it names the job)
        self._uncharged_idle_rounds_by_patch: dict = {}
        self._last_idle_round_index_by_patch: dict = {}
        self.output_events: list[ControllerOutputEvent] = []

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

    def issue_operation(self, operation: Operation, idle_rounds: int) -> None:
        """Prepare one real QPU command; the runtime learns its start boundary.

        Program roots and ordinary DAG successors are time-tagged/preloaded;
        their online controller preparation occurred before the simulated
        interval, and the runtime hears the start boundary now.  A
        feedback-blocked operation is dynamic: its actual
        ``RunOperationBody`` pays controller output processing and CQ before
        it can enter the QPU, and the runtime hears the boundary at arrival.
        """
        self.streams.begin(operation)
        for patch in patches_of(operation):
            self.end_idle_period(operation, patch)
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
        command = RunOperationBody(
            operation=issued_operation,
            round_ticks=self.round_ticks_for(operation),
            round_count=self.round_count_for(operation),
            source_round_count=source_round_count,
            emits_detector_data=operation.emits_detector_data,
            finalizes_stream_round=operation.finalizes_stream_round,
        )
        if operation.blocked_by is None:
            self.output_events.append(ControllerOutputEvent(
                "PRELOADED_COMMAND", self.engine.now, operation.id, command))
            self._start_on_qpu(operation, command)
            return
        self._dispatch_dynamic_command(operation, command)

    def _start_on_qpu(self, operation: Operation, command: RunOperationBody) -> None:
        """The command is at the QPU: it starts on the next cycle boundary."""
        self.qpu.issue(command)
        boundary = self.qpu.next_boundary()
        self.runtime.operation_started(operation, boundary)

    def _dispatch_dynamic_command(self, operation: Operation,
                                  command: RunOperationBody) -> None:
        """Carry the feedback-selected command through pulse generation and CQ."""
        self._send_controller_output(
            payload=command, operation_id=operation.id,
            event_kind="CONTROL_PULSE_COMMAND_ISSUED",
            deliver=lambda arrived: self._start_on_qpu(operation, arrived))

    def _send_controller_output(self, *, payload, operation_id,
                                event_kind: str, deliver: Callable) -> None:
        """Process one output payload, then send it over CQ to the QPU.

        The send is made at the output tick, when the pulse processing is
        done; ``deliver(payload)`` runs when the QPU has it.
        """
        output_delay_ticks = self.instruction_or_decision_to_analog_control_pulse_ticks
        attribution = TransferAttribution(
            operation_id=operation_id, patch_ids=(), window_id=None,
            first_round=None, last_round=None)

        def delivered(_transfer):
            deliver(payload)

        def output_ready():
            self.output_events.append(ControllerOutputEvent(
                event_kind, self.engine.now, operation_id, payload))
            if self.links is None:
                deliver(payload)
                return
            self.links.send(LinkPath.CQ, None, self.engine.now, attribution,
                            delivered)

        self.engine.schedule(output_delay_ticks, output_ready,
                             label="controller-output-ready")

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
        """Count one idle round toward the patch's next decode job and charge
        the job once a full commit region of idle rounds has accumulated."""
        geometry = self._resolved_patches[patch].code_geometry
        uncharged_rounds = self._uncharged_idle_rounds_by_patch.get(patch, 0) + 1
        self._uncharged_idle_rounds_by_patch[patch] = uncharged_rounds
        if uncharged_rounds == geometry.commit_round_count:
            self._submit_idle_decode(operation, patch, uncharged_rounds, round_index)
            self._uncharged_idle_rounds_by_patch[patch] = 0
        self._last_idle_round_index_by_patch[patch] = round_index

    def submit_idle_decode_for_remaining_rounds(self, operation: Operation,
                                                patch) -> None:
        """Charge the idle rounds left after the last full commit region as
        one shorter job; every idle round is decoded."""
        uncharged_rounds = self._uncharged_idle_rounds_by_patch.pop(patch, 0)
        last_round_index = self._last_idle_round_index_by_patch.pop(patch, 0)
        if uncharged_rounds:
            self._submit_idle_decode(operation, patch, uncharged_rounds, last_round_index)

    def _submit_idle_decode(self, operation: Operation, patch,
                            idle_round_count: int, round_index: int) -> None:
        """One load-only decode job for a region of idle rounds, sized to
        those rounds plus the buffer rounds a window reads past them."""
        patch_record = self._resolved_patches[patch]
        geometry = patch_record.code_geometry
        self.window_manager.accept_idle_decode_demand(
            rounds=idle_round_count + geometry.buffer_round_count,
            code=geometry.code_name,
            spatial_nodes=patch_record.spatial_node_count,
            label=f"mem({operation.name},r{round_index})")

    def end_idle_period(self, operation: Operation, patch) -> None:
        """An operation claims the patch: the idle policy settles the idle
        rounds it has not charged yet."""
        self.idle_policy.end_idle_period(self, operation, patch)

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
            payload, route,
            processing_ticks=self.measurement_signal_to_classical_bits_ticks)

    def relay_instruction(self, decision: Decision,
                          deliver: Callable[[Decision], None]) -> None:
        """Carry a Pauli-frame decision over OC to the controller.

        A release is consumed at the controller; the resulting real operation
        command subsequently traverses pulse generation and CQ in ``issue_operation``.
        A result-return without an operation still traverses those stages
        before it is reported as available at the QPU.
        """
        attribution = TransferAttribution(
            operation_id=decision.target_operation_id, patch_ids=(),
            window_id=None, first_round=None, last_round=None)

        def at_controller(_transfer=None):
            self.output_events.append(ControllerOutputEvent(
                "DECISION_AVAILABLE", self.engine.now,
                decision.target_operation_id, decision))
            if decision.releases_operation:
                deliver(decision)
            else:
                self._send_controller_output(
                    payload=decision,
                    operation_id=decision.target_operation_id,
                    event_kind="CONTROL_DECISION_ISSUED",
                    deliver=deliver)

        if self.links is None:
            self.engine.schedule(0, at_controller, label="pauli frame->controller")
            return
        self.links.send(LinkPath.OC, None, self.engine.now, attribution,
                        at_controller)
