"""The controller between the QPU, the round assembler and the Pauli frame.

QPU readouts cross qpu_to_controller and, after the readout delay,
become fragments for the assembler; admitted operations become QPU
commands; the frame's decisions travel frame_to_controller and
controller_to_qpu back to the QPU.

The QEC cycle itself is the QPU's; execution admission is the
ExecutionRuntime's; stream bookkeeping and protected regions are
FeedbackStreams'.
"""

import dataclasses
import functools
import types
from typing import Callable, Optional

import decsim.message as message
import decsim.observe.round_events as round_events
import decsim.qpu.cycle_clock as cycle_clock


class Controller:
    """The controller between the QPU, the assembler and the frame."""

    def __init__(
        self,
        engine,
        *,
        qpu,
        window_manager,
        assembler=None,
        recorder=None,
        measurement_signal_to_classical_bits_ticks: int = 0,
        instruction_or_decision_to_analog_control_pulse_ticks: int = 0,
        links=None,
        resolved_operations,
        resolved_patches,
        idle_policy,
        feedback_streams,
    ):
        self.engine = engine
        self.qpu = qpu
        self.window_manager = window_manager
        self.assembler = assembler
        if recorder is None:
            recorder = round_events.NoRoundEvents()
        self.recorder = recorder
        self.measurement_signal_to_classical_bits_ticks = (
            measurement_signal_to_classical_bits_ticks
        )
        self.instruction_or_decision_to_analog_control_pulse_ticks = (
            instruction_or_decision_to_analog_control_pulse_ticks
        )
        self.links = links
        self.runtime = None
        self.idle_policy = idle_policy
        self.streams = feedback_streams
        operation_by_id = {
            operation.operation_id: operation
            for operation in resolved_operations
        }
        self._resolved_operations = types.MappingProxyType(operation_by_id)
        patch_by_identity = {
            patch.patch_identity: patch for patch in resolved_patches
        }
        self._resolved_patches = types.MappingProxyType(patch_by_identity)
        self.idle_rounds_emitted = 0
        # idle rounds per patch not yet covered by a load-only decode job,
        # and the index of the latest one (it names the job)
        self._uncharged_idle_rounds_by_patch: dict = {}
        self._last_idle_round_index_by_patch: dict = {}

    def round_ticks_for(self, operation: message.Operation) -> int:
        """The resolved QEC cycle length of one operation, in ticks."""
        return self._resolved_operations[operation.id].round_ticks

    def round_count_for(self, operation: message.Operation) -> int:
        """The resolved round count of one operation."""
        return self._resolved_operations[operation.id].round_count

    def connect_runtime(self, runtime) -> None:
        """Bind the execution runtime that owns readiness and completion."""
        self.runtime = runtime

    def load_program(self, program) -> None:
        """Load one program: streams, then every operation, then the roots."""
        self.streams.load(program)
        for operation in program.operations:
            self.window_manager.register_op(operation)
        self.runtime.load_program(program)

    # ---- operations

    def can_start(self, operation: message.Operation) -> bool:
        """False while a protected stream holds the operation for a boundary."""
        return not self.streams.blocks_start(operation)

    def issue_operation(
        self, operation: message.Operation, idle_rounds: int
    ) -> None:
        """Prepare one QPU command; the runtime learns its start boundary.

        Program roots and ordinary DAG successors are preloaded: their
        controller preparation happened before the simulated interval,
        and the runtime hears the start boundary now. A feedback-blocked
        operation is dynamic: its RunOperationBody pays the decision to
        pulse cost and controller_to_qpu before it enters the QPU, and
        the runtime hears the boundary at arrival.
        """
        self.streams.begin(operation)
        patches = cycle_clock.patches_of(operation)
        for patch in patches:
            self.end_idle_period(operation, patch)
        if idle_rounds:
            self.window_manager.prepend_idle_rounds(operation.id, idle_rounds)
        self._log_start(operation)
        command = self._command(operation)
        if operation.blocked_by is None:
            self.recorder.output("PRELOADED_COMMAND", operation.id, command)
            self._start_on_qpu(operation, command)
            return
        self._dispatch_dynamic_command(operation, command)

    def _command(
        self, operation: message.Operation
    ) -> message.RunOperationBody:
        """The operation's command, bound to its stream when it has one."""
        binding = self.streams.binding_for(operation.id)
        if binding is None:
            issued_operation = operation
            source_operation_id = operation.id
        else:
            issued_operation = dataclasses.replace(
                operation,
                stream_id=binding.stream_id,
                stream_offset=binding.stream_offset,
            )
            source_operation_id = binding.stream_id
        source = self._resolved_operations[source_operation_id]
        round_ticks = self.round_ticks_for(operation)
        round_count = self.round_count_for(operation)
        return message.RunOperationBody(
            operation=issued_operation,
            round_ticks=round_ticks,
            round_count=round_count,
            source_round_count=source.round_count,
            emits_detector_data=operation.emits_detector_data,
            finalizes_stream_round=operation.finalizes_stream_round,
        )

    def _start_on_qpu(
        self, operation: message.Operation, command: message.RunOperationBody
    ) -> None:
        """The command is at the QPU: it starts on the next cycle boundary."""
        self.qpu.issue(command)
        boundary = self.qpu.next_boundary()
        self.runtime.operation_started(operation, boundary)

    def _dispatch_dynamic_command(
        self, operation: message.Operation, command: message.RunOperationBody
    ) -> None:
        """Carry the feedback-selected command through pulse generation."""
        deliver = functools.partial(self._start_on_qpu, operation)
        self._send_controller_output(
            payload=command,
            operation_id=operation.id,
            event_kind="CONTROL_PULSE_COMMAND_ISSUED",
            deliver=deliver,
        )

    def _send_controller_output(
        self, *, payload, operation_id, event_kind: str, deliver: Callable
    ) -> None:
        """Process one output payload, then send it to the QPU.

        The send is made at the output tick, when the pulse processing is
        done; deliver(payload) runs when the QPU has it.
        """
        output_delay_ticks = (
            self.instruction_or_decision_to_analog_control_pulse_ticks
        )
        attribution = message.TransferAttribution(
            operation_id=operation_id,
            patch_ids=(),
            window_id=None,
            first_round=None,
            last_round=None,
        )

        def delivered(_transfer):
            deliver(payload)

        def output_ready():
            self.recorder.output(event_kind, operation_id, payload)
            if self.links is None:
                deliver(payload)
                return
            self.links.send(
                message.LinkPath.CONTROLLER_TO_QPU,
                None,
                self.engine.now,
                attribution,
                delivered,
            )

        self.engine.schedule(
            output_delay_ticks, output_ready, label="controller-output-ready"
        )

    def _log_start(self, operation: message.Operation) -> None:
        kind = "non-Clifford"
        if operation.clifford:
            kind = "Clifford"
        release_note = ""
        if operation.blocked_by is not None:
            release_note = f" [unblocked by op#{operation.blocked_by}]"
        self.engine.log(
            "Controller",
            f"START {operation.name}  ({kind}, qubits "
            f"{operation.qubits}){release_note}",
        )

    def stream_binding_for(
        self, operation_id
    ) -> Optional[message.StreamBinding]:
        """The stream binding an operation was given, or None."""
        return self.streams.binding_for(operation_id)

    def _body_done(self, operation: message.Operation) -> None:
        self.runtime.body_done(operation)

    def before_successor_release(self, operation: message.Operation) -> None:
        """A body finished: its protected regions close on the boundary."""
        self.streams.request_closes(operation)

    def after_successor_release(self, operation: message.Operation) -> None:
        """Successors released: close boundaries, seal streams, stop the QPU."""
        waits_for_blocked = self.runtime.waiting_blocked_successor(operation.id)
        self.streams.close_feedback_boundary(operation, waits_for_blocked)
        if self.runtime.workload_complete:
            self.streams.seal_finished_streams()
            self.qpu.finish()

    # ---- idle rounds

    def emit_idle_round(self, operation_id, patch, round_index: int) -> None:
        """One idle cycle of a patch nobody is operating on.

        Syndrome extraction never stops, so the round is produced,
        transmitted and accounted; the idle policy decides how it travels
        and whether it costs decode work. A patch on a live protected
        stream emits through that stream instead.
        """
        if self.streams.is_live_protected_patch(patch):
            return
        operation = self.runtime.operations[operation_id]
        self.idle_policy.relay(self, operation, patch, round_index)
        self.runtime.record_idle_round(patch)
        self.idle_rounds_emitted += 1

    def emit_memory_round(
        self, operation: message.Operation, patch, round_index: int
    ) -> None:
        """An idle round travels as a feedback-memory round of the operation."""
        self.qpu.emit_feedback_memory_round(operation.id, patch, round_index)

    def extend_live_stream(self, operation: message.Operation, patch) -> bool:
        """An idle round becomes the next round of the operation's stream."""
        return self.streams.extend_live_stream(operation, patch)

    def submit_idle_decode_if_due(
        self, operation: message.Operation, patch, round_index: int
    ) -> None:
        """Count one idle round toward the patch's next decode job.

        The job is charged once a full commit region of idle rounds has
        accumulated.
        """
        geometry = self._resolved_patches[patch].code_geometry
        counted = self._uncharged_idle_rounds_by_patch.get(patch, 0)
        uncharged_rounds = counted + 1
        self._uncharged_idle_rounds_by_patch[patch] = uncharged_rounds
        if uncharged_rounds == geometry.commit_round_count:
            self._submit_idle_decode(
                operation, patch, uncharged_rounds, round_index
            )
            self._uncharged_idle_rounds_by_patch[patch] = 0
        self._last_idle_round_index_by_patch[patch] = round_index

    def submit_idle_decode_for_remaining_rounds(
        self, operation: message.Operation, patch
    ) -> None:
        """Charge the idle rounds after the last full commit region as one job.

        Every idle round is decoded; the last job may be shorter.
        """
        uncharged_rounds = self._uncharged_idle_rounds_by_patch.pop(patch, 0)
        last_round_index = self._last_idle_round_index_by_patch.pop(patch, 0)
        if uncharged_rounds:
            self._submit_idle_decode(
                operation, patch, uncharged_rounds, last_round_index
            )

    def _submit_idle_decode(
        self,
        operation: message.Operation,
        patch,
        idle_round_count: int,
        round_index: int,
    ) -> None:
        """One load-only decode job for a region of idle rounds.

        The job is sized to those rounds plus the buffer rounds a window
        reads past them.
        """
        patch_record = self._resolved_patches[patch]
        geometry = patch_record.code_geometry
        rounds = idle_round_count + geometry.buffer_round_count
        self.window_manager.accept_idle_decode_demand(
            rounds=rounds,
            code=geometry.code_name,
            spatial_nodes=patch_record.spatial_node_count,
            label=f"mem({operation.name},r{round_index})",
        )

    def end_idle_period(self, operation: message.Operation, patch) -> None:
        """An operation claims the patch: the policy settles its idle rounds."""
        self.idle_policy.end_idle_period(self, operation, patch)

    # ---- readouts and instructions

    def accept_qpu_readout(
        self, readout: message.QPUReadout, route: message.SyndromePacketRoute
    ) -> None:
        """One readout crosses qpu_to_controller to the assembler.

        The fragment reaches the assembler after the readout delay.
        """
        fragment = message.RetainedSyndromeFragment.from_readout(readout)
        fragment_count = readout.n_fragments
        self.recorder.record(
            "EMITTED",
            fragment.operation_id,
            fragment.round_index,
            route,
            patch_id=fragment.patch_id,
        )
        attribution = message.TransferAttribution.for_round(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index
        )
        processing_ticks = self.measurement_signal_to_classical_bits_ticks

        def receive():
            self.assembler.add(fragment, fragment_count, route)

        def at_controller(_transfer):
            if processing_ticks == 0:
                receive()
                return
            self.engine.schedule(
                processing_ticks,
                receive,
                label="controller-binary-availability",
            )

        self.links.send(
            message.LinkPath.QPU_TO_CONTROLLER,
            readout.size_bits,
            self.engine.now,
            attribution,
            at_controller,
        )

    def relay_instruction(
        self,
        decision: message.Decision,
        deliver: Callable[[message.Decision], None],
    ) -> None:
        """Carry a Pauli-frame decision over frame_to_controller.

        A release is consumed at the controller; the operation command it
        releases then crosses pulse generation and controller_to_qpu in
        issue_operation. A result return without an operation still
        crosses those stages before it is available at the QPU.
        """
        attribution = message.TransferAttribution(
            operation_id=decision.target_operation_id,
            patch_ids=(),
            window_id=None,
            first_round=None,
            last_round=None,
        )

        def at_controller(_transfer=None):
            self.recorder.output(
                "DECISION_AVAILABLE", decision.target_operation_id, decision
            )
            if decision.releases_operation:
                deliver(decision)
                return
            self._send_controller_output(
                payload=decision,
                operation_id=decision.target_operation_id,
                event_kind="CONTROL_DECISION_ISSUED",
                deliver=deliver,
            )

        if self.links is None:
            self.engine.schedule(
                0, at_controller, label="pauli frame->controller"
            )
            return
        self.links.send(
            message.LinkPath.FRAME_TO_CONTROLLER,
            None,
            self.engine.now,
            attribution,
            at_controller,
        )
