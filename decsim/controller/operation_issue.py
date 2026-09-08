"""The issuer: an admitted operation becomes one QPU command.

The execution runtime admits an operation when its predecessors are done
and its release has arrived; the issuer then opens its protected
regions, settles the idle rounds of its patches, prepends the idle rounds
the windows must plan for, and hands the command to the output. A
program root or an ordinary successor is preloaded and starts at the
next boundary; a feedback-blocked operation is dynamic and pays the
output path first. The runtime hears the start boundary through
on_started, so the issuer never calls the runtime.
"""

import dataclasses
import types
from typing import Callable, Optional

import decsim.records.log_sources as log_sources
import decsim.records.program as program_records


class OperationIssuer:
    """What the runtime asks of the controller for one operation."""

    def __init__(
        self,
        engine,
        streams,
        idle_rounds,
        windows,
        resolved_operations,
        output,
    ) -> None:
        self.engine = engine
        self.streams = streams
        self.idle_rounds = idle_rounds
        self.windows = windows
        operation_by_id = {
            operation.operation_id: operation
            for operation in resolved_operations
        }
        self.resolved_operation_by_id = types.MappingProxyType(operation_by_id)
        self.output = output

    def round_ticks_for(self, operation: program_records.Operation) -> int:
        """The resolved QEC cycle length of one operation, in ticks."""
        return self.resolved_operation_by_id[operation.id].round_ticks

    def round_count_for(self, operation: program_records.Operation) -> int:
        """The resolved round count of one operation."""
        return self.resolved_operation_by_id[operation.id].round_count

    def can_start(self, operation: program_records.Operation) -> bool:
        """False while a protected stream holds the operation for a boundary."""
        return not self.streams.blocks_start(operation)

    def issue_operation(
        self,
        operation: program_records.Operation,
        on_started: Callable[[int], None],
    ) -> None:
        """Prepare one QPU command; on_started hears its start boundary."""
        self.streams.begin(operation)
        patches = program_records.patches_of(operation)
        for patch in patches:
            self.idle_rounds.end_idle_period(operation, patch)
        idle_round_count = self.idle_rounds.claim(operation)
        if idle_round_count:
            self.windows.prepend_idle_rounds(operation.id, idle_round_count)
        self._log_start(operation)
        command = self._command(operation)
        if operation.blocked_by is None:
            self.output.start_preloaded(command, on_started)
            return
        self.output.send_command(command, on_started)

    def before_successor_release(
        self, operation: program_records.Operation
    ) -> None:
        """A body finished: its protected regions close on the boundary."""
        self.streams.request_closes(operation)

    def after_successor_release(
        self,
        operation: program_records.Operation,
        waits_for_blocked: bool,
        is_workload_complete: bool,
    ) -> None:
        """Successors released: close boundaries, seal streams, stop the QPU."""
        self.streams.close_feedback_boundary(operation, waits_for_blocked)
        if is_workload_complete:
            self.streams.seal_finished_streams()
            self.output.finish()

    def stream_binding_for(
        self, operation_id
    ) -> Optional[program_records.StreamBinding]:
        """The stream binding an operation was given, or None."""
        return self.streams.binding_for(operation_id)

    def _command(
        self, operation: program_records.Operation
    ) -> program_records.RunOperationBody:
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
        source = self.resolved_operation_by_id[source_operation_id]
        round_ticks = self.round_ticks_for(operation)
        round_count = self.round_count_for(operation)
        return program_records.RunOperationBody(
            operation=issued_operation,
            round_ticks=round_ticks,
            round_count=round_count,
            source_round_count=source.round_count,
            emits_detector_data=operation.emits_detector_data,
            finalizes_stream_round=operation.finalizes_stream_round,
        )

    def _log_start(self, operation: program_records.Operation) -> None:
        kind = "non-Clifford"
        if operation.clifford:
            kind = "Clifford"
        release_note = ""
        if operation.blocked_by is not None:
            release_note = f" [unblocked by op#{operation.blocked_by}]"
        self.engine.log(
            log_sources.CONTROLLER,
            f"START {operation.name}  ({kind}, qubits "
            f"{operation.qubits}){release_note}",
        )
