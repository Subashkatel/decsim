"""Idle rounds per patch: how they travel and what decode work they cost.

Syndrome extraction never stops, so a patch nobody operates on emits a
round every cycle. A patch that holds an unsealed stream sends it as the
stream's next round; for any other patch the idle policy decides how it
travels and whether it is charged as decode work. An operation claims
every round its patches emitted since the last claim, so the windows can
prepend them. Idle rounds are decoder workload: the backlog bound counts
every generated syndrome bit (Terhal 1302.3428; Battistel et al.
2303.00054).
"""

import dataclasses
from typing import Any, Optional

import decsim.controller.feedback_streams as feedback_streams
import decsim.ports as ports
import decsim.records.program as program_records
import decsim.trace_source as trace_source


class IdleRoundAccounting:
    """Accounts for each idle round as the idle policy rules.

    The policy decides how the round travels and which decodes it
    charges.
    """

    decode_queue = ports.Port(ports.DecodeQueue)
    streams = ports.Port(feedback_streams.Streams)
    qpu = ports.Port(ports.Qpu)
    windows = ports.Port(ports.WindowInput)

    def __init__(
        self, policy: ports.IdlePolicy, geometry_by_patch: dict
    ) -> None:
        self.policy = policy
        self.geometry_by_patch = geometry_by_patch
        self.operation_by_id: dict = {}
        self.idle_by_patch: dict = {}
        self.trace = _TraceSources()

    def load(self, program: program_records.ExecutionProgram) -> None:
        """Know the operations an idle patch names."""
        for operation in program.operations:
            self.operation_by_id[operation.id] = operation

    def emit_idle_round(
        self,
        operation_id: Any,  # an opaque identity
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """One idle cycle of a patch nobody is operating on.

        A patch on a live protected stream emits through that stream, and one
        that holds an unsealed stream continues it whatever the policy.
        """
        if self.streams.is_live_protected_patch(patch):
            return
        operation = self.operation_by_id[operation_id]
        if self.streams.extend_live_stream(operation, patch):
            return
        self.policy.relay(self, operation, patch, round_index)
        idle = self._idle(patch)
        idle.operation = operation
        idle.unclaimed += 1
        self.trace.idle_round_emitted.fire(operation_id, patch, round_index)

    def start_command(
        self, command: program_records.RunOperationBody
    ) -> program_records.RunOperationBody:
        """The operation starts on its patches and claims their idle rounds.

        A waiting patch keeps measuring, and those rounds are history the
        decoder must read too (Quantum Machines 2412.00289; Holmes et al.
        2004.04794). The QPU emits a boundary's idle rounds before the commands
        on it, so the claim takes every one.
        """
        operation = command.operation
        for patch in program_records.patches_of(operation):
            self.policy.end_idle_period(self, operation, patch)
        idle_round_count = self.claim(operation)
        if idle_round_count:
            self.windows.prepend_idle_rounds(operation.id, idle_round_count)
        return self.streams.bind_at_start(command)

    def claim(self, operation: program_records.Operation) -> int:
        """The idle cycles on the operation's patches since the last claim.

        A code cycle measures every check of every patch once (Litinski
        1808.02892), so patches idle through the same cycles count them once:
        the longest run among the patches.
        """
        patches = operation.patches
        if not patches:
            patches = operation.qubits
        cycle_count = 0
        for patch in patches:
            idle = self.idle_by_patch.get(patch)
            if idle is None:
                continue
            cycle_count = max(cycle_count, idle.unclaimed)
            idle.unclaimed = 0
        return cycle_count

    def end_every_idle_period(self) -> None:
        """The workload is complete: the policy settles every idle patch.

        No idle round follows: the QPU stops its idle patches on the boundary
        the workload completes on.
        """
        for patch, idle in self.idle_by_patch.items():
            self.policy.end_idle_period(self, idle.operation, patch)

    # ---- what a policy may do with a round

    def emit_memory_round(
        self,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """The round travels as a feedback-memory round of the operation."""
        self.qpu.emit_feedback_memory_round(operation.id, patch, round_index)

    def submit_idle_decode_if_due(
        self,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Count one idle round; charge a job at each full commit region."""
        geometry = self.geometry_by_patch[patch].code_geometry
        idle = self._idle(patch)
        idle.uncharged += 1
        if idle.uncharged == geometry.commit_round_count:
            self._submit_idle_decode(
                operation, patch, idle.uncharged, round_index
            )
            idle.uncharged = 0
        idle.last_round_index = round_index

    def submit_idle_decode_for_remaining_rounds(
        self,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
    ) -> None:
        """Charge the rounds after the last full commit region as one job."""
        idle = self._idle(patch)
        uncharged = idle.uncharged
        last_round_index = idle.last_round_index
        idle.uncharged = 0
        idle.last_round_index = 0
        if uncharged:
            self._submit_idle_decode(
                operation, patch, uncharged, last_round_index
            )

    def _submit_idle_decode(
        self, operation, patch, idle_round_count: int, round_index: int
    ) -> None:
        """One load-only decode job: the idle rounds plus the buffer rounds."""
        patch_record = self.geometry_by_patch[patch]
        geometry = patch_record.code_geometry
        rounds = idle_round_count + geometry.buffer_round_count
        self.decode_queue.enqueue_without_input(
            rounds,
            on_done=_ignore_completion,
            code=geometry.code_name,
            spatial_nodes=patch_record.spatial_node_count,
            label=f"mem({operation.name},r{round_index})",
        )

    def _idle(self, patch) -> "_PatchIdle":
        idle = self.idle_by_patch.get(patch)
        if idle is None:
            idle = _PatchIdle()
            self.idle_by_patch[patch] = idle
        return idle


@dataclasses.dataclass
class _PatchIdle:
    """One patch's idle rounds: since the last claim, since the last job.

    operation, the one that left the patch idle, labels a job charged at
    the workload's end.
    """

    operation: Optional[program_records.Operation] = None
    unclaimed: int = 0
    uncharged: int = 0
    last_round_index: int = 0


def _ignore_completion() -> None:
    """An idle decode's completion has no listener."""


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the idle round accounting reports, as one member."""

    idle_round_emitted: trace_source.TraceSource = trace_source.new_source()
