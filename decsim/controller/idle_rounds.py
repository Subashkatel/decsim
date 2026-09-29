"""Idle rounds per patch: how they travel and what decode work they cost.

Syndrome extraction never stops, so a patch nobody is operating on
emits a round every cycle. The QPU reports each one here. A patch that
holds an unsealed stream sends it as that stream's next round; for any
other patch the idle policy (controller/policies.py) decides how it
travels (a feedback-memory round of the operation that left the patch)
and whether it is charged as decode work (one load-only job per commit
region of idle rounds, the last one shorter). An operation that
claims the patch takes the rounds emitted since the last claim, so the
window manager can prepend them to its plan; the rounds of a patch no
operation claims again settle when the workload completes. Idle rounds
are decoder workload: the backlog bound counts every generated syndrome
bit against the decoder's processing rate (Terhal 1302.3428 lines
3151-3159; Battistel et al. 2303.00054 line 144).
"""

import dataclasses
from typing import Optional

import decsim.controller.feedback_streams as feedback_streams
import decsim.ports as ports
import decsim.records.program as program_records
import decsim.trace_source as trace_source


@dataclasses.dataclass
class _PatchIdle:
    """One patch's idle rounds: since the last claim, since the last job.

    operation is the one that left the patch idle, which a job charged
    at the workload's end is labelled with.
    """

    operation: Optional[program_records.Operation] = None
    unclaimed: int = 0
    uncharged: int = 0
    last_round_index: int = 0


class IdleRoundAccounting:
    """Routes each idle round by the policy and charges the decodes.

    Trace source: idle_round_emitted(operation_id, patch, round_index)
    for every idle round the policy relayed.
    """

    decode_queue = ports.Port(ports.DecodeQueue)
    streams = ports.Port(feedback_streams.Streams)
    qpu = ports.Port(ports.Qpu)

    def __init__(self, policy: ports.IdlePolicy, geometry_by_patch) -> None:
        self.policy = policy
        self.geometry_by_patch = geometry_by_patch
        self.operation_by_id: dict = {}
        self.idle_by_patch: dict = {}
        self.trace = _TraceSources()

    def load(self, program) -> None:
        """Know the operations an idle patch names."""
        for operation in program.operations:
            self.operation_by_id[operation.id] = operation

    def emit_idle_round(self, operation_id, patch, round_index: int) -> None:
        """One idle cycle of a patch nobody is operating on.

        The round is produced, transmitted and accounted; the policy
        decides how it travels and whether it costs decode work. A patch
        on a live protected stream emits through that stream instead, and
        a patch that holds an unsealed stream continues it whatever the
        policy.
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

    def bind_at_start(
        self, command: program_records.RunOperationBody
    ) -> program_records.RunOperationBody:
        """The streams bind a continuation after its patches' idle rounds."""
        return self.streams.bind_at_start(command)

    def claim(self, operation) -> int:
        """The idle cycles on the operation's patches since the last claim.

        A code cycle measures every check of every patch once (Litinski
        1808.02892 lines 204-206), so patches idle through the same cycles
        are those cycles once. Each patch's unclaimed rounds are the
        cycles since it was last claimed, one per cycle, all ending here,
        so the longest run among the patches is every cycle they idled.
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

    def end_idle_period(self, operation, patch) -> None:
        """An operation claims the patch: the policy settles its idle rounds."""
        self.policy.end_idle_period(self, operation, patch)

    def end_every_idle_period(self) -> None:
        """The workload is complete: the policy settles every idle patch.

        No idle round follows: the workload completes when a body ends
        on a cycle boundary, after that boundary's idle rounds, and the
        QPU stops its idle patches on the same boundary
        (qpu/cycle_clock.py _cross_boundary).
        """
        for patch, idle in self.idle_by_patch.items():
            self.policy.end_idle_period(self, idle.operation, patch)

    # ---- what a policy may do with a round

    def emit_memory_round(self, operation, patch, round_index: int) -> None:
        """The round travels as a feedback-memory round of the operation."""
        self.qpu.emit_feedback_memory_round(operation.id, patch, round_index)

    def submit_idle_decode_if_due(self, operation, patch, round_index) -> None:
        """Count one idle round toward the patch's next decode job.

        The job is charged once a full commit region of idle rounds has
        accumulated.
        """
        geometry = self.geometry_by_patch[patch].code_geometry
        idle = self._idle(patch)
        idle.uncharged += 1
        if idle.uncharged == geometry.commit_round_count:
            self._submit_idle_decode(
                operation, patch, idle.uncharged, round_index
            )
            idle.uncharged = 0
        idle.last_round_index = round_index

    def submit_idle_decode_for_remaining_rounds(self, operation, patch) -> None:
        """Charge the idle rounds after the last full commit region as one job.

        Every idle round is decoded; the last job may be shorter.
        """
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
        """One load-only decode job for a region of idle rounds.

        The job is sized to those rounds plus the buffer rounds a window
        reads past them.
        """
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

    def _idle(self, patch) -> _PatchIdle:
        idle = self.idle_by_patch.get(patch)
        if idle is None:
            idle = _PatchIdle()
            self.idle_by_patch[patch] = idle
        return idle


def _ignore_completion() -> None:
    """An idle decode's completion has no listener."""


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the idle round accounting reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    idle_round_emitted: trace_source.TraceSource = trace_source.new_source()
