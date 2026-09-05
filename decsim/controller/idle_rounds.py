"""Idle rounds per patch: how they travel and what decode work they cost.

Syndrome extraction never stops, so a patch nobody is operating on
emits a round every cycle. The QPU reports each one here; the idle policy
(controller/policies.py) decides how it travels (a feedback-memory round
of the operation that left the patch, or the next round of a live
stream) and whether it is charged as decode work (one load-only job per
commit region of idle rounds, the last one shorter). An operation that
claims the patch takes the rounds emitted since the last claim, so the
window manager can prepend them to its plan. Idle rounds are decoder
workload in every reference system (SWIPER 2412.05115, XQsim, Terhal's
backlog bound via Battistel 2303.00054).
"""

import dataclasses


@dataclasses.dataclass
class _PatchIdle:
    """One patch's idle rounds: since the last claim, since the last job."""

    unclaimed: int = 0
    uncharged: int = 0
    last_round_index: int = 0
    emitted: int = 0


class IdleRoundAccounting:
    """Routes each idle round by the policy and charges the decodes."""

    def __init__(self, policy, decode_queue, geometry_by_patch, streams, qpu):
        self.policy = policy
        self.decode_queue = decode_queue
        self.geometry_by_patch = geometry_by_patch
        self.streams = streams
        self.qpu = qpu
        self.operation_by_id: dict = {}
        self.idle_by_patch: dict = {}

    def load(self, program) -> None:
        """Know the operations an idle patch names."""
        for operation in program.operations:
            self.operation_by_id[operation.id] = operation

    @property
    def emitted_count(self) -> int:
        """Every idle round emitted so far, over all patches."""
        total = 0
        for idle in self.idle_by_patch.values():
            total += idle.emitted
        return total

    def emit_idle_round(self, operation_id, patch, round_index: int) -> None:
        """One idle cycle of a patch nobody is operating on.

        The round is produced, transmitted and accounted; the policy
        decides how it travels and whether it costs decode work. A patch
        on a live protected stream emits through that stream instead.
        """
        if self.streams.is_live_protected_patch(patch):
            return
        operation = self.operation_by_id[operation_id]
        self.policy.relay(self, operation, patch, round_index)
        idle = self._idle(patch)
        idle.unclaimed += 1
        idle.emitted += 1

    def claim(self, operation) -> int:
        """The idle rounds on the operation's patches since the last claim."""
        patches = operation.patches
        if not patches:
            patches = operation.qubits
        total = 0
        for patch in patches:
            idle = self.idle_by_patch.get(patch)
            if idle is None:
                continue
            total += idle.unclaimed
            idle.unclaimed = 0
        return total

    def end_idle_period(self, operation, patch) -> None:
        """An operation claims the patch: the policy settles its idle rounds."""
        self.policy.end_idle_period(self, operation, patch)

    # ---- what a policy may do with a round

    def emit_memory_round(self, operation, patch, round_index: int) -> None:
        """The round travels as a feedback-memory round of the operation."""
        self.qpu.emit_feedback_memory_round(operation.id, patch, round_index)

    def extend_live_stream(self, operation, patch) -> bool:
        """The round becomes the next round of the operation's live stream."""
        return self.streams.extend_live_stream(operation, patch)

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
