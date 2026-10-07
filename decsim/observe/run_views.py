"""A frozen view of a run's decode backlog.

The decode backlog sampler (observe/metrics.py) takes a view after every
action. The builder receives the owners it reads and writes nothing back.
"""

import dataclasses

import decsim.ports as ports
import decsim.records.identity as identity_records


@dataclasses.dataclass(frozen=True)
class BacklogView:
    """Decode backlog at one instant, per lane, per op, per patch, total.

    The rounds are syndrome rounds the QPU read out whose final
    correction is not in place yet.
    """

    ready_jobs: int  # jobs waiting across every queue
    per_lane: tuple  # ((lane, queued_jobs), ...); "" = default
    per_op_rounds: tuple  # ((operation_id, rounds_waiting), ...)
    per_patch_rounds: tuple  # ((patch, rounds_waiting), ...)
    total_rounds: int  # system-level depth


def backlog_view(
    window_manager: ports.WindowBacklog,
    decoder_managers: tuple,
    rounds_read_out_by_operation: dict,
) -> BacklogView:
    """Snapshot the job queues and the per-op, per-patch, system backlog.

    The queues are every pool's over both sides' managers.
    """
    waiting_by_pool = _waiting_by_pool(decoder_managers)
    ready_jobs = len(waiting_by_pool["default"])
    per_lane = [("", ready_jobs)]
    pool_items = waiting_by_pool.items()
    for lane, queue in sorted(pool_items):
        if lane == "default":
            continue
        queue_length = len(queue)
        per_lane.append((lane, queue_length))
        ready_jobs += queue_length
    final_rounds = window_manager.final_round_counts()
    per_op = []
    per_patch: dict = {}
    total_rounds = 0
    for operation_id, patch, final in final_rounds:
        read_out = rounds_read_out_by_operation.get(operation_id, 0)
        not_final = read_out - final
        waiting = max(0, not_final)
        per_op.append((operation_id, waiting))
        per_patch[patch] = per_patch.get(patch, 0) + waiting
        total_rounds += waiting
    per_op_rounds = sorted(per_op, key=_first_identity_order)
    patch_items = per_patch.items()
    per_patch_rounds = sorted(patch_items, key=_first_identity_order)
    return BacklogView(
        ready_jobs=ready_jobs,
        per_lane=tuple(per_lane),
        per_op_rounds=tuple(per_op_rounds),
        per_patch_rounds=tuple(per_patch_rounds),
        total_rounds=total_rounds,
    )


def _waiting_by_pool(decoder_managers) -> dict:
    """Every pool's ready queue, over both sides' managers."""
    waiting_by_pool = {}
    for manager in decoder_managers:
        waiting_by_pool[manager.pool.name] = manager.queue.waiting
    return waiting_by_pool


def _first_identity_order(item) -> tuple:
    return identity_records.stable_identity_bytes(item[0])
