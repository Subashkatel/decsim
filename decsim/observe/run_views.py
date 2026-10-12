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

    The rounds are those of windows the QPU read out whole whose final
    correction is not in place yet.
    """

    ready_jobs: int  # jobs waiting across every queue
    per_lane: tuple  # ((pool, queued_jobs), ...), the entry pool first
    per_op_rounds: tuple  # ((operation_id, rounds_waiting), ...)
    per_patch_rounds: tuple  # ((patch, rounds_waiting), ...)
    total_rounds: int  # system-level depth


def backlog_view(
    window_manager: ports.WindowBacklog,
    decoder_managers: tuple,
    rounds_read_out_by_operation: dict,
) -> BacklogView:
    """Snapshot the job queues and the per-op, per-patch, system backlog.

    The queues are every pool's, each lane by its pool's name: the entry
    pool's first, then the others in name order.
    """
    entry_manager = decoder_managers[0]
    ready_jobs = len(entry_manager.queue.waiting)
    per_lane = [(entry_manager.pool.name, ready_jobs)]
    other_managers = decoder_managers[1:]
    waiting_by_pool = _waiting_by_pool(other_managers)
    pool_items = waiting_by_pool.items()
    for lane, queue in sorted(pool_items):
        queue_length = len(queue)
        per_lane.append((lane, queue_length))
        ready_jobs += queue_length
    backlog_rounds = window_manager.backlog_round_counts(
        rounds_read_out_by_operation
    )
    per_op = []
    per_patch: dict = {}
    total_rounds = 0
    for operation_id, patch, waiting in backlog_rounds:
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
    """Each of these managers' ready queues, by its pool's name."""
    waiting_by_pool = {}
    for manager in decoder_managers:
        waiting_by_pool[manager.pool.name] = manager.queue.waiting
    return waiting_by_pool


def _first_identity_order(item) -> tuple:
    return identity_records.stable_identity_bytes(item[0])
