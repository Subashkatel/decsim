"""Frozen views of a run's state.

The decode backlog sampler (observe/metrics.py) takes a view after every
action and the switching study reads one at the end of the run. Each
builder receives the owners it reads and writes nothing back; the flight
recorder that used to live here is observe/flight_recorder.py.
"""

import dataclasses

import decsim.message as message
import decsim.observe.decode_records as decode_records
import decsim.observe.window_ledger as window_ledger_module


@dataclasses.dataclass(frozen=True)
class BacklogView:
    """Decode backlog at one instant, per lane, per op, per patch, total.

    The rounds are syndrome rounds produced but not yet decoded.
    """

    ready_jobs: int  # jobs waiting across every queue
    per_lane: tuple  # ((lane, queued_jobs), ...); "" = default
    per_op_rounds: tuple  # ((op_id, rounds_waiting), ...)
    per_patch_rounds: tuple  # ((patch, rounds_waiting), ...)
    total_rounds: int  # system-level depth


@dataclasses.dataclass(frozen=True)
class SwitchingRecordsView:
    """The switching study's three tables at the end of the run."""

    windows: tuple[window_ledger_module.FinalWindowRow, ...]
    requests: tuple[decode_records.TerminalRequestRecord, ...]
    services: tuple[decode_records.TerminalServiceRecord, ...]


def backlog_view(
    window_manager, decoder_manager, include_rounds: bool = True
) -> BacklogView:
    """Snapshot the job queues and the per-op, per-patch, system backlog.

    include_rounds=False skips the rounds scan for an observer that only
    needs the queue depths.
    """
    waiting_by_pool = decoder_manager.queue.waiting_by_pool
    ready_jobs = len(waiting_by_pool["default"])
    per_lane = [("", ready_jobs)]
    pool_items = waiting_by_pool.items()
    for lane, queue in sorted(pool_items):
        if lane == "default":
            continue
        queue_length = len(queue)
        per_lane.append((lane, queue_length))
        ready_jobs += queue_length
    backlog = ()
    if include_rounds:
        backlog = window_manager.rounds_backlog()
    per_op = []
    per_patch: dict = {}
    total_rounds = 0
    for op_id, patch, waiting in backlog:
        per_op.append((op_id, waiting))
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


def switching_records_view(
    windows: window_ledger_module.WindowLedger,
    records: decode_records.DecodeRecordLedger,
) -> SwitchingRecordsView:
    """Compose the terminal owner facts without duplicating transfer timing."""
    final_rows = windows.final_rows()
    requests = tuple(records.requests)
    services = tuple(records.services)
    return SwitchingRecordsView(final_rows, requests, services)


def _first_identity_order(item) -> tuple:
    return message.stable_identity_order_key(item[0])
