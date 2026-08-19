"""Typed read-only metric views.

Frozen snapshot dataclasses and the builders that populate them. Metrics
consume these views instead of reaching into owners; the engine invokes
Metric.observe(...) only between events, so every snapshot is taken at a
consistent instant. Each builder receives the state owners it consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..decoders.decoder_memory import DecoderMemorySnapshot
from ..decoders.decoder_manager import TerminalRequestRecord, TerminalServiceRecord
from ..message import DecoderRequestKey, stable_identity_order_key


WINDOW_STAGES = ("buffer_fill", "dep_block", "queue_wait", "service", "total")


@dataclass(frozen=True)
class UtilizationView:
    """Decoder-unit occupancy across all pools at one instant."""

    busy_units: int
    total_units: int
    per_pool: tuple                 # ((pool_name, busy, total), ...)


@dataclass(frozen=True)
class BacklogView:
    """Decode backlog at one instant, per lane and hierarchical
    (op -> patch -> system) in syndrome rounds."""

    ready_jobs: int                 # jobs waiting across every queue
    per_lane: tuple                 # ((lane, queued_jobs), ...); "" = default
    per_op_rounds: tuple            # ((op_id, rounds_waiting), ...)
    per_patch_rounds: tuple         # ((patch, rounds_waiting), ...)
    total_rounds: int               # system-level depth


@dataclass(frozen=True)
class WindowStageRow:
    """One fully-decoded window's latency decomposition."""

    op: int
    window: int
    buffer_fill: int
    dep_block: int
    queue_wait: int
    service: int
    total: int


@dataclass(frozen=True)
class WindowLatencyView:
    """Per-window latency rows; `stages` is the pipeline stage sequence.

    ``queue_wait`` is the ready-queue wait for a unit; ``service`` runs from
    unit assignment through the input transfer into that unit's memory to the
    end of the decode.
    """

    stages: tuple = WINDOW_STAGES
    rows: tuple = ()                # (WindowStageRow, ...)


@dataclass(frozen=True)
class OpReactionInfo:
    """Static per-op facts the reaction metrics need."""

    op: int
    name: str
    blocked_by: object              # op id or None
    round_ticks: int
    rounds: int


@dataclass(frozen=True)
class ReactionView:
    """The reaction-path timeline of one run."""

    execution_done: object               # last physical-work tick (None if unfinished)
    fully_done: int                 # engine.now at snapshot
    body_done_time: tuple           # ((op_id, tick), ...)
    decode_release_time: tuple      # ((op_id, tick), ...)
    ops: tuple                      # (OpReactionInfo, ...)


@dataclass(frozen=True)
class TruthView:
    """Sampled ground truth next to the decoder's published predictions."""

    observables: tuple              # ((op_id, (bit, ...)), ...)
    predictions: tuple              # ((op_id, predicted), ...)


@dataclass(frozen=True)
class StrongWorkPhaseView:
    """One closed phase of outstanding strong decoder input."""

    jobs: int
    full_input_rounds: int


@dataclass(frozen=True)
class StrongWorkView:
    """Global assigned strong work before and after decoder admission."""

    waiting_far_boundary: StrongWorkPhaseView
    waiting_terminal_data: StrongWorkPhaseView
    in_transit: StrongWorkPhaseView
    queued: StrongWorkPhaseView
    running: StrongWorkPhaseView
    total_jobs: int
    total_full_input_rounds: int
    strong_needed: int


@dataclass(frozen=True)
class FinalWindowRow:
    destination_key: tuple[object, int]
    weak_buffer_lo: int
    weak_commit_lo: int
    weak_commit_hi: int
    weak_buffer_hi: int
    final_commit_lo: Optional[int]
    final_commit_hi: Optional[int]
    window_disposition: str
    absorbed_into: Optional[tuple[object, int]]
    selected_request_key: Optional[DecoderRequestKey]


@dataclass(frozen=True)
class DecoderMemoryView:
    """Every decoder unit's input memory at one instant, one row per unit."""

    per_unit: tuple[DecoderMemorySnapshot, ...]


@dataclass(frozen=True)
class SwitchingRecordsView:
    windows: tuple[FinalWindowRow, ...]
    requests: tuple[TerminalRequestRecord, ...]
    services: tuple[TerminalServiceRecord, ...]


def utilization_view(decoder_manager) -> UtilizationView:
    """Snapshot decoder occupancy."""
    totals = getattr(decoder_manager, "unit_totals", None)
    if totals is None:
        busy = decoder_manager.num_units - decoder_manager.free_units
        return UtilizationView(busy, decoder_manager.num_units,
                               (("", busy, decoder_manager.num_units),))
    total = sum(totals.values())
    busy = total - sum(decoder_manager.pool_free.values())
    per_pool = tuple(
        (name, totals[name] - decoder_manager.pool_free.get(name, 0), totals[name])
        for name in sorted(totals))
    return UtilizationView(busy, total, per_pool)


def backlog_view(window_manager, decoder_manager,
                 include_rounds: bool = True) -> BacklogView:
    """Snapshot job queues + per-op/per-patch/system syndrome backlog.

    include_rounds=False skips the (comparatively costly) rounds scan for
    per-event observers that only need the job-queue depths."""
    pools = getattr(decoder_manager, "pool_ready", None)
    ready_jobs = len(decoder_manager.ready)
    per_lane = [("", ready_jobs)]
    if pools is not None:
        per_lane += [(lane, len(queue)) for lane, queue in sorted(pools.items())]
        ready_jobs += sum(len(queue) for queue in pools.values())

    per_op, per_patch = [], {}
    for op_id, patch, waiting in (window_manager.rounds_backlog() if include_rounds else ()):
        per_op.append((op_id, waiting))
        per_patch[patch] = per_patch.get(patch, 0) + waiting
    return BacklogView(ready_jobs=ready_jobs,
                       per_lane=tuple(per_lane),
                       per_op_rounds=tuple(sorted(
                           per_op, key=lambda item: stable_identity_order_key(item[0])
                       )),
                       per_patch_rounds=tuple(sorted(per_patch.items(),
                           key=lambda item: stable_identity_order_key(item[0]))),
                       total_rounds=sum(w for _, w in per_op))


def window_latency_view(window_manager) -> WindowLatencyView:
    """Snapshot the per-window stage decomposition (fully-decoded only)."""
    rows = []
    for (op_id, window_index), window in sorted(
        window_manager.windows.items(),
        key=lambda item: stable_identity_order_key(item[0]),
    ):
        stamps = (window.t_first_round, window.t_data_complete,
                  window.t_queued, window.t_dispatch, window.t_done)
        if any(stamp is None for stamp in stamps):
            continue
        rows.append(WindowStageRow(
            op=op_id, window=window_index,
            buffer_fill=window.t_data_complete - window.t_first_round,
            dep_block=window.t_queued - window.t_data_complete,
            queue_wait=window.t_dispatch - window.t_queued,
            service=window.t_done - window.t_dispatch,
            total=window.t_done - window.t_first_round))
    return WindowLatencyView(rows=tuple(rows))


def reaction_view(execution_runtime) -> ReactionView:
    """Snapshot execution admission and controller cadence state."""
    controller = execution_runtime.controller
    ops = tuple(
        OpReactionInfo(op=op_id, name=op.name, blocked_by=op.blocked_by,
                       round_ticks=controller.round_ticks_for(op),
                       rounds=controller.round_count_for(op))
        for op_id, op in sorted(
            execution_runtime.operations.items(),
            key=lambda item: stable_identity_order_key(item[0])))
    return ReactionView(
        execution_done=execution_runtime.last_finish_time,
        fully_done=execution_runtime.engine.now,
        body_done_time=tuple(sorted(
            execution_runtime.body_done_time.items(),
            key=lambda item: stable_identity_order_key(item[0]))),
        decode_release_time=tuple(sorted(
            execution_runtime.decode_release_time.items(),
            key=lambda item: stable_identity_order_key(item[0]))),
        ops=ops)

def truth_view(window_manager, device) -> TruthView:
    """Snapshot sampled truth (device) next to published predictions."""
    sampled_truth = getattr(device, "sampled_truth", None)
    truth = sampled_truth() if sampled_truth is not None else {}
    observables = tuple(sorted(truth.items(),
                               key=lambda item: stable_identity_order_key(item[0])))
    predictions = tuple(sorted(
        window_manager.op_results.items(),
        key=lambda item: stable_identity_order_key(item[0]),
    ))
    return TruthView(observables=observables, predictions=predictions)


def strong_work_view(window_manager, decoder_manager) -> StrongWorkView:
    """Compose exact global strong work from its two lifecycle owners."""
    pending = window_manager.escalation.pending_strong_work_snapshot()
    admitted = decoder_manager.admitted_strong_work_snapshot()
    pending_keys = {key for key, _, _ in pending}
    admitted_keys = {key for keys, _, _ in admitted for key in keys}
    overlap = pending_keys & admitted_keys
    if overlap:
        raise RuntimeError(f"strong work has overlapping owners for {overlap!r}")

    values = {
        phase: [0, 0]
        for phase in (
            "waiting_far_boundary", "waiting_terminal_data",
            "in_transit", "queued", "running",
        )
    }
    for _, phase, rounds in pending:
        values[phase][0] += 1
        values[phase][1] += rounds
    for _, phase, rounds in admitted:
        values[phase][0] += 1
        values[phase][1] += rounds
    phases = {
        phase: StrongWorkPhaseView(*counts)
        for phase, counts in values.items()
    }
    return StrongWorkView(
        **phases,
        total_jobs=sum(value.jobs for value in phases.values()),
        total_full_input_rounds=sum(
            value.full_input_rounds for value in phases.values()
        ),
        strong_needed=decoder_manager.strong_needed,
    )


def decoder_memory_view(decoder_manager) -> DecoderMemoryView:
    """Snapshot every unit's input memory."""
    return DecoderMemoryView(per_unit=tuple(
        memory.snapshot()
        for _key, memory in sorted(decoder_manager.decoder_memories.items())))


def switching_records_view(window_manager, decoder_manager) -> SwitchingRecordsView:
    """Compose terminal owner facts without duplicating transfer timing."""
    rows = []
    for key, window in sorted(window_manager.windows.items(),
                              key=lambda item: stable_identity_order_key(item[0])):
        contribution = window_manager.ledger.contributions.get(key)
        absorbed = key in window_manager.absorbed_windows
        absorbed_into = None
        if absorbed:
            owners = [owner for owner, value in
                      window_manager.ledger.contributions.items()
                      if value.ownership_kind == "strong_slab"
                      and value.commit_lo <= window.commit_lo
                      and value.commit_hi >= window.commit_hi]
            if len(owners) != 1:
                raise RuntimeError(f"absorbed window {key} has no unique owner")
            absorbed_into = owners[0]
        elif contribution is None:
            raise RuntimeError(f"final window {key} has no logical contribution")
        rows.append(FinalWindowRow(
            key, window.start_round, window.commit_lo, window.commit_hi,
            window.buffer_hi,
            None if absorbed else contribution.commit_lo,
            None if absorbed else contribution.commit_hi,
            "absorbed" if absorbed else contribution.ownership_kind,
            absorbed_into, None if absorbed else
            window_manager.selected_request_key(key)))
    return SwitchingRecordsView(
        tuple(rows), decoder_manager.terminal_request_records_snapshot(),
        decoder_manager.terminal_service_records_snapshot())
