"""Typed read-only metric views (spec §8.9).

Frozen snapshot dataclasses + the builders that populate them. Metrics
consume these views instead of reaching into live window_manager objects; the
engine invokes Metric.observe(...) only between events, so every
snapshot is taken at a consistent instant (principle 7: no mid-mutation
reads). The builders read the shared cluster/gate metric surface, so
they read the shared cluster surface (ClusterFacade).
"""

from __future__ import annotations

from dataclasses import dataclass

from .message import stable_identity_order_key

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
    """Per-window latency rows; `stages` is the pipeline stage sequence."""

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

    chip_done: object               # last physical-work tick (None if unfinished)
    fully_done: int                 # engine.now at snapshot
    body_done_time: tuple           # ((op_id, tick), ...)
    decode_release_time: tuple      # ((op_id, tick), ...)
    idle_cap_hits: tuple            # one record per capped patch
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


# ---------------------------------------------------------------- builders

def utilization_view(cluster) -> UtilizationView:
    """Snapshot decoder occupancy from the cluster metric surface."""
    totals = getattr(cluster, "unit_totals", None)
    if totals is None:
        busy = cluster.num_units - cluster.free_units
        return UtilizationView(busy, cluster.num_units,
                               (("", busy, cluster.num_units),))
    total = sum(totals.values())
    busy = total - sum(cluster.pool_free.values())
    per_pool = tuple(
        (name, totals[name] - cluster.pool_free.get(name, 0), totals[name])
        for name in sorted(totals))
    return UtilizationView(busy, total, per_pool)


def _rounds_decoded(cluster, op_id) -> int:
    """Rounds decoded in an unbroken prefix from round 1."""
    committed_ranges = sorted(
        (cluster.windows[key].commit_lo, cluster.windows[key].commit_hi)
        for key in cluster.committed_windows
        if key[0] == op_id)
    decoded = 0
    for start_round, end_round in committed_ranges:
        if start_round <= decoded + 1:
            decoded = max(decoded, end_round)
        else:
            break
    return decoded


def _patch_of(op, op_id):
    """The patch a metric attributes an op to."""
    if op is not None and op.patches:
        return op.patches[0]
    if op is not None and op.qubits:
        return op.qubits[0]
    return op_id


def backlog_view(cluster, include_rounds: bool = True) -> BacklogView:
    """Snapshot job queues + per-op/per-patch/system syndrome backlog.

    include_rounds=False skips the (comparatively costly) rounds scan for
    per-event observers that only need the job-queue depths."""
    pools = getattr(cluster, "pool_ready", None)
    ready_jobs = len(cluster.ready)
    per_lane = [("", ready_jobs)]
    if pools is not None:
        per_lane += [(lane, len(queue)) for lane, queue in sorted(pools.items())]
        ready_jobs += sum(len(queue) for queue in pools.values())

    per_op, per_patch = [], {}
    for op_id in sorted(
        (cluster.window_manager._ops if include_rounds else ()),
        key=stable_identity_order_key,
    ):
        waiting = max(0, cluster.rounds_arrived.get(op_id, 0)
                      - _rounds_decoded(cluster, op_id))
        per_op.append((op_id, waiting))
        patch = _patch_of(cluster.window_manager._ops.get(op_id), op_id)
        per_patch[patch] = per_patch.get(patch, 0) + waiting
    return BacklogView(ready_jobs=ready_jobs,
                       per_lane=tuple(per_lane),
                       per_op_rounds=tuple(sorted(
                           per_op, key=lambda item: stable_identity_order_key(item[0])
                       )),
                       per_patch_rounds=tuple(sorted(per_patch.items(),
                           key=lambda item: stable_identity_order_key(item[0]))),
                       total_rounds=sum(w for _, w in per_op))


def window_latency_view(cluster) -> WindowLatencyView:
    """Snapshot the per-window stage decomposition (fully-decoded only)."""
    rows = []
    for (op_id, window_index), window in sorted(
        cluster.windows.items(),
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


def reaction_view(gate) -> ReactionView:
    """Snapshot the reaction timeline from the gate (chip control half)."""
    ops = tuple(
        OpReactionInfo(op=op_id, name=op.name, blocked_by=op.blocked_by,
                       round_ticks=gate._round_ticks_for(op),
                       rounds=gate._round_count_for(op))
        for op_id, op in sorted(
            gate._ops.items(), key=lambda item: stable_identity_order_key(item[0])
        ))
    return ReactionView(
        chip_done=gate.last_finish_time,
        fully_done=gate.engine.now,
        body_done_time=tuple(sorted(
            gate.body_done_time.items(),
            key=lambda item: stable_identity_order_key(item[0]),
        )),
        decode_release_time=tuple(sorted(
            gate.decode_release_time.items(),
            key=lambda item: stable_identity_order_key(item[0]),
        )),
        idle_cap_hits=tuple(tuple(sorted(hit.items()))
                            for hit in gate.idle_cap_hits),
        ops=ops)


def truth_view(cluster, device) -> TruthView:
    """Snapshot sampled truth (device) next to published predictions."""
    truth = getattr(device, "_truth", {}) or {}
    observables = tuple(sorted(
        (
            (op_id, tuple(int(bit) for bit in bits))
            for op_id, bits in truth.items()
        ),
        key=lambda item: stable_identity_order_key(item[0]),
    ))
    predictions = tuple(sorted(
        cluster.op_results.items(),
        key=lambda item: stable_identity_order_key(item[0]),
    ))
    return TruthView(observables=observables, predictions=predictions)


def strong_work_view(cluster) -> StrongWorkView:
    """Compose exact global strong work from its two lifecycle owners."""
    pending = cluster.pending_strong_work_snapshot()
    admitted = cluster.admitted_strong_work_snapshot()
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
        strong_needed=cluster.strong_needed,
    )
