"""Read-only simulation metrics.

Every metric is an observer over the typed views in views.py (spec §8.9,
principle 7): observe()/result() take a frozen snapshot of the core
surface and compute from the view alone. Public numbers are unchanged
from the pre-view implementations.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

from .views import (WINDOW_STAGES, backlog_view, reaction_view,
                    strong_pool_view, utilization_view,
                    window_latency_view)

if TYPE_CHECKING:
    from .engine import Engine


class DecoderUtilization:
    """Time-weighted fraction of decoder units that were busy."""

    name = "decoder_utilization"

    def __init__(self, cluster):
        self.cluster = cluster
        self._t = 0
        self._busy_area = 0.0
        self._last_busy = 0

    def observe(self, engine: "Engine") -> None:
        """Add the busy level held since the last event, then re-sample it."""
        view = utilization_view(self.cluster)
        self._busy_area += self._last_busy * (engine.now - self._t)
        self._t = engine.now
        self._last_busy = view.busy_units

    def result(self) -> float:
        """Fraction of decoder-unit-time that was busy (0..1), across all pools."""
        total = utilization_view(self.cluster).total_units
        return self._busy_area / (total * self._t) if self._t else 0.0


class ReadyQueueStats:
    """Peak and time-average number of decode jobs waiting."""

    name = "ready_queue"

    def __init__(self, cluster):
        self.cluster = cluster
        self._t = 0
        self._area = 0.0
        self._last_len = 0
        self.peak = 0

    def observe(self, engine: "Engine") -> None:
        """Accumulate time-weighted queue length and track the peak."""
        view = backlog_view(self.cluster, include_rounds=False)
        self._area += self._last_len * (engine.now - self._t)
        self._t = engine.now
        self._last_len = view.ready_jobs
        self.peak = max(self.peak, self._last_len)

    def result(self) -> dict:
        """Peak and time-average ready-queue length."""
        return {"peak": self.peak, "time_avg": (self._area / self._t if self._t else 0.0)}


class WindowLatencyBreakdown:
    """Per-window buffer, dependency, queue, and service latency."""

    name = "window_latency"

    def __init__(self, cluster):
        self.cluster = cluster

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; the cluster stamps the windows)."""
        return None

    def rows(self) -> list:
        """One record per fully-decoded window: op, window index, and the four stages."""
        view = window_latency_view(self.cluster)
        return [{
            "op": row.op,
            "window": row.window,
            "buffer_fill": row.buffer_fill,
            "dep_block": row.dep_block,
            "queue_wait": row.queue_wait,
            "service": row.service,
            "total": row.total,
        } for row in view.rows]

    def result(self) -> dict:
        """Per-stage {mean, max, n} in ticks across all decoded windows."""
        rows = self.rows()
        if not rows:
            return {stage: {"mean": 0.0, "max": 0, "n": 0}
                    for stage in WINDOW_STAGES}
        return {
            stage: {
                "mean": sum(row[stage] for row in rows) / len(rows),
                "max": max(row[stage] for row in rows),
                "n": len(rows),
            }
            for stage in WINDOW_STAGES
        }


class DecodeBacklog:
    """Rounds of syndrome data produced but not yet decoded."""

    name = "decode_backlog"

    def __init__(self, cluster):
        self.cluster = cluster
        self._t = 0
        self._area = 0.0
        self._last = 0
        self.peak = 0
        self.trace = []

    def observe(self, engine: "Engine") -> None:
        """Sample the backlog and update peak, average, and trace."""
        view = backlog_view(self.cluster)
        self._area += self._last * (engine.now - self._t)
        self._t = engine.now
        self._last = view.total_rounds
        self.peak = max(self.peak, self._last)
        if not self.trace or self.trace[-1][1] != self._last:
            self.trace.append((engine.now, self._last))

    def rows(self) -> list:
        """Backlog time series, one record per value change."""
        return [{"t": time_ticks, "backlog_rounds": backlog_rounds}
                for time_ticks, backlog_rounds in self.trace]

    def result(self) -> dict:
        """Peak and time-average backlog, in rounds waiting to be decoded."""
        return {"peak_rounds": self.peak,
                "time_avg_rounds": (self._area / self._t if self._t else 0.0)}


class BacklogEarlyWarning:
    """Divergence early warning over the hierarchical backlog ledger.

    Event-driven observer (like DecodeBacklog) that cuts wall time into
    fixed bins of window_ticks and estimates the backlog growth rate
    per bin in rounds-of-backlog per round of wall time — the same f
    as the QLX stall model f = (lat - cycle)/cycle (see
    tests/test_qlx_backlog_crosscheck.py). WARNS (latched) at the end
    of the k-th consecutive bin whose slope is STRICTLY greater than
    threshold_f (integer backlog quantizes slope to
    round_ticks/window_ticks; ">=" would fire on steady-state jitter),
    and attributes the warning to the patch(es) with the largest
    positive backlog slope over those k bins.

    Initialization note: the bin-start baseline seeds from the first
    observe() only when it lands at tick 0 (true for engine runs,
    which always start at t=0); a first event later than tick 0
    treats the pre-event backlog as 0, which is correct in-engine but
    makes synthetic traces that start mid-stream show an artificial
    first-bin slope.
    """

    name = "backlog_early_warning"

    def __init__(self, cluster, round_ticks: int, window_ticks: int,
                 threshold_f: float = 0.1, consecutive: int = 2):
        self.cluster = cluster
        self.round_ticks = int(round_ticks)
        self.window_ticks = int(window_ticks)
        self.threshold_f = float(threshold_f)
        self.consecutive = int(consecutive)
        self.warned = False
        self.t_warn = None
        self.attribution = ()
        self.slopes = []                 # (bin_end_tick, slope_f)
        self._bin_start = 0
        self._start_backlog = 0
        self._start_per_patch = {}
        self._last_view = None
        self._streak = 0
        self._streak_start_per_patch = {}

    def _slope(self, delta_rounds: int, ticks: int) -> float:
        return delta_rounds * self.round_ticks / ticks if ticks else 0.0

    def observe(self, engine: "Engine") -> None:
        """Close every bin the current event time has passed.

        Right-continuous convention: an event landing exactly on a bin
        boundary counts in the ENDING bin; bins strictly before now use
        the value held since the previous event.
        """
        view = backlog_view(self.cluster)
        if self._last_view is None and engine.now == self._bin_start:
            self._start_backlog = view.total_rounds
            self._start_per_patch = dict(view.per_patch_rounds)
        while engine.now >= self._bin_start + self.window_ticks:
            bin_end = self._bin_start + self.window_ticks
            if engine.now == bin_end or self._last_view is None:
                end_view = view
            else:
                end_view = self._last_view
            self._close_bin(bin_end, end_view)
        self._last_view = view

    def _close_bin(self, bin_end: int, view) -> None:
        slope = self._slope(view.total_rounds - self._start_backlog,
                            self.window_ticks)
        self.slopes.append((bin_end, slope))
        # STRICT >: backlog is integer-valued, so one round per bin
        # quantizes slope to round_ticks/window_ticks (= 0.1 at the
        # default parameters); ">=" would fire on steady-state jitter
        if slope > self.threshold_f and not self.warned:
            if self._streak == 0:
                self._streak_start_per_patch = dict(self._start_per_patch)
            self._streak += 1
            if self._streak >= self.consecutive:
                self.warned = True
                self.t_warn = bin_end
                per_patch = dict(view.per_patch_rounds)
                patches = set(per_patch) | set(self._streak_start_per_patch)
                deltas = {p: per_patch.get(p, 0)
                          - self._streak_start_per_patch.get(p, 0)
                          for p in patches}
                worst = max(deltas.values(), default=0)
                self.attribution = tuple(sorted(
                    p for p, dv in deltas.items() if dv == worst and dv > 0))
        else:
            self._streak = 0
        self._bin_start = bin_end
        self._start_backlog = view.total_rounds
        self._start_per_patch = dict(view.per_patch_rounds)

    def result(self) -> dict:
        """Warning verdict, timing, attribution, and the slope trace."""
        return {"warned": self.warned,
                "t_warn_ticks": self.t_warn,
                "bins_evaluated": len(self.slopes),
                "max_slope": max((s for _, s in self.slopes), default=0.0),
                "mean_slope": (sum(s for _, s in self.slopes)
                               / len(self.slopes) if self.slopes else 0.0),
                "attribution": list(self.attribution),
                "slopes": [{"t": t, "slope_f": s} for t, s in self.slopes]}


class BurstEscalationDetector:
    """Passive burst detector: per-patch escalation counters + quorum.

    Consumes per-patch detection-event counts in fixed time bins
    (ingest_bin). A patch ESCALATES in a bin when its count exceeds
    mu + z*sigma of its own trailing baseline (trailing `baseline_bins`
    bins, after `warmup_bins` of history); the detector FIRES (latched)
    when at least `patch_quorum` patches escalate in the SAME bin —
    cosmic-ray-class bursts are spatially correlated (McEwen V16:
    5-10 of 26 qubits), isolated single-patch noise is not a burst.
    Escalated bins are excluded from the trailing baseline, so an
    ABRUPT burst cannot poison its own reference while it stays hot.

    KNOWN BLIND SPOT (design property, Codex P6 review 2026-07-04):
    a SLOW ramp that stays below z*sigma per bin is appended into the
    baseline and ratchets it upward indefinitely — the detector
    targets McEwen-class abrupt onsets (rise << bin at fitted
    profiles) and will not fire on gradual drifts; pair it with a
    trend rule (e.g. BacklogEarlyWarning-style slopes) if slow drift
    matters. Missing patch keys in a bin are treated as ZERO events:
    live adapters must not feed absent telemetry as an empty dict.

    Standalone by design (Gate 7 P6b): wire-up to live decsim
    detection streams is a thin observe() adapter left to the
    integration that needs it; validation here is against injected
    profiles fitted from the McEwen FAST artifact.
    """

    name = "burst_escalation_detector"

    def __init__(self, patches, z: float = 6.0, baseline_bins: int = 100,
                 warmup_bins: int = 30, patch_quorum: int = 3):
        self.patches = list(patches)
        self.z = float(z)
        self.baseline_bins = int(baseline_bins)
        self.warmup_bins = int(warmup_bins)
        self.patch_quorum = int(patch_quorum)
        self.fired = False
        self.fired_bin = None
        self.fired_patches = ()
        self.bin_index = -1
        self._history = {p: [] for p in self.patches}
        self.escalations = []            # (bin, tuple(patches))

    def _escalated(self, patch, count) -> bool:
        hist = self._history[patch]
        if len(hist) < self.warmup_bins:
            return False
        base = hist[-self.baseline_bins:]
        mu = sum(base) / len(base)
        var = sum((c - mu) ** 2 for c in base) / len(base)
        return count > mu + self.z * (var ** 0.5)

    def ingest_bin(self, counts: dict) -> bool:
        """Feed one bin of per-patch counts; returns fired-this-bin."""
        self.bin_index += 1
        hot = tuple(p for p in self.patches
                    if self._escalated(p, counts.get(p, 0)))
        for p in self.patches:
            if p not in hot:             # keep the baseline burst-free
                self._history[p].append(counts.get(p, 0))
        if hot:
            self.escalations.append((self.bin_index, hot))
        if len(hot) >= self.patch_quorum and not self.fired:
            self.fired = True
            self.fired_bin = self.bin_index
            self.fired_patches = hot
            return True
        return False

    def result(self) -> dict:
        return {"fired": self.fired, "fired_bin": self.fired_bin,
                "fired_patches": list(self.fired_patches),
                "bins": self.bin_index + 1,
                "escalation_bins": len(self.escalations)}


class StrongDecoderBacklog:
    """Outstanding strong-decoder jobs under decoder switching."""

    name = "strong_backlog"

    def __init__(self, cluster, pool: str = "strong"):
        if type(pool) is not str or not pool:
            raise TypeError("pool must be a nonempty built-in str")
        self.cluster = cluster
        self.pool = pool
        self.peak_jobs = 0
        self._t = 0
        self._area = 0.0
        self._last = 0
        self.trace = []

    def run_manifest_config(self):
        return {"pool": self.pool}

    def observe(self, engine: "Engine") -> None:
        """Sample outstanding strong work and update peak, average, and trace."""
        view = strong_pool_view(self.cluster, self.pool)
        self._area += self._last * (engine.now - self._t)
        self._t = engine.now
        self._last = view.queued_jobs + view.busy_units
        self.peak_jobs = max(self.peak_jobs, self._last)
        if not self.trace or self.trace[-1][1] != self._last:
            self.trace.append((engine.now, self._last))

    def rows(self) -> list:
        """Strong-backlog time series, one record per value change.

        SCOPE: `rounds` is a job count times the configured nominal redo size
        (commit + 2*buffer). It is NOT the decoder input a strong job is billed
        for: a double-window slab reads one buffer of context per face and is
        priced for that wider extent (see Switching), and an end-clamped slab or
        a bulk batch differs again. Theorem 1 of arXiv:2510.25222 bounds
        tau_strong against unprocessed rounds of decoder INPUT, so this series
        may not be published as that quantity."""
        per_job = strong_pool_view(self.cluster, self.pool).redo_rounds
        return [{"t": time_ticks, "jobs": jobs, "rounds": jobs * per_job}
                for time_ticks, jobs in self.trace]

    def result(self) -> dict:
        """Peak and time-average outstanding strong jobs."""
        view = strong_pool_view(self.cluster, self.pool)
        return {"peak_jobs": self.peak_jobs,
                "time_avg_jobs": (self._area / self._t if self._t else 0.0),
                "strong_needed": view.strong_needed}


class BacklogTrajectory:
    """Per-feedback-gate reaction wait and backlog in rounds."""

    name = "backlog_trajectory"

    def __init__(self, chip):
        self.chip = chip

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; the chip stamps the timestamps)."""
        return None

    def rows(self) -> list:
        """One record per released feedback-blocked gate."""
        view = reaction_view(self.chip)
        body_done = dict(view.body_done_time)
        info = {op.op: op for op in view.ops}
        rows = []
        releases = sorted(view.decode_release_time, key=lambda item: item[1])
        for operation_id, release_time in releases:
            op = info[operation_id]
            blocked_start_time = body_done.get(op.blocked_by)
            if blocked_start_time is None:
                continue
            wait_time = release_time - blocked_start_time
            backlog_rounds = wait_time / op.round_ticks + op.rounds
            rows.append({
                "op": operation_id,
                "name": op.name,
                "released_at": release_time,
                "wait": wait_time,
                "backlog_rounds": backlog_rounds,
            })
        return rows

    def result(self) -> dict:
        """Summary over all released gates: count, mean/max wait, mean/max backlog."""
        rows = self.rows()
        if not rows:
            return {"n": 0, "mean_wait": 0.0, "max_wait": 0,
                    "mean_backlog_rounds": 0.0, "max_backlog_rounds": 0.0}
        waits = [row["wait"] for row in rows]
        backlog = [row["backlog_rounds"] for row in rows]
        return {"n": len(rows),
                "mean_wait": sum(waits) / len(waits), "max_wait": max(waits),
                "mean_backlog_rounds": sum(backlog) / len(backlog),
                "max_backlog_rounds": max(backlog)}


class ConditionalReactionTime:
    """Reaction-time wait for feedback-blocked operations."""

    # ref: SWIPER; average divides by every conditional op, not only finished ones.

    name = "conditional_reaction_time"

    def __init__(self, chip, divergence_threshold_rounds: float | None = None,
                 require_all_released: bool = True):
        self.chip = chip
        self.divergence_threshold_rounds = divergence_threshold_rounds
        self.require_all_released = require_all_released

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample. The chip stamps body-done and release times."""
        return None

    def _view(self):
        return reaction_view(self.chip)

    def conditional_operation_ids(self) -> list[int]:
        """Operation ids that wait for an earlier decode result."""
        return [op.op for op in self._view().ops if op.blocked_by is not None]

    def rows(self) -> list[dict]:
        """One released conditional operation with wait in ticks and rounds."""
        view = self._view()
        released = dict(view.decode_release_time)
        body_done = dict(view.body_done_time)
        rows = []
        for op in view.ops:
            if op.blocked_by is None:
                continue
            release_time = released.get(op.op)
            blocking_done_time = body_done.get(op.blocked_by)
            if release_time is None or blocking_done_time is None:
                continue
            wait_ticks = release_time - blocking_done_time
            rows.append({
                "op": op.op,
                "name": op.name,
                "blocked_by": op.blocked_by,
                "blocking_done_at": blocking_done_time,
                "released_at": release_time,
                "wait_ticks": wait_ticks,
                "wait_rounds": wait_ticks / op.round_ticks,
            })
        return rows

    def pending_operation_ids(self) -> list[int]:
        """Conditional operations that never received a decode release."""
        released = {row["op"] for row in self.rows()}
        return [
            operation_id
            for operation_id in self.conditional_operation_ids()
            if operation_id not in released
        ]

    def _threshold_failure(self, max_wait_rounds: float) -> str:
        threshold = self.divergence_threshold_rounds
        if threshold is None or threshold <= 0:
            return ""
        if max_wait_rounds <= threshold:
            return ""
        return f"conditioned wait exceeded {threshold} rounds"

    def _failure_reason(self, released_count: int, total_count: int,
                        max_wait_rounds: float) -> str:
        threshold_reason = self._threshold_failure(max_wait_rounds)
        if threshold_reason:
            return threshold_reason
        if self._view().idle_cap_hits:
            return "idle-round cap reached"
        if self.require_all_released and released_count < total_count:
            return "not all conditionals released"
        return ""

    def result(self) -> dict:
        """Reaction-time summary for feedback-blocked operations."""
        total_count = len(self.conditional_operation_ids())
        rows = self.rows()
        released_count = len(rows)
        wait_sum = sum(row["wait_rounds"] for row in rows)
        max_wait = max((row["wait_rounds"] for row in rows), default=0.0)
        mean_released = wait_sum / released_count if released_count else 0.0
        average = wait_sum / total_count if total_count else 0.0
        failure_reason = self._failure_reason(released_count, total_count, max_wait)
        return {
            "success": failure_reason == "",
            "failed": failure_reason != "",
            "diverged": bool(self._threshold_failure(max_wait)),
            "failure_reason": failure_reason,
            "total_conditionals": total_count,
            "released_conditionals": released_count,
            "pending_conditionals": self.pending_operation_ids(),
            "conditioned_decode_wait_times": {
                str(row["op"]): row["wait_rounds"]
                for row in rows
            },
            "avg_conditioned_decode_wait_time": average,
            "mean_released_wait_rounds": mean_released,
            "max_wait_rounds": max_wait,
        }


class MagicStateLatency:
    """Magic-state distillation, correction-decode, delivery, and total latency."""

    # StateTrace is already a typed per-state record, i.e. the factory's view.

    name = "magic_state_latency"

    def __init__(self, factory):
        self.factory = factory

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (the factory stamps each state's StateTrace)."""
        return None

    def result(self) -> dict:
        """Per-stage {mean, max, n} in ticks across all delivered states."""
        traces = [t for t in getattr(self.factory, "traces", [])
                  if t.t_delivered is not None and t.t_corr_done is not None]
        stages = ("distill", "corr_decode", "deliver", "total")
        if not traces:
            return {stage: {"mean": 0.0, "max": 0, "n": 0} for stage in stages}
        values = {"distill": [trace.t_phys_done - trace.t_distill_start for trace in traces],
                  "corr_decode": [trace.t_corr_done - trace.t_phys_done for trace in traces],
                  "deliver": [trace.t_delivered - trace.t_corr_done for trace in traces],
                  "total": [trace.t_delivered - trace.t_distill_start for trace in traces]}
        return {stage: {"mean": sum(stage_values) / len(stage_values),
                        "max": max(stage_values), "n": len(stage_values)}
                for stage, stage_values in values.items()}
