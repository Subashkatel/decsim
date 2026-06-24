"""Read-only simulation metrics."""


from __future__ import annotations

from typing import TYPE_CHECKING

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

    def _units(self) -> tuple:
        """Return (busy, total) across all decoder pools."""
        totals = getattr(self.cluster, "unit_totals", None)
        if totals is None:
            return (self.cluster.num_units - self.cluster.free_units,
                    self.cluster.num_units)
        total = sum(totals.values())
        return total - sum(self.cluster.pool_free.values()), total

    def observe(self, engine: "Engine") -> None:
        """Add the busy level held since the last event, then read the current busy level."""
        self._busy_area += self._last_busy * (engine.now - self._t)
        self._t = engine.now
        self._last_busy = self._units()[0]

    def result(self) -> float:
        """Fraction of decoder-unit-time that was busy (0..1), across all pools."""
        total = self._units()[1]
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

    def _queued(self) -> int:
        """Jobs waiting across every decoder pool."""
        pools = getattr(self.cluster, "pool_ready", None)
        default_queue_length = len(self.cluster.ready)
        if pools is None:
            return default_queue_length
        return default_queue_length + sum(len(queue) for queue in pools.values())

    def observe(self, engine: "Engine") -> None:
        """Accumulate time-weighted queue length and track the peak."""
        self._area += self._last_len * (engine.now - self._t)
        self._t = engine.now
        self._last_len = self._queued()
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
        rows = []
        for (operation_id, window_index), window in sorted(self.cluster.windows.items()):
            stamps = (
                window.t_first_round,
                window.t_data_complete,
                window.t_queued,
                window.t_dispatch,
                window.t_done,
            )
            if any(stamp is None for stamp in stamps):
                continue
            rows.append({
                "op": operation_id,
                "window": window_index,
                "buffer_fill": window.t_data_complete - window.t_first_round,
                "dep_block": window.t_queued - window.t_data_complete,
                "queue_wait": window.t_dispatch - window.t_queued,
                "service": window.t_done - window.t_dispatch,
                "total": window.t_done - window.t_first_round,
            })
        return rows

    def result(self) -> dict:
        """Per-stage {mean, max, n} in ticks across all decoded windows."""
        rows = self.rows()
        stages = ("buffer_fill", "dep_block", "queue_wait", "service", "total")
        if not rows:
            return {stage: {"mean": 0.0, "max": 0, "n": 0} for stage in stages}
        return {
            stage: {
                "mean": sum(row[stage] for row in rows) / len(rows),
                "max": max(row[stage] for row in rows),
                "n": len(rows),
            }
            for stage in stages
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

    def _rounds_decoded(self, op_id: int) -> int:
        """Rounds decoded in an unbroken prefix from round 1."""
        committed_ranges = sorted(
            (self.cluster.windows[key].commit_lo, self.cluster.windows[key].commit_hi)
            for key in self.cluster.committed_windows
            if key[0] == op_id
        )
        decoded = 0
        for start_round, end_round in committed_ranges:
            if start_round <= decoded + 1:
                decoded = max(decoded, end_round)
            else:
                break
        return decoded

    def _backlog(self) -> int:
        """Total rounds waiting across all operations: arrived minus already-decoded."""
        return sum(max(0, self.cluster.rounds_arrived.get(op_id, 0)
                          - self._rounds_decoded(op_id))
                   for op_id in self.cluster.ops)

    def observe(self, engine: "Engine") -> None:
        """Sample the backlog and update peak, average, and trace."""
        self._area += self._last * (engine.now - self._t)
        self._t = engine.now
        self._last = self._backlog()
        self.peak = max(self.peak, self._last)
        if not self.trace or self.trace[-1][1] != self._last:
            self.trace.append((engine.now, self._last))

    def rows(self) -> list:
        """Backlog time series, one record per value change."""
        return [{"t": time_ticks, "backlog_rounds": backlog_rounds}
                for time_ticks, backlog_rounds in self.trace]

    def result(self) -> dict:
        """The largest backlog seen and the time-average, both in rounds waiting to be decoded."""
        return {"peak_rounds": self.peak,
                "time_avg_rounds": (self._area / self._t if self._t else 0.0)}


class StrongDecoderBacklog:
    """Outstanding strong-decoder jobs under decoder switching."""

    name = "strong_backlog"

    def __init__(self, cluster, pool: str = "strong"):
        self.cluster = cluster
        self.pool = pool
        self.peak_jobs = 0
        self._t = 0
        self._area = 0.0
        self._last = 0
        self.trace = []

    def _outstanding(self) -> int:
        """Strong jobs not yet finished: waiting on the strong pool plus in flight on it."""
        queued = len(self.cluster.pool_ready.get(self.pool, []))
        busy = (self.cluster.unit_totals.get(self.pool, 0)
                - self.cluster.pool_free.get(self.pool, 0))
        return queued + busy

    def observe(self, engine: "Engine") -> None:
        """Sample outstanding strong work and update peak, average, and trace."""
        self._area += self._last * (engine.now - self._t)
        self._t = engine.now
        self._last = self._outstanding()
        self.peak_jobs = max(self.peak_jobs, self._last)
        if not self.trace or self.trace[-1][1] != self._last:
            self.trace.append((engine.now, self._last))

    def rows(self) -> list:
        """Strong-backlog time series, one record per value change."""
        per_job = self.cluster.commit + 2 * self.cluster.buffer
        return [{"t": time_ticks, "jobs": jobs, "rounds": jobs * per_job}
                for time_ticks, jobs in self.trace]

    def result(self) -> dict:
        """Peak and time-average outstanding strong jobs."""
        return {"peak_jobs": self.peak_jobs,
                "time_avg_jobs": (self._area / self._t if self._t else 0.0),
                "strong_needed": getattr(self.cluster, "strong_needed", 0)}


class StrongBacklogRounds:
    """Outstanding strong-decoder work measured in rounds."""

    name = "strong_backlog_rounds"

    def __init__(self, cluster, pool: str = "strong"):
        self.cluster = cluster
        self.pool = pool
        self.peak_rounds = 0
        self._t = 0
        self._area = 0.0
        self._last = 0
        self.trace = []

    def _rounds(self) -> int:
        """Strong rounds waiting or running."""
        queued = sum(job.n_rounds for job in self.cluster.pool_ready.get(self.pool, []))
        return queued + getattr(self.cluster, "strong_running_rounds", 0)

    def observe(self, engine: "Engine") -> None:
        self._area += self._last * (engine.now - self._t)
        self._t = engine.now
        self._last = self._rounds()
        self.peak_rounds = max(self.peak_rounds, self._last)
        if not self.trace or self.trace[-1][1] != self._last:
            self.trace.append((engine.now, self._last))

    def rows(self) -> list:
        """Strong-round backlog time series, one record per value change."""
        return [{"t": time_ticks, "rounds": rounds}
                for time_ticks, rounds in self.trace]

    def result(self) -> dict:
        return {"peak_rounds": self.peak_rounds,
                "time_avg_rounds": (self._area / self._t if self._t else 0.0),
                "strong_needed": getattr(self.cluster, "strong_needed", 0)}


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
        chip = self.chip
        rows = []
        releases = sorted(chip.decode_release_time.items(),
                          key=lambda item: item[1])
        for operation_id, release_time in releases:
            operation = chip.ops[operation_id]
            blocked_start_time = chip.body_done_time.get(operation.blocked_by)
            if blocked_start_time is None:
                continue
            wait_time = release_time - blocked_start_time
            backlog_rounds = (
                wait_time / chip._round_ticks_for(operation)
                + chip.cluster.rounds_for(operation)
            )
            rows.append({
                "op": operation_id,
                "name": operation.name,
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
    """SWIPER-style wait time for feedback-blocked operations.

    The reported average divides by every conditional operation, not only the ones
    that finished. That matches SWIPER's reaction-time denominator.
    """

    name = "conditional_reaction_time"

    def __init__(self, chip, divergence_threshold_rounds: float | None = None,
                 require_all_released: bool = True):
        self.chip = chip
        self.divergence_threshold_rounds = divergence_threshold_rounds
        self.require_all_released = require_all_released

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample. The chip stamps body-done and release times."""
        return None

    def conditional_operation_ids(self) -> list[int]:
        """Operation ids that wait for an earlier decode result."""
        return [
            operation_id
            for operation_id, operation in sorted(self.chip.ops.items())
            if operation.blocked_by is not None
        ]

    def rows(self) -> list[dict]:
        """One released conditional operation with wait in ticks and rounds."""
        rows = []
        for operation_id in self.conditional_operation_ids():
            operation = self.chip.ops[operation_id]
            release_time = self.chip.decode_release_time.get(operation_id)
            blocking_done_time = self.chip.body_done_time.get(operation.blocked_by)
            if release_time is None or blocking_done_time is None:
                continue
            wait_ticks = release_time - blocking_done_time
            wait_rounds = wait_ticks / self.chip._round_ticks_for(operation)
            rows.append({
                "op": operation_id,
                "name": operation.name,
                "blocked_by": operation.blocked_by,
                "blocking_done_at": blocking_done_time,
                "released_at": release_time,
                "wait_ticks": wait_ticks,
                "wait_rounds": wait_rounds,
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
        if getattr(self.chip, "idle_cap_hits", []):
            return "idle-round cap reached"
        if self.require_all_released and released_count < total_count:
            return "not all conditionals released"
        return ""

    def result(self) -> dict:
        """SWIPER-style reaction-time summary for feedback-blocked operations."""
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
                row["op"]: row["wait_rounds"]
                for row in rows
            },
            "avg_conditioned_decode_wait_time": average,
            "mean_released_wait_rounds": mean_released,
            "max_wait_rounds": max_wait,
        }


class MagicStateLatency:
    """Magic-state distillation, correction-decode, delivery, and total latency."""

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


class LogicalErrorRate:
    """Per-shot logical-error verdict for real-decoding runs."""

    name = "logical_error_rate"

    def __init__(self, cluster, device):
        self.cluster = cluster
        self.device = device

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample before the final decode result exists."""
        return None

    def verdicts(self) -> dict:
        """Verdicts for operations that have both prediction and sampled truth."""
        truth = getattr(self.device, "_truth", {}) or {}
        verdicts = {}
        for operation_id, observables in truth.items():
            prediction = self.cluster.op_results.get(operation_id)
            if prediction is None:
                continue
            true_bit = int(observables[0])
            verdicts[operation_id] = {
                "predicted": int(prediction),
                "truth": true_bit,
                "error": int(int(prediction) != true_bit),
            }
        return verdicts

    def result(self) -> dict:
        """The per-op verdict for this shot (see verdicts())."""
        return self.verdicts()


class MemoryErrorPenalty:
    """Analytic logical-error penalty from measured idle memory rounds."""

    name = "memory_error_penalty"

    def __init__(self, cluster):
        self.cluster = cluster

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample: the counts are read at end-of-run."""
        return None

    def result(self) -> dict:
        """Return the total penalty and a per-operation breakdown."""
        per_op: dict = {}
        total = 0.0
        for operation_id, idle_rounds in getattr(self.cluster, "memory_rounds", {}).items():
            if idle_rounds <= 0:
                continue
            operation = self.cluster.ops.get(operation_id)
            patch = (operation.patches[0] if operation and operation.patches else
                     (operation.qubits[0] if operation and operation.qubits else operation_id))
            code = self.cluster.layout.code_for_patch(patch)
            memory_error = getattr(code, "memory_error", None)
            if memory_error is None:
                continue
            penalty = memory_error(idle_rounds)
            per_op[operation_id] = {
                "idle_rounds": idle_rounds,
                "d": code.distance,
                "penalty": penalty,
            }
            total += penalty
        return {"total": total, "per_op": per_op}
