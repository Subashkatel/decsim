"""Read-only simulation metrics.

Every metric is an observer over the typed views in views.py: observe() and
result() take a frozen snapshot of the core surface and compute from the
view alone; a metric never writes to the run.
"""


from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Optional

from ..message import (DecoderRequestKey, stable_identity_json,
                      stable_identity_order_key)
from .run_views import (WINDOW_STAGES, backlog_view, decoder_memory_view,
                    reaction_view, strong_work_view, utilization_view,
                    switching_records_view, window_latency_view)

if TYPE_CHECKING:
    from ..engine import Engine


class WindowSwitchingRecords:
    name = "window_switching_records"
    result_schema_version = 1

    def __init__(self, window_manager, decoder_manager):
        self.window_manager = window_manager
        self.decoder_manager = decoder_manager

    def observe(self, engine: "Engine") -> None:
        return None

    def result(self) -> dict:
        view = switching_records_view(self.window_manager, self.decoder_manager)

        def json_value(value):
            if isinstance(value, Enum):
                return value.value
            if type(value) is DecoderRequestKey:
                return {"operation_id": stable_identity_json(value.operation_id),
                        "window_id": value.window_id, "tier": value.tier.value,
                        "run_sequence": value.run_sequence}
            if is_dataclass(value):
                return {field.name: json_value(getattr(value, field.name))
                        for field in fields(value)}
            if isinstance(value, (tuple, list)):
                return [json_value(item) for item in value]
            return value
        return {"identity_scope": "single_primary_run", "tick_unit": "ticks",
                "windows": json_value(view.windows),
                "requests": json_value(view.requests),
                "services": json_value(view.services)}


@dataclass
class _StepIntegral:
    first_tick: Optional[int] = None
    last_tick: Optional[int] = None
    last_value: int = 0
    area: int = 0

    def observe(self, tick: int, value: int) -> None:
        if self.first_tick is None:
            self.first_tick = tick
            self.last_tick = tick
        if tick < self.last_tick:
            raise ValueError("metric observations must be monotone")
        self.area += self.last_value * (tick - self.last_tick)
        self.last_tick = tick
        self.last_value = value

    @property
    def span_ticks(self) -> int:
        if self.first_tick is None:
            return 0
        return self.last_tick - self.first_tick

    def time_average(self) -> float:
        return self.area / self.span_ticks if self.span_ticks else 0.0


class DecoderUtilization:
    """Time-weighted fraction of decoder units that were busy."""

    name = "decoder_utilization"
    result_schema_version = 1

    def __init__(self, decoder_manager):
        self.decoder_manager = decoder_manager
        self._topology = None
        self._aggregate = _StepIntegral()
        self._per_pool = {}

    def observe(self, engine: "Engine") -> None:
        """Add the busy level held since the last event, then re-sample it."""
        view = utilization_view(self.decoder_manager)
        topology = tuple((name, total) for name, _, total in view.per_pool)
        if self._topology is None:
            self._topology = topology
            self._per_pool = {name: _StepIntegral() for name, _ in topology}
        elif topology != self._topology:
            raise RuntimeError("decoder pool topology changed during measurement")
        self._aggregate.observe(engine.now, view.busy_units)
        for name, busy, _ in view.per_pool:
            self._per_pool[name].observe(engine.now, busy)

    def result(self) -> dict:
        """Aggregate and named per-pool time-weighted busy fractions."""
        topology = self._topology or ()
        totals = dict(topology)
        aggregate_total = sum(totals.values())
        return {
            "observation_span_ticks": self._aggregate.span_ticks,
            "aggregate_busy_fraction": (
                self._aggregate.time_average() / aggregate_total
                if aggregate_total else 0.0
            ),
            "aggregate_total_units": aggregate_total,
            "per_pool_busy_fraction": {
                name: (self._per_pool[name].time_average() / total)
                for name, total in topology
            },
            "per_pool_total_units": totals,
        }


class ReadyQueueStats:
    """Peak and time-average number of decode jobs waiting."""

    name = "ready_queue"
    result_schema_version = 1

    def __init__(self, decoder_manager):
        self.decoder_manager = decoder_manager
        self._integral = _StepIntegral()
        self.peak = 0

    def observe(self, engine: "Engine") -> None:
        """Accumulate time-weighted queue length and track the peak."""
        view = backlog_view(None, self.decoder_manager, include_rounds=False)
        self._integral.observe(engine.now, view.ready_jobs)
        self.peak = max(self.peak, view.ready_jobs)

    def result(self) -> dict:
        """Peak and time-average ready-queue length."""
        return {"observation_span_ticks": self._integral.span_ticks,
                "peak_jobs": self.peak,
                "time_avg_jobs": self._integral.time_average()}


def _time_average(integrals_by_pool: dict, pool: str) -> float:
    """One pool's time average, zero before it was ever observed."""
    integral = integrals_by_pool.get(pool)
    return 0.0 if integral is None else integral.time_average()


def _request_key_json(request_key: Optional[DecoderRequestKey]):
    """One decoder request key as plain JSON-ready values."""
    if request_key is None:
        return None
    return {"operation_id": stable_identity_json(request_key.operation_id),
            "window_id": request_key.window_id,
            "tier": request_key.tier.value,
            "run_sequence": request_key.run_sequence}


class DecoderMemoryOccupancy:
    """Rounds held in each decoder unit's input memory: current, peak and
    time-average occupancy per unit, and the fraction of capacity when the
    unit's memory is finite. Sampled at event boundaries; peaks are the units'
    own exact high-water marks."""

    name = "decoder_memory_occupancy"
    result_schema_version = 2

    def __init__(self, decoder_manager):
        self.decoder_manager = decoder_manager
        self._occupied = {}

    def observe(self, engine: "Engine") -> None:
        for row in decoder_memory_view(self.decoder_manager).per_unit:
            key = f"{row.pool}#{row.unit}"
            self._occupied.setdefault(key, _StepIntegral()).observe(engine.now, row.occupied_rounds)

    def rows(self) -> list:
        return [{"unit": f"{row.pool}#{row.unit}", "capacity_rounds": row.capacity_rounds,
                 "occupied_rounds": row.occupied_rounds,
                 "peak_occupied_rounds": row.peak_occupied_rounds,
                 "admissions": row.admissions}
                for row in decoder_memory_view(self.decoder_manager).per_unit]

    def result(self) -> dict:
        per_unit = {}
        for row in decoder_memory_view(self.decoder_manager).per_unit:
            key = f"{row.pool}#{row.unit}"
            integral = self._occupied.get(key)
            average = integral.time_average() if integral is not None else 0.0
            per_unit[key] = {
                "capacity_rounds": row.capacity_rounds,
                "occupied_rounds": row.occupied_rounds,
                "peak_occupied_rounds": row.peak_occupied_rounds,
                "time_avg_occupied_rounds": average,
                "time_avg_occupied_fraction": (
                    None if row.capacity_rounds is None else average / row.capacity_rounds),
                "admissions": row.admissions,
            }
        return {"per_unit": per_unit}


class WindowLatencyBreakdown:
    """Per-window buffer, dependency, queue, and service latency."""

    name = "window_latency"
    result_schema_version = 1

    def __init__(self, window_manager):
        self.window_manager = window_manager

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; the cluster stamps the windows)."""
        return None

    def rows(self) -> list:
        """One record per fully-decoded window: op, window index, and the four stages."""
        view = window_latency_view(self.window_manager)
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
    result_schema_version = 1

    def __init__(self, window_manager, decoder_manager):
        self.window_manager = window_manager
        self.decoder_manager = decoder_manager
        self._integral = _StepIntegral()
        self.peak = 0
        self.trace = []

    def observe(self, engine: "Engine") -> None:
        """Sample the backlog and update peak, average, and trace."""
        view = backlog_view(self.window_manager, self.decoder_manager)
        self._integral.observe(engine.now, view.total_rounds)
        self.peak = max(self.peak, view.total_rounds)
        if not self.trace or self.trace[-1][1] != view.total_rounds:
            self.trace.append((engine.now, view.total_rounds))

    def rows(self) -> list:
        """Backlog time series, one record per value change."""
        return [{"t": time_ticks, "backlog_rounds": backlog_rounds}
                for time_ticks, backlog_rounds in self.trace]

    def result(self) -> dict:
        """Peak and time-average backlog, in rounds waiting to be decoded."""
        return {"peak_rounds": self.peak,
                "time_avg_rounds": self._integral.time_average()}


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

    The first observation owns the bin epoch, so late registration never
    fabricates backlog before the metric existed.
    """

    name = "backlog_early_warning"
    result_schema_version = 1

    def __init__(self, window_manager, decoder_manager,
                 round_ticks: int, window_ticks: int,
                 threshold_f: float = 0.1, consecutive: int = 2):
        round_ticks = int(round_ticks)
        window_ticks = int(window_ticks)
        threshold_f = float(threshold_f)
        consecutive = int(consecutive)
        if round_ticks <= 0:
            raise ValueError("round_ticks must be positive")
        if window_ticks <= 0:
            raise ValueError("window_ticks must be positive")
        if not math.isfinite(threshold_f) or threshold_f < 0:
            raise ValueError("threshold_f must be finite and nonnegative")
        if consecutive <= 0:
            raise ValueError("consecutive must be positive")
        self.window_manager = window_manager
        self.decoder_manager = decoder_manager
        self.round_ticks = round_ticks
        self.window_ticks = window_ticks
        self.threshold_f = threshold_f
        self.consecutive = consecutive
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
        view = backlog_view(self.window_manager, self.decoder_manager)
        if self._last_view is None:
            self._bin_start = engine.now
            self._start_backlog = view.total_rounds
            self._start_per_patch = dict(view.per_patch_rounds)
            self._last_view = view
            return
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
                    (p for p, dv in deltas.items() if dv == worst and dv > 0),
                    key=stable_identity_order_key,
                ))
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


class StrongDecoderBacklog:
    """Exact global strong input assigned but not yet completed."""

    name = "strong_backlog"
    result_schema_version = 1

    _phase_names = (
        "waiting_far_boundary", "waiting_terminal_data",
        "in_transit", "queued", "running",
    )

    def __init__(self, window_manager, decoder_manager):
        self.window_manager = window_manager
        self.decoder_manager = decoder_manager
        self.peak_jobs = 0
        self.peak_full_input_rounds = 0
        self._jobs = _StepIntegral()
        self._rounds = _StepIntegral()
        self.trace = []

    def observe(self, engine: "Engine") -> None:
        """Sample outstanding strong work and update peak, average, and trace."""
        view = strong_work_view(self.window_manager, self.decoder_manager)
        self._jobs.observe(engine.now, view.total_jobs)
        self._rounds.observe(engine.now, view.total_full_input_rounds)
        self.peak_jobs = max(self.peak_jobs, view.total_jobs)
        self.peak_full_input_rounds = max(
            self.peak_full_input_rounds, view.total_full_input_rounds
        )
        row = {"t_ticks": engine.now}
        for phase_name in self._phase_names:
            phase = getattr(view, phase_name)
            row[f"{phase_name}_jobs"] = phase.jobs
            row[f"{phase_name}_full_input_rounds"] = phase.full_input_rounds
        row["total_jobs"] = view.total_jobs
        row["total_full_input_rounds"] = view.total_full_input_rounds
        comparable = {key: value for key, value in row.items() if key != "t_ticks"}
        if not self.trace or any(
            self.trace[-1][key] != value for key, value in comparable.items()
        ):
            self.trace.append(row)

    def rows(self) -> list:
        """Return fresh phase rows in ticks, jobs, and full input rounds."""
        return [dict(row) for row in self.trace]

    def result(self) -> dict:
        """Peak and time-average outstanding strong jobs."""
        view = strong_work_view(self.window_manager, self.decoder_manager)
        return {
            "observation_span_ticks": self._jobs.span_ticks,
            "peak_jobs": self.peak_jobs,
            "time_avg_jobs": self._jobs.time_average(),
            "peak_full_input_rounds": self.peak_full_input_rounds,
            "time_avg_full_input_rounds": self._rounds.time_average(),
            "strong_needed": view.strong_needed,
            "trace": self.rows(),
        }


class BacklogTrajectory:
    """Per-feedback-gate reaction wait and backlog in rounds."""

    name = "backlog_trajectory"
    result_schema_version = 1

    def __init__(self, execution_runtime):
        self.execution_runtime = execution_runtime

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; the execution runtime stamps the timestamps)."""
        return None

    def rows(self) -> list:
        """One record per released feedback-blocked gate."""
        view = reaction_view(self.execution_runtime)
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

    name = "conditional_reaction_time"
    result_schema_version = 1

    def __init__(self, execution_runtime, divergence_threshold_rounds: float | None = None,
                 require_all_released: bool = True):
        self.execution_runtime = execution_runtime
        self.divergence_threshold_rounds = divergence_threshold_rounds
        self.require_all_released = require_all_released

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample. The controller records body-done and release times."""
        return None

    def _view(self):
        return reaction_view(self.execution_runtime)

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
    result_schema_version = 1

    def __init__(self, factory):
        self.factory = factory

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (the factory stamps each state's StateTrace)."""
        return None

    def result(self) -> dict:
        """Per-stage {mean, max, n} in ticks across all delivered states."""
        totals = self.factory.latency_aggregate_snapshot()
        return {
            stage: {
                "mean": row["sum"] / row["n"] if row["n"] else 0.0,
                "max": row["max"],
                "n": row["n"],
            }
            for stage, row in totals.items()
        }
