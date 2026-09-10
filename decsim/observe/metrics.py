"""The integrated metrics: step functions of time over one run.

Each holds one quantity as a step function and integrates it, so a peak
and a time average come out exact. DecoderUtilization and
DecoderMemoryOccupancy step where the quantity changes, on the decoder
pool's and the unit memories' own sources; DecodeBacklog is the one that
still samples, after every action, because the rounds waiting to be
decoded are spread over the window manager and the queues. They are
built only when the observation section asks; the D7 harness reads
DecodeBacklog, the unit-count and memory sweeps read the other two.
"""

from collections.abc import Mapping
from typing import Optional

import decsim.observe.run_views as run_views


class DecoderUtilization:
    """Time-weighted fraction of decoder units whose compute was busy.

    A listener on the pool's unit_busy and unit_freed sources: a unit is
    busy from the tick its compute leaves the pool's free list until the
    tick it goes back, so the integral steps exactly where the occupancy
    changes and no state is sampled. thrust 2 sweeps the unit count over
    this number.
    """

    def __init__(self, engine, units_by_pool: Mapping[str, int]) -> None:
        self.engine = engine
        self.total_by_pool = dict(units_by_pool)
        self.busy_by_pool = dict.fromkeys(self.total_by_pool, 0)
        self._aggregate = _StepIntegral()
        self._per_pool: dict = {}
        for pool in self.total_by_pool:
            self._per_pool[pool] = _StepIntegral()
        self._observe(0)

    def unit_busy(self, unit) -> None:
        """One unit's compute left its pool's free list."""
        self.busy_by_pool[unit.pool] += 1
        self._observe(self.engine.now)

    def unit_freed(self, unit) -> None:
        """One unit's compute went back to its pool's free list."""
        self.busy_by_pool[unit.pool] -= 1
        self._observe(self.engine.now)

    def result(self) -> dict:
        """Aggregate and named per-pool time-weighted busy fractions."""
        self._observe(self.engine.now)
        totals = dict(self.total_by_pool)
        total_values = totals.values()
        aggregate_total = sum(total_values)
        aggregate_fraction = 0.0
        if aggregate_total:
            aggregate_average = self._aggregate.time_average()
            aggregate_fraction = aggregate_average / aggregate_total
        per_pool_fraction = {}
        for pool, total in totals.items():
            average = self._per_pool[pool].time_average()
            per_pool_fraction[pool] = average / total
        return {
            "observation_span_ticks": self._aggregate.span_ticks,
            "busy_unit_ticks": self._aggregate.area,
            "aggregate_busy_fraction": aggregate_fraction,
            "aggregate_total_units": aggregate_total,
            "per_pool_busy_fraction": per_pool_fraction,
            "per_pool_total_units": totals,
        }

    def _observe(self, tick: int) -> None:
        """Hold every integral at the busy counts this tick leaves."""
        busy_counts = self.busy_by_pool.values()
        total_busy = sum(busy_counts)
        self._aggregate.observe(tick, total_busy)
        for pool, busy in self.busy_by_pool.items():
            self._per_pool[pool].observe(tick, busy)


class DecoderMemoryOccupancy:
    """Rounds held in each unit's input memory: now, peak, time average.

    A listener on every unit memory's deposited and taken sources: the
    held-round count steps at the deposit and at the take, so the peak
    and the integral are exact and nothing is sampled. The fraction of
    capacity is reported when the unit's memory is finite; thrust 2
    sweeps the unit memory over these numbers.
    """

    def __init__(
        self, engine, capacity_by_unit: Mapping[str, Optional[int]]
    ) -> None:
        self.engine = engine
        self.by_unit: dict = {}
        for unit_name, capacity_rounds in capacity_by_unit.items():
            self.by_unit[unit_name] = _UnitMemoryOccupancy(capacity_rounds)

    def deposited(self, unit_name: str, _job, decoder_input) -> None:
        """One job's rounds landed in that unit's memory."""
        occupancy = self.by_unit[unit_name]
        rounds = len(decoder_input.rounds)
        occupancy.deposit(self.engine.now, rounds)

    def taken(self, unit_name: str, _job, decoder_input) -> None:
        """One job's rounds were freed from that unit's memory."""
        occupancy = self.by_unit[unit_name]
        rounds = len(decoder_input.rounds)
        occupancy.take(self.engine.now, rounds)

    def rows(self) -> list:
        """One record per unit with its current and peak occupancy."""
        rows = []
        for unit_name, occupancy in self.by_unit.items():
            rows.append(
                {
                    "unit": unit_name,
                    "capacity_rounds": occupancy.capacity_rounds,
                    "occupied_rounds": occupancy.held_rounds,
                    "peak_occupied_rounds": occupancy.peak_rounds,
                    "admissions": occupancy.admissions,
                }
            )
        return rows

    def result(self) -> dict:
        """Per unit: capacity, occupancy now, peak, time average, fraction."""
        per_unit = {}
        for unit_name, occupancy in self.by_unit.items():
            average = occupancy.integral.time_average()
            fraction = None
            if occupancy.capacity_rounds is not None:
                fraction = average / occupancy.capacity_rounds
            per_unit[unit_name] = {
                "capacity_rounds": occupancy.capacity_rounds,
                "occupied_rounds": occupancy.held_rounds,
                "peak_occupied_rounds": occupancy.peak_rounds,
                "time_avg_occupied_rounds": average,
                "time_avg_occupied_fraction": fraction,
                "admissions": occupancy.admissions,
            }
        return {"per_unit": per_unit}


class DecodeBacklog:
    """Rounds of syndrome data produced but not yet decoded.

    Sampled once when built, before the first action, and after every
    action; the trace keeps one row per change of value.
    """

    def __init__(self, window_manager, decoder_manager) -> None:
        self.window_manager = window_manager
        self.decoder_manager = decoder_manager
        self._integral = _StepIntegral()
        self.peak = 0
        self.trace: list = []
        self.observe(0)

    def observe(self, tick: int) -> None:
        """Sample the backlog and update the peak, the average, the trace."""
        view = run_views.backlog_view(self.window_manager, self.decoder_manager)
        self._integral.observe(tick, view.total_rounds)
        self.peak = max(self.peak, view.total_rounds)
        is_first = not self.trace
        if is_first or self.trace[-1][1] != view.total_rounds:
            self.trace.append((tick, view.total_rounds))

    def rows(self) -> list:
        """Backlog time series, one record per value change."""
        rows = []
        for time_ticks, backlog_rounds in self.trace:
            rows.append({"t": time_ticks, "backlog_rounds": backlog_rounds})
        return rows

    def result(self) -> dict:
        """Peak and time-average backlog, in rounds waiting to be decoded."""
        time_average = self._integral.time_average()
        return {"peak_rounds": self.peak, "time_avg_rounds": time_average}


class _UnitMemoryOccupancy:
    """One unit memory's held rounds as its own events describe them."""

    def __init__(self, capacity_rounds: Optional[int]) -> None:
        self.capacity_rounds = capacity_rounds
        self.held_rounds = 0
        self.peak_rounds = 0
        self.admissions = 0
        self.integral = _StepIntegral()

    def deposit(self, tick: int, rounds: int) -> None:
        """Rounds landed here; the count rises from this tick."""
        self.held_rounds += rounds
        self.peak_rounds = max(self.peak_rounds, self.held_rounds)
        self.admissions += 1
        self.integral.observe(tick, self.held_rounds)

    def take(self, tick: int, rounds: int) -> None:
        """Rounds were freed; the count falls from this tick."""
        self.held_rounds -= rounds
        self.integral.observe(tick, self.held_rounds)


class _StepIntegral:
    """The integral of a step function sampled at every change."""

    def __init__(self) -> None:
        self.first_tick: Optional[int] = None
        self.last_tick: Optional[int] = None
        self.last_value = 0
        self.area = 0

    def observe(self, tick: int, value: int) -> None:
        """The value holds from this tick until the next observation."""
        if self.first_tick is None:
            self.first_tick = tick
            self.last_tick = tick
        assert tick >= self.last_tick, "metric observations run backwards"
        self.area += self.last_value * (tick - self.last_tick)
        self.last_tick = tick
        self.last_value = value

    @property
    def span_ticks(self) -> int:
        """The ticks between the first and the last observation."""
        if self.first_tick is None:
            return 0
        return self.last_tick - self.first_tick

    def time_average(self) -> float:
        """The area over the span; zero before the first observation."""
        if not self.span_ticks:
            return 0.0
        return self.area / self.span_ticks
