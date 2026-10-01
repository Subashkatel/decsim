"""The integrated metrics: step functions of time over one run.

Each holds one quantity as a step function and integrates it, so a peak
and a time average come out exact. A time average runs over the whole
run, from tick 0 to the tick it is read, as gem5's AvgStor integrates
from its last reset to curTick() (src/base/stats/storage.hh:130-213).
DecoderUtilization steps where the quantity changes, on the decoder
pool's own sources; DecodeBacklog keeps only its peak, sampled after
every action. DecoderUtilization is always built, since every run's
pool columns read each tier's busy fraction off it; DecodeBacklog is
built only when the observation section asks for backlog_trace.
"""

from collections.abc import Mapping

import decsim.observe.run_views as run_views


class DecoderUtilization:
    """Time-weighted fraction of decoder units whose compute was busy.

    A listener on the pool's unit_busy and unit_freed sources: a unit is
    busy from the tick its compute leaves the pool's free list until the
    tick it goes back, so the integral steps exactly where the occupancy
    changes and no state is sampled. The pool sweep reads each tier's
    fraction, Triage's utilization rate (2605.04459 lines 1024-1031).
    """

    def __init__(self, engine, units_by_pool: Mapping[str, int]) -> None:
        self.engine = engine
        self.total_by_pool = dict(units_by_pool)
        self.busy_by_pool = dict.fromkeys(self.total_by_pool, 0)
        self._aggregate = _StepIntegral()
        self._per_pool: dict = {}
        for pool in self.total_by_pool:
            self._per_pool[pool] = _StepIntegral()

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


class DecodeBacklog:
    """The most rounds of syndrome data produced and not yet decoded.

    Sampled after every action, since the rounds waiting are spread over
    the window manager and the queues and no one source reports them.
    """

    def __init__(self, window_manager, decoder_managers) -> None:
        self.window_manager = window_manager
        self.decoder_managers = decoder_managers
        self.peak = 0

    def observe(self, tick: int) -> None:
        """Sample the backlog after an action and keep its peak."""
        del tick
        view = run_views.backlog_view(
            self.window_manager, self.decoder_managers
        )
        self.peak = max(self.peak, view.total_rounds)


class _StepIntegral:
    """The integral of a step function from tick 0, sampled at every change.

    The quantity is zero when the run starts, AvgStor's reset (gem5
    src/base/stats/storage.hh:143-146), so the span always begins at
    tick 0; a reader observes at the tick it reads, as AvgStor's
    prepare integrates to curTick() before a dump (:198-203). gem5
    divides by curTick() - lastReset + 1, counting both end ticks; a
    step here holds over [tick, next tick), so the span is the last tick.
    """

    def __init__(self) -> None:
        self.last_tick = 0
        self.last_value = 0
        self.area = 0

    def observe(self, tick: int, value: int) -> None:
        """The value holds from this tick until the next observation."""
        assert tick >= self.last_tick, "metric observations run backwards"
        self.area += self.last_value * (tick - self.last_tick)
        self.last_tick = tick
        self.last_value = value

    @property
    def span_ticks(self) -> int:
        """The ticks from the run's start to the last observation."""
        return self.last_tick

    def time_average(self) -> float:
        """The area over the span; zero at tick 0."""
        if not self.span_ticks:
            return 0.0
        return self.area / self.span_ticks
