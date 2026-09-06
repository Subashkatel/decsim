"""The sampled metrics: listeners on the engine's action_done source.

Each samples one view of the run (observe/run_views.py) after every
action and integrates it as a step function of time, so a peak and a
time average come out exact. They are built only when the observation
section asks; the D7 harness reads DecodeBacklog, the unit-count and
memory sweeps read DecoderUtilization and DecoderMemoryOccupancy.
"""

from typing import Optional

import decsim.observe.run_views as run_views


class DecoderUtilization:
    """Time-weighted fraction of decoder units that were busy."""

    def __init__(self, decoder_manager) -> None:
        self.decoder_manager = decoder_manager
        self._topology: Optional[tuple] = None
        self._aggregate = _StepIntegral()
        self._per_pool: dict = {}

    def observe(self, tick: int) -> None:
        """Add the busy level held since the last action, then re-sample."""
        view = run_views.utilization_view(self.decoder_manager)
        topology = []
        for name, _busy, total in view.per_pool:
            topology.append((name, total))
        topology = tuple(topology)
        if self._topology is None:
            self._topology = topology
            for name, _total in topology:
                self._per_pool[name] = _StepIntegral()
        elif topology != self._topology:
            raise RuntimeError(
                "decoder pool topology changed during measurement"
            )
        self._aggregate.observe(tick, view.busy_units)
        for name, busy, _total in view.per_pool:
            self._per_pool[name].observe(tick, busy)

    def result(self) -> dict:
        """Aggregate and named per-pool time-weighted busy fractions."""
        topology = self._topology or ()
        totals = dict(topology)
        total_values = totals.values()
        aggregate_total = sum(total_values)
        aggregate_fraction = 0.0
        if aggregate_total:
            aggregate_average = self._aggregate.time_average()
            aggregate_fraction = aggregate_average / aggregate_total
        per_pool_fraction = {}
        for name, total in topology:
            average = self._per_pool[name].time_average()
            per_pool_fraction[name] = average / total
        return {
            "observation_span_ticks": self._aggregate.span_ticks,
            "aggregate_busy_fraction": aggregate_fraction,
            "aggregate_total_units": aggregate_total,
            "per_pool_busy_fraction": per_pool_fraction,
            "per_pool_total_units": totals,
        }


class DecoderMemoryOccupancy:
    """Rounds held in each unit's input memory: current, peak, time average.

    The fraction of capacity is reported when the unit's memory is
    finite. Sampled after every action; the peaks are the units' own
    exact high-water marks.
    """

    def __init__(self, decoder_manager) -> None:
        self.decoder_manager = decoder_manager
        self._occupied: dict = {}

    def observe(self, tick: int) -> None:
        """Re-sample every unit's occupancy."""
        view = run_views.decoder_memory_view(self.decoder_manager)
        for row in view.per_unit:
            key = f"{row.pool}#{row.unit}"
            integral = self._occupied.get(key)
            if integral is None:
                integral = _StepIntegral()
                self._occupied[key] = integral
            integral.observe(tick, row.occupied_rounds)

    def rows(self) -> list:
        """One record per unit with its current and peak occupancy."""
        view = run_views.decoder_memory_view(self.decoder_manager)
        rows = []
        for row in view.per_unit:
            rows.append(
                {
                    "unit": f"{row.pool}#{row.unit}",
                    "capacity_rounds": row.capacity_rounds,
                    "occupied_rounds": row.occupied_rounds,
                    "peak_occupied_rounds": row.peak_occupied_rounds,
                    "admissions": row.admissions,
                }
            )
        return rows

    def result(self) -> dict:
        """Per unit: capacity, occupancy now, peak, time average, fraction."""
        view = run_views.decoder_memory_view(self.decoder_manager)
        per_unit = {}
        for row in view.per_unit:
            key = f"{row.pool}#{row.unit}"
            integral = self._occupied.get(key)
            average = 0.0
            if integral is not None:
                average = integral.time_average()
            fraction = None
            if row.capacity_rounds is not None:
                fraction = average / row.capacity_rounds
            per_unit[key] = {
                "capacity_rounds": row.capacity_rounds,
                "occupied_rounds": row.occupied_rounds,
                "peak_occupied_rounds": row.peak_occupied_rounds,
                "time_avg_occupied_rounds": average,
                "time_avg_occupied_fraction": fraction,
                "admissions": row.admissions,
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
