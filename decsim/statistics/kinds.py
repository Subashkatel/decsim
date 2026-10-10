"""The kinds of statistic a part keeps, and the null one it keeps when off.

A part updates its statistics where the event happens, as a gem5
SimObject increments its own stats (src/base/statistics.hh); a disabled
statistic is one shared null object whose updates do nothing, as SST
hands out createDisabledStatistic (sst-core src/sst/core/statapi/
statengine.h:84-91). Every kind keeps only fields that add across shots
and pieces, so a run's totals are sums; a mean, a fraction or a tail is
derived when the table is read, as gem5 evaluates a Formula at dump.

    Count         events: count
    Distribution  values: count, total, minimum, maximum, and each
                  value's count, for exact medians and tails
    Level         a level over time from tick 0: area (level x ticks),
                  span, the peak a tick ends at, and the ticks spent at
                  each level, for a time-weighted distribution
    History       a Level that also keeps every (tick, level) it was
                  set to, the time series a gate or a plot reads

Every kind is built on the run's clock, anything with a `now` in ticks
(the engine), as every gem5 stat reads curTick(); only a Level reads it.
"""

import collections
from typing import Any, Optional


class Count:
    """How many times something happened."""

    def __init__(self, clock: Any) -> None:
        del clock
        self.count = 0

    def add(self, amount: int = 1) -> None:
        """Add amount events."""
        self.count += amount

    def finish(self) -> None:
        """Nothing holds over time."""

    def fields(self) -> list:
        """(field, value) pairs, in the order a table writes them."""
        return [("count", self.count)]

    def value_counts(self) -> dict:
        """No values are kept apart."""
        return {}


class Distribution:
    """Values, each counted, with their total, minimum and maximum.

    A value is an integer in the statistic's unit (ticks for a time), so
    totals are exact and a median or a p99 is read from the counts.
    """

    def __init__(self, clock: Any) -> None:
        del clock
        self.counts: collections.Counter = collections.Counter()
        self.total = 0
        self.minimum: Optional[int] = None
        self.maximum: Optional[int] = None

    def add(self, value: int) -> None:
        """One more value."""
        self.counts[value] += 1
        self.total += value
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    def finish(self) -> None:
        """Nothing holds over time."""

    def fields(self) -> list:
        """(field, value) pairs; no minimum or maximum before a value."""
        counts = self.counts.values()
        count = sum(counts)
        pairs = [("count", count), ("total", self.total)]
        if self.minimum is None:
            return pairs
        return [*pairs, ("minimum", self.minimum), ("maximum", self.maximum)]

    def value_counts(self) -> dict:
        """Each value's count."""
        return dict(self.counts)


class Level:
    """A level that holds from one change to the next, from tick 0 at 0.

    A step holds over [tick, next change), so its area is level x ticks
    and the span is the last tick seen, the integral gem5's AvgStor keeps
    (src/base/stats/storage.hh:126-213). The peak counts only the level a
    tick ends at: a level set and replaced in one tick never held, as a
    queue's length is weighed by time (gem5 adds current * (curTick() -
    last), storage.hh:153-159; Ciw weighs each state by its time,
    ciw/trackers/state_tracker.py:55-102).
    """

    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.tick = 0
        self.level = 0
        self.area = 0
        self.peak = 0
        self.ticks_by_level: collections.Counter = collections.Counter()

    def set(self, level: int) -> None:
        """The level from now on."""
        tick = self.clock.now
        held = tick - self.tick
        if held:
            self._hold(held)
        self.tick = tick
        self.level = level

    def finish(self) -> None:
        """Hold the last level to now, the shot's end; the end counts."""
        self.set(self.level)
        self.peak = max(self.peak, self.level)

    def fields(self) -> list:
        """(field, value) pairs: area, span and peak."""
        return [("area", self.area), ("span", self.tick), ("peak", self.peak)]

    def value_counts(self) -> dict:
        """The ticks spent at each level."""
        return dict(self.ticks_by_level)

    def _hold(self, held: int) -> None:
        self.area += self.level * held
        self.ticks_by_level[self.level] += held
        self.peak = max(self.peak, self.level)


class History(Level):
    """A Level that keeps every (tick, level) it is set to, repeats too.

    The fields are a Level's; the series is read whole, as the gate
    reads the jobs waiting at every change.
    """

    def __init__(self, clock: Any) -> None:
        Level.__init__(self, clock)
        self.changes: list = []

    def set(self, level: int) -> None:
        """The level from now on, kept in the series."""
        Level.set(self, level)
        change = (self.tick, level)
        self.changes.append(change)

    def finish(self) -> None:
        """Hold the last level to now; the series is not extended."""
        Level.set(self, self.level)
        self.peak = max(self.peak, self.level)


class Disabled:
    """A statistic that is off: every update does nothing."""

    def add(self, value: int = 1) -> None:
        """Nothing is kept."""

    def set(self, level: int) -> None:
        """Nothing is kept."""

    def finish(self) -> None:
        """Nothing is kept."""

    def fields(self) -> list:
        """No fields."""
        return []

    def value_counts(self) -> dict:
        """No values."""
        return {}


# the one null statistic every disabled statistic is
DISABLED = Disabled()
