"""Statistics declared on a part's class, kept per part path, read as rows.

A part class declares what it measures, as a gem5 SimObject declares
its stats with ADD_STAT and an SST element with
SST_ELI_DOCUMENT_STATISTICS (name, description, unit, enable level):

    class DecoderPool:
        STATISTICS = (
            registry.Statistic("busy_units", kinds.Level, "units decoding",
                               "units"),
        )
        statistics = registry.unrecorded(STATISTICS)

A part's class holds an unrecorded group (every statistic the null
one), so a part built alone, as a test builds it, records nothing. When
the machine is built, the part that holds a component attaches it to
the run's Registry at its path (the path its settings have,
weak_decoder.queue), as gem5 binds each SimObject's stats into the
hierarchy after construction (src/python/m5/stats/__init__.py:277-309).
The component then gets an accumulator for every statistic the run
enables and the shared null statistic for the rest (SST's load level:
a statistic is on when its level is at most the run's), and updates it
where the event happens; nothing else is wired.

At the end of the shot the registry's rows are one long table,
(path, statistic, field, value), as SST's CSV output writes one row per
component, statistic and field (statoutputcsv.cc:77-102): a new part, a
second pool or a new link adds rows, never columns.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import decsim.statistics.kinds as kinds

# the level every declared statistic has unless it says otherwise, and
# the level a run records at unless it asks for another
BASIC = 1
DETAILED = 2
OFF = 0


@dataclasses.dataclass(frozen=True)
class Statistic:
    """One statistic a part class declares.

    kind is a class of kinds.py; unit names the unit its values are in
    (ticks, rounds, jobs, units, bits); level is the least run level
    that records it.
    """

    name: str
    kind: type
    description: str
    unit: str
    level: int = BASIC


@dataclasses.dataclass(frozen=True)
class Row:
    """One field of one statistic of one part."""

    path: str
    statistic: str
    field: str
    value: Any


@dataclasses.dataclass(frozen=True)
class ValueCount:
    """How often one value, or how many ticks one level, was seen."""

    path: str
    statistic: str
    value: int
    count: int


class Group:
    """One part's statistics, each an attribute by its declared name.

    gem5 makes a stat a member of its SimObject's stats group
    (src/base/stats/group.hh:75); here a group is built from the class's
    declarations, so `group.busy_units.set(busy)` updates one.
    """

    def __init__(
        self,
        path: str,
        statistics: Sequence[Statistic],
        level: int,
        clock: Any,
    ) -> None:
        self.path = path
        self.declared = tuple(statistics)
        for statistic in self.declared:
            accumulator = _accumulator(statistic, level, clock)
            setattr(self, statistic.name, accumulator)

    def accumulators(self) -> list:
        """(statistic, accumulator) for each declared statistic, in order."""
        return [
            (statistic, getattr(self, statistic.name))
            for statistic in self.declared
        ]


class Registry:
    """A run's statistics: one group per part path, at one level.

    clock is the run's engine, read for the tick a level changes at;
    level is the most detailed level recorded; OFF records nothing, and
    every group's statistics are then the null one.
    """

    def __init__(self, clock: Any, level: int = BASIC) -> None:
        self.clock = clock
        self.level = level
        self.groups: dict = {}

    def attach(self, component: Any, path: str) -> Group:
        """The component's declared statistics, recorded at this path."""
        group = self.group(path, component.STATISTICS)
        component.statistics = group
        return group

    def group(self, path: str, statistics: Sequence[Statistic]) -> Group:
        """The part at this path's statistics; a path is one part's only.

        gem5 refuses two stat groups of one name in one parent
        (src/base/stats/group.cc:117-125).
        """
        if path in self.groups:
            raise ValueError(f"two parts keep statistics at the path {path}")
        group = Group(path, statistics, self.level, self.clock)
        self.groups[path] = group
        return group

    def finish(self) -> None:
        """Hold every level to now, the shot's last tick."""
        for group in self.groups.values():
            for _, accumulator in group.accumulators():
                accumulator.finish()

    def rows(self) -> list:
        """Every recorded field, by path, then in declaration order."""
        found = []
        for path in sorted(self.groups):
            group = self.groups[path]
            group_rows = _rows_of(group)
            found.extend(group_rows)
        return found

    def value_counts(self) -> list:
        """Every recorded value's count, by path, statistic, then value."""
        found = []
        for path in sorted(self.groups):
            group = self.groups[path]
            group_counts = _value_counts_of(group)
            found.extend(group_counts)
        return found


def unrecorded(statistics: Sequence[Statistic]) -> Group:
    """A group whose every statistic is the null one: a class's default."""
    return Group("", statistics, OFF, None)


def _accumulator(statistic: Statistic, level: int, clock: Any) -> Any:
    if statistic.level > level:
        return kinds.DISABLED
    return statistic.kind(clock)


def _rows_of(group: Group) -> list:
    rows = []
    for statistic, accumulator in group.accumulators():
        for field, value in accumulator.fields():
            row = Row(group.path, statistic.name, field, value)
            rows.append(row)
    return rows


def _value_counts_of(group: Group) -> list:
    counts = []
    for statistic, accumulator in group.accumulators():
        by_value = accumulator.value_counts()
        for value in sorted(by_value):
            count = ValueCount(
                group.path, statistic.name, value, by_value[value]
            )
            counts.append(count)
    return counts
