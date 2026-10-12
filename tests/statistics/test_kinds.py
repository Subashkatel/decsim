"""The kinds of statistic (statistics/kinds.py).

The referents are computed inside the tests from the steps alone: a
Level's area is the sum of each step's level times the ticks to the
next change, from tick 0 at 0 (gem5's AvgStor), its peak is the most of
the levels a tick ends at, and a Distribution's counts give the values
back exactly.
"""

import itertools
import random
import types

import decsim.statistics.kinds as kinds


def _random_steps(seed: int) -> list:
    """(tick, level) changes in tick order, several in some ticks."""
    generator = random.Random(seed)
    steps = []
    tick = 0
    for _ in range(200):
        tick += generator.choice([0, 0, 1, 3, 10])
        level = generator.randrange(6)
        steps.append((tick, level))
    return steps


def _stepped(kind, steps, end_tick):
    """The statistic set at each step and finished at end_tick."""
    clock = types.SimpleNamespace(now=0)
    statistic = kind(clock)
    for tick, level in steps:
        clock.now = tick
        statistic.set(level)
    clock.now = end_tick
    statistic.finish()
    return statistic


def _area(steps, end_tick):
    """Each level times the ticks it held, from tick 0 at level 0."""
    changes = [(0, 0), *steps, (end_tick, None)]
    area = 0
    for (tick, level), (next_tick, _) in itertools.pairwise(changes):
        area += level * (next_tick - tick)
    return area


def _peak(steps):
    """The most of the levels a tick ends at."""
    last_by_tick = dict(steps)
    levels = last_by_tick.values()
    return max(levels)


def test_a_levels_area_is_each_level_times_the_ticks_it_held():
    for seed in range(5):
        steps = _random_steps(seed)
        last_tick = steps[-1][0]
        end_tick = last_tick + 7

        level = _stepped(kinds.Level, steps, end_tick)

        assert level.area == _area(steps, end_tick)
        assert level.tick == end_tick


def test_a_level_peaks_at_the_most_a_tick_ends_at():
    for seed in range(5):
        steps = _random_steps(seed)
        last_tick = steps[-1][0]

        level = _stepped(kinds.Level, steps, last_tick)

        assert level.peak == _peak(steps)


def test_a_level_set_and_replaced_in_one_tick_never_held():
    level = _stepped(kinds.Level, [(5, 9), (5, 1)], 8)

    assert level.peak == 1
    assert level.value_counts() == {0: 5, 1: 3}
    assert level.area == 3


def test_a_history_keeps_every_change_and_the_fields_of_a_level():
    steps = _random_steps(0)
    last_tick = steps[-1][0]
    end_tick = last_tick + 7

    history = _stepped(kinds.History, steps, end_tick)
    level = _stepped(kinds.Level, steps, end_tick)

    assert history.changes == steps
    assert history.fields() == level.fields()
    assert history.value_counts() == level.value_counts()


def test_a_distribution_gives_its_values_back_exactly():
    distribution = kinds.Distribution(None)
    values = [7, 3, 7, 0, 12]

    for value in values:
        distribution.add(value)

    assert distribution.fields() == [
        ("count", 5),
        ("total", 29),
        ("minimum", 0),
        ("maximum", 12),
    ]
    assert distribution.value_counts() == {7: 2, 3: 1, 0: 1, 12: 1}


def test_an_empty_distribution_has_no_minimum_or_maximum():
    distribution = kinds.Distribution(None)

    assert distribution.fields() == [("count", 0), ("total", 0)]


def test_a_disabled_statistic_keeps_nothing():
    disabled = kinds.DISABLED

    disabled.add(3)
    disabled.set(2)
    disabled.finish()

    assert disabled.fields() == []
    assert disabled.value_counts() == {}


def _level_at(steps, tick):
    """The level held over the tick: the last set at or before it."""
    held = 0
    for step_tick, level in steps:
        if step_tick > tick:
            break
        held = level
    return held


def _reset_at(kind, steps, reset_tick, end_tick):
    """The statistic set at each step, reset after reset_tick's steps."""
    clock = types.SimpleNamespace(now=0)
    statistic = kind(clock)
    is_reset = False
    for tick, level in steps:
        if tick > reset_tick and not is_reset:
            clock.now = reset_tick
            statistic.reset()
            is_reset = True
        clock.now = tick
        statistic.set(level)
    clock.now = end_tick
    statistic.finish()
    return statistic


def test_a_reset_level_keeps_its_level_and_counts_from_the_reset_on():
    """gem5's AvgStor::reset: total to 0 and lastReset to now, level kept.

    The referents are read off the whole run's step function over the
    ticks from the reset to the end.
    """
    for seed in range(10):
        steps = _random_steps(seed)
        end_tick = steps[-1][0] + 7
        generator = random.Random(seed)
        reset_tick = generator.randrange(1, steps[-1][0])
        ticks = range(reset_tick, end_tick)
        held = [_level_at(steps, tick) for tick in ticks]
        ended = [*held, _level_at(steps, end_tick)]

        level = _reset_at(kinds.Level, steps, reset_tick, end_tick)

        assert level.area == sum(held)
        assert level.span == end_tick - reset_tick
        assert level.peak == max(ended)


def test_a_reset_history_starts_its_series_at_the_level_held():
    steps = [(2, 3), (5, 1), (9, 4)]

    history = _reset_at(kinds.History, steps, 6, 12)

    assert history.changes == [(6, 1), (9, 4)]
    assert history.fields() == [("area", 15), ("span", 6), ("peak", 4)]


def test_a_reset_count_and_distribution_keep_nothing_from_before():
    count = kinds.Count(None)
    distribution = kinds.Distribution(None)
    count.add(4)
    distribution.add(9)

    count.reset()
    distribution.reset()
    count.add()
    distribution.add(2)

    assert count.fields() == [("count", 1)]
    assert distribution.fields() == [
        ("count", 1),
        ("total", 2),
        ("minimum", 2),
        ("maximum", 2),
    ]


def test_only_a_history_changes_kind_in_a_run_that_keeps_no_series():
    assert kinds.History.series_free() is kinds.Level
    assert kinds.Level.series_free() is kinds.Level
    assert kinds.Count.series_free() is kinds.Count
    assert kinds.Distribution.series_free() is kinds.Distribution
