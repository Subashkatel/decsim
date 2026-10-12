"""Statistics declared on a class, kept per part path (statistics/registry.py).

gem5's rules: a part's stats are members of its own group, two groups
of one name are refused, and the dump walks the groups in path order;
SST's: a statistic above the run's level is the shared null one, and
the output is one row per part, statistic and field.
"""

import types

import pytest

import decsim.statistics.kinds as kinds
import decsim.statistics.registry as registry

POOL_STATISTICS = (
    registry.Statistic("busy_units", kinds.Level, "units decoding", "units"),
    registry.Statistic("service", kinds.Distribution, "decode time", "ticks"),
    registry.Statistic(
        "dispatches",
        kinds.Count,
        "jobs dispatched",
        "jobs",
        level=registry.DETAILED,
    ),
)


def test_a_part_built_alone_records_nothing():
    pool = _Pool()

    pool.statistics.busy_units.set(3)

    assert pool.statistics.busy_units is kinds.DISABLED


def test_an_attached_part_records_at_its_path():
    clock = _clock()
    run = registry.Registry(clock)
    pool = _Pool()

    run.attach(pool, "weak_decoder.units")
    clock.now = 4
    pool.statistics.busy_units.set(2)
    clock.now = 10
    run.finish()

    assert run.groups["weak_decoder.units"] is pool.statistics
    assert pool.statistics.busy_units.area == 12
    assert _Pool.statistics.busy_units is kinds.DISABLED


def test_a_group_holds_each_declared_statistic_by_name():
    clock = _clock()
    run = registry.Registry(clock)

    group = run.group("weak_decoder", POOL_STATISTICS)

    assert isinstance(group.busy_units, kinds.Level)
    assert isinstance(group.service, kinds.Distribution)


def test_a_statistic_above_the_runs_level_is_the_null_one():
    clock = _clock()
    basic = registry.Registry(clock, registry.BASIC)
    detailed = registry.Registry(clock, registry.DETAILED)
    off = registry.Registry(clock, registry.OFF)

    basic_group = basic.group("weak_decoder", POOL_STATISTICS)
    detailed_group = detailed.group("weak_decoder", POOL_STATISTICS)
    off_group = off.group("weak_decoder", POOL_STATISTICS)

    assert basic_group.dispatches is kinds.DISABLED
    assert isinstance(detailed_group.dispatches, kinds.Count)
    assert off_group.busy_units is kinds.DISABLED
    assert off.rows() == []


def test_two_parts_at_one_path_are_refused():
    clock = _clock()
    run = registry.Registry(clock)
    run.group("weak_decoder", POOL_STATISTICS)

    with pytest.raises(ValueError, match="path weak_decoder"):
        run.group("weak_decoder", POOL_STATISTICS)


def test_the_rows_are_one_per_part_statistic_and_field_in_path_order():
    clock = _clock()
    run = registry.Registry(clock, registry.DETAILED)
    strong = run.group("strong_decoder", POOL_STATISTICS)
    weak = run.group("weak_decoder", POOL_STATISTICS)
    weak.service.add(40)
    clock.now = 10
    weak.busy_units.set(1)
    strong.dispatches.add()
    clock.now = 30

    run.finish()

    rows = run.rows()
    paths = [row.path for row in rows]
    assert paths == sorted(paths)
    assert registry.Row("weak_decoder", "busy_units", "area", 20) in rows
    assert registry.Row("weak_decoder", "service", "maximum", 40) in rows
    assert registry.Row("strong_decoder", "dispatches", "count", 1) in rows


def test_the_value_counts_keep_each_value_and_each_levels_ticks():
    clock = _clock()
    run = registry.Registry(clock)
    weak = run.group("weak_decoder", POOL_STATISTICS)
    weak.service.add(40)
    weak.service.add(40)
    clock.now = 10
    weak.busy_units.set(1)
    clock.now = 30

    run.finish()

    assert run.value_counts() == [
        registry.ValueCount("weak_decoder", "busy_units", 0, 10),
        registry.ValueCount("weak_decoder", "busy_units", 1, 20),
        registry.ValueCount("weak_decoder", "service", 40, 2),
    ]


def test_a_reset_restarts_the_counts_keeps_configuration_and_calls_back():
    """gem5's m5.stats.reset(): every stat, then the reset callbacks."""
    clock = _clock()
    run = registry.Registry(clock)
    declared = (
        registry.Statistic(
            "units", kinds.Count, "units", "units", is_configuration=True
        ),
        registry.Statistic("dispatches", kinds.Count, "jobs", "jobs"),
    )
    group = run.group("weak_decoder.units", declared)
    group.units.add(4)
    group.dispatches.add(7)
    heard = []
    run.on_reset(heard.append)

    clock.now = 30
    run.reset()

    assert group.units.count == 4
    assert group.dispatches.count == 0
    assert heard == [30]
    assert run.reset_tick == 30


def _clock():
    return types.SimpleNamespace(now=0)


class _Pool:
    """A part class that declares the pool's statistics."""

    STATISTICS = POOL_STATISTICS
    statistics = registry.unrecorded(POOL_STATISTICS)


def test_a_run_that_keeps_no_series_keeps_a_historys_fields_alone():
    clock = _clock()
    run = registry.Registry(clock, keeps_series=False)
    declared = (
        registry.Statistic("queued", kinds.History, "jobs waiting", "jobs"),
    )

    group = run.group("decoder_manager", declared)
    group.queued.set(2)
    group.queued.finish()

    assert not hasattr(group.queued, "changes")
    assert group.queued.peak == 2
