"""The region of interest restarts the statistics after a warm-up.

The referent is gem5's: a statistic after m5.stats.reset() holds what
happened since (src/python/m5/stats/__init__.py:439-450), and a level's
average is over the ticks since lastReset (src/base/stats/storage.hh,
AvgStor). So the run with a warm-up must give, statistic by statistic,
the whole run's value less what the whole run had kept at the tick the
warm-up ended, read off a copy of its registry taken at that readout.
"""

import collections
import copy

import pytest

import decsim.machine as machine_module
import decsim.observe.metrics as metrics
import decsim.statistics.kinds as kinds
import tests.observe.gate_point as gate_point

WARMUP_ROUNDS = 2
# more rounds than the gate point reads out
ENDLESS_WARMUP_ROUNDS = 1_000_000


class _Snapshot:
    """A copy of the registry's statistics at the warm-up's last readout."""

    def __init__(self, registry) -> None:
        self.registry = registry
        self.tick = None
        self.kept: dict = {}

    def round_read_out(self, readout) -> None:
        if self.tick is not None or readout.round_index <= WARMUP_ROUNDS:
            return
        self.tick = self.registry.clock.now
        for path, group in self.registry.groups.items():
            for statistic, accumulator in group.accumulators():
                key = (path, statistic.name)
                self.kept[key] = _copied(accumulator)


def _copied(accumulator):
    """The accumulator's numbers now, apart from its later updates."""
    copied = copy.copy(accumulator)
    counts = getattr(accumulator, "counts", None)
    if counts is not None:
        copied.counts = collections.Counter(counts)
    return copied


def _whole_run_and_snapshot():
    point = gate_point.settings()
    machine = machine_module.Machine.build(point, gate_point.SEED)
    snapshot = _Snapshot(machine.observation.statistics)
    machine.qpu.device.trace.round_emitted.connect(snapshot.round_read_out)
    machine.run()
    return machine, snapshot


def _count_since(whole, before, _snapshot) -> tuple:
    return (whole.count - before.count,)


def _values_since(whole, before, _snapshot) -> tuple:
    counts = collections.Counter(whole.counts)
    counts.subtract(before.counts)
    kept = +counts
    return (whole.total - before.total, dict(kept))


def _level_since(whole, before, snapshot) -> tuple:
    held_since_change = before.level * (snapshot.tick - before.tick)
    area_before = before.area + held_since_change
    span = whole.span - snapshot.tick
    return (whole.area - area_before, span)


def _count_now(statistic) -> tuple:
    return (statistic.count,)


def _values_now(statistic) -> tuple:
    return (statistic.total, dict(statistic.counts))


def _level_now(statistic) -> tuple:
    return (statistic.area, statistic.span)


SINCE = {
    kinds.Count: (_count_since, _count_now),
    kinds.Distribution: (_values_since, _values_now),
    kinds.Level: (_level_since, _level_now),
    kinds.History: (_level_since, _level_now),
}


def test_a_warm_up_leaves_each_statistic_what_happened_after_it():
    whole_machine, snapshot = _whole_run_and_snapshot()
    warm_machine, _result = gate_point.run(warmup_rounds=WARMUP_ROUNDS)
    whole_groups = whole_machine.observation.statistics.groups
    warm_registry = warm_machine.observation.statistics

    compared = 0
    for path, group in warm_registry.groups.items():
        whole_group = whole_groups[path]
        compared += _compare_group(path, group, whole_group, snapshot)

    assert warm_registry.reset_tick == snapshot.tick
    assert compared > 5


def _compare_group(path, group, whole_group, snapshot) -> int:
    """Each recorded statistic against the whole run's since the reset."""
    compared = 0
    for statistic, warm in group.accumulators():
        if warm is kinds.DISABLED:
            continue
        whole = getattr(whole_group, statistic.name)
        before = snapshot.kept[(path, statistic.name)]
        since, now = SINCE[type(warm)]
        if statistic.is_configuration:
            assert now(warm) == now(whole)
            continue
        assert now(warm) == since(whole, before, snapshot), statistic
        compared += 1
    return compared


def test_a_warm_up_leaves_the_backlog_peak_of_the_ticks_after_it(
    monkeypatch,
):
    """The peak over the counts ticks end at, from the reset's tick on."""
    ended_at = {}
    observe = metrics.DecodeBacklog.observe

    def recording(backlog, tick) -> None:
        observe(backlog, tick)
        ended_at[tick] = backlog.rounds

    monkeypatch.setattr(metrics.DecodeBacklog, "observe", recording)
    gate_point.run(backlog_trace=True)
    whole_ended_at = dict(ended_at)
    warm_machine, _result = gate_point.run(
        backlog_trace=True, warmup_rounds=WARMUP_ROUNDS
    )
    reset_tick = warm_machine.observation.statistics.reset_tick

    after = [
        rounds for tick, rounds in whole_ended_at.items() if tick >= reset_tick
    ]
    assert warm_machine.observation.decode_backlog.peak == max(after)


def test_a_warm_up_moves_no_number_but_the_statistics():
    plain_machine, plain_result = gate_point.run()
    warm_machine, warm_result = gate_point.run(warmup_rounds=WARMUP_ROUNDS)

    assert warm_result.operation_results == plain_result.operation_results
    assert warm_machine.engine.now == plain_machine.engine.now


def test_a_run_that_ends_inside_its_warm_up_is_refused():
    with pytest.raises(ValueError, match="no region of interest began"):
        gate_point.run(warmup_rounds=ENDLESS_WARMUP_ROUNDS)
