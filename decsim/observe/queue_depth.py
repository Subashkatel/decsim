"""The decode queues' depth over time, one sample per change.

A listener on each manager's depth_changed(tick, depth), one manager
per side, so a sample is the jobs waiting over both; the samples are
what the gate pins as queue_log and what the switching study reads as
the ready-queue's peak. It also hears pool_depth_changed(tick, pool,
depth) and keeps each pool's samples apart, for the max backlog per
decoder instance DART-Q reports (2605.09142 lines 1101-1109), which the
pool sweep reads per tier.

A peak counts a depth only when time passes at it. A job that joins and
leaves the queue in one tick is dispatched without waiting, and a
queue's length is weighed by the time spent at each length: gem5 adds
current * (curTick() - last) when an average changes (gem5
src/base/stats/storage.hh:153-159), and Ciw weighs each state by the
time the system spent in it (Ciw ciw/trackers/state_tracker.py:55-102).
A depth held for zero ticks weighs nothing, so the peak is the largest
depth a tick ends at.
"""

import collections


class QueueDepthLog:
    """(tick, jobs waiting over every pool of every manager) at every change."""

    def __init__(self) -> None:
        self.samples: list = []
        self.depth_by_manager: dict = {}
        self.samples_by_pool: dict = collections.defaultdict(list)

    def depth_changed(self, manager, tick: int, depth: int) -> None:
        """One manager's depth now; the sample is the sum over managers."""
        self.depth_by_manager[manager] = depth
        depths = self.depth_by_manager.values()
        total = sum(depths)
        self.samples.append((tick, total))

    def pool_depth_changed(self, tick: int, pool: str, depth: int) -> None:
        """One pool's depth now, kept apart from the total's samples."""
        self.samples_by_pool[pool].append((tick, depth))

    @property
    def peak(self) -> int:
        """The most jobs that waited at once; zero when none ever waited."""
        return _held_peak(self.samples)

    @property
    def peak_by_pool(self) -> dict:
        """Each pool's most jobs that waited at once."""
        peaks = {}
        for pool, samples in self.samples_by_pool.items():
            peaks[pool] = _held_peak(samples)
        return peaks


def _held_peak(samples: list) -> int:
    """The largest depth a tick ends at, the depth time passes at.

    The samples arrive in tick order, so the last sample of a tick
    overwrites the ones before it in the same tick.
    """
    depth_by_tick = {}
    for tick, depth in samples:
        depth_by_tick[tick] = depth
    depths = depth_by_tick.values()
    return max(depths, default=0)
