"""The decode queues' depth over time, one sample per change.

One manager per side, so a sample is the jobs waiting over both; the
gate pins the samples as queue_log. Each pool's samples are also kept
apart, for DART-Q's max backlog per decoder instance (2605.09142 lines
1101-1109).

A peak counts a depth only when time passes at it, since a queue's
length is weighed by time: gem5 adds current * (curTick() - last)
(src/base/stats/storage.hh:153-159) and Ciw weighs each state by its
time (ciw/trackers/state_tracker.py:55-102). A job that joins and leaves
in one tick never waited.
"""

import collections


class QueueDepthLog:
    """(tick, jobs waiting over both managers' pools) at every change."""

    def __init__(self) -> None:
        self.samples: list = []
        self.depth_by_pool: dict = {}
        self.samples_by_pool: dict = collections.defaultdict(list)

    def depth_changed(self, pool: str, tick: int, depth: int) -> None:
        """One pool's depth now; the sample is the sum over pools."""
        self.depth_by_pool[pool] = depth
        depths = self.depth_by_pool.values()
        total = sum(depths)
        self.samples.append((tick, total))
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
