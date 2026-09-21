"""The decode queues' depth over time, one sample per change.

A listener on each manager's depth_changed(tick, depth), one manager
per side, so a sample is the jobs waiting over both; the samples are
what the gate pins as queue_log and what the switching study reads as
the ready-queue's peak. It also hears pool_depth_changed(tick, pool,
depth) and keeps each pool's peak alone, the max backlog per decoder
instance DART-Q reports (2605.09142 lines 1101-1109), which the pool
sweep reads per tier.
"""


class QueueDepthLog:
    """(tick, jobs waiting over every pool of every manager) at every change."""

    def __init__(self) -> None:
        self.samples: list = []
        self.depth_by_manager: dict = {}
        self.peak_by_pool: dict = {}

    def depth_changed(self, manager, tick: int, depth: int) -> None:
        """One manager's depth now; the sample is the sum over managers."""
        self.depth_by_manager[manager] = depth
        depths = self.depth_by_manager.values()
        total = sum(depths)
        self.samples.append((tick, total))

    def pool_depth_changed(self, tick: int, pool: str, depth: int) -> None:
        """One pool's depth now; only its peak is kept."""
        del tick
        peak = self.peak_by_pool.get(pool, 0)
        self.peak_by_pool[pool] = max(peak, depth)

    @property
    def peak(self) -> int:
        """The most jobs waiting at once; zero when nothing ever waited."""
        depths = []
        for _tick, depth in self.samples:
            depths.append(depth)
        return max(depths, default=0)
