"""The decode queue's depth over time, one sample per change.

A listener on the queue's depth_changed(tick, depth); the samples are
what the gate pins as queue_log and what the switching study reads as
the ready-queue's peak. It also hears pool_depth_changed(tick, pool,
depth) and keeps each pool's peak alone, the max backlog per decoder
instance DART-Q reports (2605.09142 lines 1101-1109), which the pool
sweep reads per tier.
"""


class QueueDepthLog:
    """(tick, jobs waiting over every pool) at every change."""

    def __init__(self) -> None:
        self.samples: list = []
        self.peak_by_pool: dict = {}

    def depth_changed(self, tick: int, depth: int) -> None:
        """One more sample."""
        self.samples.append((tick, depth))

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
