"""The decode queue's depth over time, one sample per change.

A listener on the queue's depth_changed(tick, depth); the samples are
what the gate pins as queue_log and what the switching study reads as
the ready-queue's peak.
"""


class QueueDepthLog:
    """(tick, jobs waiting over every pool) at every change."""

    def __init__(self) -> None:
        self.samples: list = []

    def depth_changed(self, tick: int, depth: int) -> None:
        """One more sample."""
        self.samples.append((tick, depth))

    @property
    def peak(self) -> int:
        """The most jobs waiting at once; zero when nothing ever waited."""
        depths = []
        for _tick, depth in self.samples:
            depths.append(depth)
        return max(depths, default=0)
