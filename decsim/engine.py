"""Small discrete-event simulation engine."""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import fmt


@dataclass(order=True)
class Event:
    """One scheduled action in the discrete-event queue."""

    time: int
    priority: int
    seq: int
    action: Callable[[], None] = field(compare=False)
    label: str = field(compare=False, default="")


class Engine:
    """A minimal discrete event simulator: a clock plus a priority queue of events."""

    def __init__(
        self,
        verbose: bool = True,
        *,
        construction_guarded: bool = False,
    ):
        """Create an empty simulator with the clock at zero."""
        self.now: int = 0
        self._event_queue: list[Event] = []
        self._seq = itertools.count()
        self.verbose = verbose
        self.log_lines: list[str] = []
        self.metrics: list = []
        self.log_sink = None
        self._construction_guarded = construction_guarded

    def _finish_construction(self) -> None:
        """Permit event execution after the composition root binds RNG state."""
        self._construction_guarded = False

    def schedule(self, delay: int, action: Callable[[], None],
                 label: str = "", priority: int = 0) -> None:
        """Schedule an action `delay` ticks from now.

        Same-tick events fire lowest priority first, then in insertion
        order (the seq counter breaks ties)."""
        if delay < 0:
            raise ValueError(
                f"Cannot schedule an event in the past delay={delay} "
                f"(now={self.now})")
        event = Event(self.now + delay, priority, next(self._seq), action, label)
        heapq.heappush(self._event_queue, event)

    def log(self, who: str, msg: str) -> None:
        """Store one timestamped log line and print it when verbose."""
        line = f"[{fmt(self.now)}] {who}: {msg}"
        self.log_lines.append(line)
        if self.log_sink is not None:
            self.log_sink(line)
        if self.verbose:
            print(line)

    def add_metric(self, metric):
        """Observe this metric after every event."""
        self.metrics.append(metric)
        return metric

    def metric_results(self) -> dict:
        """Return final metric values keyed by metric name."""
        return {metric.name: metric.result() for metric in self.metrics}

    def run(self, until: Optional[int] = None) -> None:
        """Run until the event queue is empty or the optional time limit is reached.

        Events at exactly ``until`` still fire; if later events remain, the
        clock is left at ``until`` so a follow-up run() resumes from there."""
        if self._construction_guarded:
            raise RuntimeError(
                "engine construction is guarded until run-seed binding "
                "finishes"
            )
        while self._event_queue:
            next_event_time = self._event_queue[0].time
            if until is not None and next_event_time > until:
                self.now = until
                break

            event = heapq.heappop(self._event_queue)
            if event.time < self.now:
                raise ValueError(f"Event scheduled in the past: {event} "
                                 f"(now={self.now})")
            self.now = event.time
            event.action()
            for metric in self.metrics:
                metric.observe(self)
