"""Small discrete-event simulation engine: a clock plus a priority queue of
events. Simulated time never moves backwards: a delay is nonnegative and the
next event is never behind the clock. Metrics observe once before the first
event and after every event."""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable

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
    def __init__(self, verbose: bool = True, io_trace: bool = False):
        """Create an empty simulator with the clock at zero."""
        self.now: int = 0
        self._event_queue: list[Event] = []
        self._seq = itertools.count()
        self.verbose = verbose
        self.io_trace = io_trace
        self.log_lines: list[str] = []
        self.metrics: list = []

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

    @property
    def idle(self) -> bool:
        """No event is scheduled."""
        return not self._event_queue

    def log(self, who: str, msg: str) -> None:
        """Store one timestamped log line and print it when verbose."""
        line = f"[{fmt(self.now)}] {who}: {msg}"
        self.log_lines.append(line)
        if self.verbose:
            print(line)

    def log_io(self, who: str, describe: Callable[[], str]) -> None:
        """Component I/O narration: what a component received, holds, or
        emitted. `describe` is called only when io_trace is on, so a store
        never walks its contents for a line nobody records."""
        if self.io_trace:
            self.log(who, describe())

    def add_metric(self, metric):
        """Register one observer under a unique name, sampling it first."""
        if any(existing.name == metric.name for existing in self.metrics):
            raise ValueError(f"metric name {metric.name!r} is already registered")
        metric.observe(self)
        self.metrics.append(metric)
        return metric

    def _observe_metrics(self) -> None:
        for metric in tuple(self.metrics):
            metric.observe(self)

    def metric_results(self) -> dict:
        """Return final metric values keyed by metric name."""
        return {metric.name: metric.result() for metric in self.metrics}

    def run(self) -> None:
        """Run until the event queue is empty."""
        self._observe_metrics()
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.time < self.now:
                raise ValueError(f"Event scheduled in the past: {event} "
                                 f"(now={self.now})")
            self.now = event.time
            event.action()
            self._observe_metrics()
