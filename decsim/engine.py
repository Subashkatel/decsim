"""The simulation clock and the queue of actions scheduled on it.

Time is a count of ticks; one microsecond is one million ticks
(config.TICKS_PER_MICROSECOND). Time never runs backwards. When two
actions are due at the same tick, the one with the lower priority number
runs first; when they share a priority, the one scheduled first runs
first. That is SimPy's ordering (simpy.core.Environment): time, then
priority, then arrival.

Metrics observe the run once before the first action and again after
every action.
"""

import dataclasses
import heapq
import itertools
from typing import Any, Callable, Protocol, runtime_checkable

import decsim.config as config

Action = Callable[[], None]


@runtime_checkable
class Metric(Protocol):
    """An observer of the run: it reads the engine and reports one result."""

    name: str
    result_schema_version: int

    def observe(self, engine) -> None:
        """Look at the run before the first action and after every action."""

    def result(self) -> Any:
        """The metric's final value."""


@dataclasses.dataclass(order=True)
class Event:
    """One action waiting in the queue, ordered by time, priority, arrival."""

    time: int
    priority: int
    sequence_number: int
    action: Action = dataclasses.field(compare=False)
    label: str = dataclasses.field(compare=False, default="")


class Engine:
    """Runs scheduled actions in time order and keeps the run's log."""

    def __init__(self, verbose: bool = True, io_trace: bool = False):
        self.now: int = 0
        self.verbose = verbose
        self.io_trace = io_trace
        self.log_lines: list[str] = []
        self.metrics: list = []
        self._event_queue: list[Event] = []
        self._sequence_numbers = itertools.count()

    def schedule(
        self, delay: int, action: Action, label: str = "", priority: int = 0
    ) -> None:
        """Queue an action to run `delay` ticks from now."""
        if delay < 0:
            raise ValueError(
                f"cannot schedule an action in the past: delay {delay} "
                f"at tick {self.now}"
            )
        due_time = self.now + delay
        sequence_number = next(self._sequence_numbers)
        event = Event(due_time, priority, sequence_number, action, label)
        heapq.heappush(self._event_queue, event)

    @property
    def idle(self) -> bool:
        """True when nothing is scheduled."""
        return not self._event_queue

    def log(self, component_name: str, message: str) -> None:
        """Keep one timestamped line, and print it when verbose."""
        stamp = config.format_ticks(self.now)
        line = f"[{stamp}] {component_name}: {message}"
        self.log_lines.append(line)
        if self.verbose:
            print(line)

    def log_io(
        self, component_name: str, describe_state: Callable[[], str]
    ) -> None:
        """Log what a component received, holds, or emitted.

        `describe_state` runs only when the I/O trace is on, so a store
        never walks its contents for a line nobody records.
        """
        if not self.io_trace:
            return
        message = describe_state()
        self.log(component_name, message)

    def add_metric(self, metric):
        """Register a metric under a name no other metric uses."""
        for existing in self.metrics:
            if existing.name == metric.name:
                raise RuntimeError(
                    f"metric name {metric.name!r} is already registered"
                )
        metric.observe(self)
        self.metrics.append(metric)
        return metric

    def metric_results(self) -> dict:
        """Final value of every metric, keyed by metric name."""
        return {metric.name: metric.result() for metric in self.metrics}

    def run(self) -> None:
        """Run every scheduled action until the queue is empty."""
        self._observe_metrics()
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self.now = event.time
            event.action()
            self._observe_metrics()

    def _observe_metrics(self) -> None:
        metrics_now = tuple(self.metrics)
        for metric in metrics_now:
            metric.observe(self)
