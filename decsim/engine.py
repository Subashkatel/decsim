"""Small discrete-event simulation engine."""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import fmt
from .message import is_stable_string


class SimulationFailed(RuntimeError):
    """A later operation attempted to use a terminally failed simulation."""


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
        self._phase = "construction" if construction_guarded else "open"
        self._failure_cause: Optional[BaseException] = None
        self._event_action_in_progress = False
        self._metric_callback_in_progress = False

    def _start_running(self) -> None:
        """Open one root-owned engine for its only primary drain."""
        if self._phase != "construction":
            raise RuntimeError(
                f"engine cannot start running from phase {self._phase}"
            )
        self._phase = "running"

    def _begin_finalization(self) -> None:
        """Seal scheduling after a quiescent primary drain."""
        if self._phase != "running" or self._event_queue:
            raise RuntimeError(
                "engine finalization requires a running engine with an "
                "empty event queue"
            )
        self._phase = "finalizing"

    def _complete(self) -> None:
        """Publish the terminal completed phase."""
        if self._phase != "finalizing" or self._event_queue:
            raise RuntimeError(
                "engine completion requires sealed empty finalization"
            )
        self._phase = "completed"

    def _invalidate(self, cause: BaseException) -> None:
        """Prevent further use while retaining the first originating failure."""
        if not isinstance(cause, BaseException):
            raise TypeError("engine invalidation requires an exception cause")
        if self._phase != "completed":
            if self._failure_cause is None:
                self._failure_cause = cause
            self._phase = "invalid"

    def _raise_if_failed(self, action: str) -> None:
        if self._phase != "invalid":
            return
        failure = SimulationFailed(
            f"engine cannot {action} after the simulation failed"
        )
        raise failure from self._failure_cause

    def schedule(self, delay: int, action: Callable[[], None],
                 label: str = "", priority: int = 0) -> None:
        """Schedule an action `delay` ticks from now.

        Same-tick events fire lowest priority first, then in insertion
        order (the seq counter breaks ties)."""
        self._raise_if_failed("schedule events")
        if self._phase in ("finalizing", "completed"):
            raise RuntimeError(
                f"engine cannot schedule events while {self._phase}"
            )
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
        """Register one observer at a stable boundary, sampling it first."""
        self._raise_if_failed("register metrics")
        if self._phase in ("finalizing", "completed"):
            raise RuntimeError(
                f"engine cannot register metrics while {self._phase}"
            )
        if self._event_action_in_progress or self._metric_callback_in_progress:
            raise RuntimeError("metrics may be registered only at a stable boundary")
        name = metric.name
        version = metric.result_schema_version
        if not is_stable_string(name) or not name:
            raise TypeError("metric name must be a nonempty Unicode scalar string")
        if type(version) is not int or version < 1:
            raise TypeError("metric result_schema_version must be a positive built-in int")
        self._invoke_metric_callback(
            lambda: metric.observe(self), callback_kind="initial observation"
        )
        current_name = metric.name
        current_version = metric.result_schema_version
        if (
            not is_stable_string(current_name)
            or current_name != name
            or type(current_version) is not int
            or current_version != version
        ):
            raise RuntimeError("metric identity changed during initial observation")
        self.metrics.append(metric)
        return metric

    def _invoke_metric_callback(self, callback, *, callback_kind: str):
        if self._event_action_in_progress or self._metric_callback_in_progress:
            raise RuntimeError(
                f"metric {callback_kind} requires a stable engine boundary"
            )
        self._metric_callback_in_progress = True
        try:
            return callback()
        finally:
            self._metric_callback_in_progress = False

    def _observe_metrics(self) -> None:
        for metric in tuple(self.metrics):
            self._invoke_metric_callback(
                lambda metric=metric: metric.observe(self),
                callback_kind="observation",
            )

    def metric_results(self) -> dict:
        """Return final metric values keyed by metric name."""
        return {
            metric.name: self._invoke_metric_callback(
                metric.result, callback_kind="result"
            )
            for metric in self.metrics
        }

    def run(self, until: Optional[int] = None) -> None:
        """Run until the event queue is empty or the optional time limit is reached.

        Events at exactly ``until`` still fire; if later events remain, the
        clock is left at ``until`` so a follow-up run() resumes from there."""
        self._raise_if_failed("run")
        if self._phase == "construction":
            raise RuntimeError(
                "engine construction is guarded until run-seed binding "
                "finishes"
            )
        if self._phase in ("finalizing", "completed"):
            raise RuntimeError(
                f"engine cannot run while {self._phase}"
            )
        self._observe_metrics()
        while self._event_queue:
            next_event_time = self._event_queue[0].time
            if until is not None and next_event_time > until:
                self.now = until
                self._observe_metrics()
                break

            event = heapq.heappop(self._event_queue)
            if event.time < self.now:
                raise ValueError(f"Event scheduled in the past: {event} "
                                 f"(now={self.now})")
            self.now = event.time
            self._event_action_in_progress = True
            try:
                event.action()
            finally:
                self._event_action_in_progress = False
            self._observe_metrics()
