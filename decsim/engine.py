"""The simulation clock and the queue of actions scheduled on it.

Time is a count of ticks (config.TICKS_PER_MICROSECOND) and never runs
backwards. Actions due at the same tick run by priority, then in the
order they were scheduled: SimPy's ordering (simpy.core.Environment).

The engine holds no observer and reports through three trace sources:
line, each narrated line; io_line, a component's I/O line, which fires
only when someone listens; and action_done, the tick after every action
(simpy/core.py step()). The line text is built here, as gem5 builds
curTick() and name() before the flag's text (src/base/trace.hh DPRINTF).
"""

import dataclasses
import enum
import heapq
import itertools
from collections.abc import Callable

import decsim.config as config
import decsim.trace_source as trace_source

Action = Callable[[], None]


class Priority(enum.IntEnum):
    """The order two actions due at the same tick run in; lower first.

    gem5 names each priority beside its reason (src/sim/eventq.hh lines
    138-244). The other two order one protected cycle inside its
    boundary tick: the boundary opens and held operations may start,
    then the stream's round is emitted, then a region whose close was
    requested releases its patch (controller/feedback_streams.py).
    """

    DEFAULT = 0
    PROTECTED_ROUND = 1
    PROTECTED_RELEASE = 2


@dataclasses.dataclass(order=True)
class Event:
    """One action waiting in the queue, ordered by time, priority, arrival."""

    time: int
    priority: int
    sequence_number: int
    action: Action = dataclasses.field(compare=False)
    label: str = dataclasses.field(compare=False, default="")
    is_descheduled: bool = dataclasses.field(compare=False, default=False)


class Engine:
    """Runs scheduled actions in time order and narrates through sources."""

    def __init__(self) -> None:
        self.now: int = 0
        self.line = trace_source.TraceSource()
        self.io_line = trace_source.TraceSource()
        self.action_done = trace_source.TraceSource()
        self._event_queue: list[Event] = []
        self._sequence_numbers = itertools.count()

    def schedule(
        self,
        delay: int,
        action: Action,
        label: str = "",
        priority: Priority = Priority.DEFAULT,
    ) -> Event:
        """Queue an action to run `delay` ticks from now; the event returns."""
        if delay < 0:
            raise ValueError(
                f"cannot schedule an action in the past: delay {delay} "
                f"at tick {self.now}"
            )
        due_time = self.now + delay
        sequence_number = next(self._sequence_numbers)
        event = Event(due_time, priority, sequence_number, action, label)
        heapq.heappush(self._event_queue, event)
        return event

    def deschedule(self, event: Event) -> None:
        """Take a queued action back: it never runs and never sets the time.

        gem5's EventQueue::deschedule (src/sim/eventq.hh:790), for a timer
        stopped before it fires, so a run does not end at a timer that
        had nothing left to do.
        """
        event.is_descheduled = True

    @property
    def idle(self) -> bool:
        """True when nothing is scheduled."""
        return not self._event_queue

    def log(self, component_name: str, message: str) -> None:
        """Narrate one timestamped line on the line source."""
        line = self._line_text(component_name, message)
        self.line.fire(line)

    def log_io(
        self, component_name: str, describe_state: Callable[[], str]
    ) -> None:
        """Narrate what a component received, holds, or emitted.

        `describe_state` runs only when the io_line source has a listener.
        """
        if not self.io_line.has_listeners:
            return
        message = describe_state()
        line = self._line_text(component_name, message)
        self.io_line.fire(line)

    def run(self) -> None:
        """Run every scheduled action until the queue is empty."""
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.is_descheduled:
                continue
            self.now = event.time
            event.action()
            self.action_done.fire(self.now)

    def _line_text(self, component_name: str, message: str) -> str:
        stamp = config.format_ticks(self.now)
        return f"[{stamp}] {component_name}: {message}"
