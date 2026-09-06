"""The simulation clock and the queue of actions scheduled on it.

Time is a count of ticks; one microsecond is one million ticks
(config.TICKS_PER_MICROSECOND). Time never runs backwards. When two
actions are due at the same tick, the one with the lower priority number
runs first; when they share a priority, the one scheduled first runs
first. That is SimPy's ordering (simpy.core.Environment): time, then
priority, then arrival.

The engine reports through three trace sources (observe/trace_source.py)
and holds no observer: line carries each narrated line's text, io_line
carries a component's I/O line and fires only when someone listens, so
a store never walks its contents for a line nobody records, and
action_done carries the tick after every action, SimPy's callbacks on
the event (simpy/core.py step()). The line text is built here, as gem5
builds curTick() and name() before the flag's text (src/base/trace.hh
DPRINTF), so the narrator's listeners write the text they are given.
"""

import dataclasses
import heapq
import itertools
from typing import Callable

import decsim.config as config
import decsim.observe.trace_source as trace_source

Action = Callable[[], None]


@dataclasses.dataclass(order=True)
class Event:
    """One action waiting in the queue, ordered by time, priority, arrival."""

    time: int
    priority: int
    sequence_number: int
    action: Action = dataclasses.field(compare=False)
    label: str = dataclasses.field(compare=False, default="")


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
        """Narrate one timestamped line on the line source."""
        line = self._line_text(component_name, message)
        self.line.fire(line)

    def log_io(
        self, component_name: str, describe_state: Callable[[], str]
    ) -> None:
        """Narrate what a component received, holds, or emitted.

        `describe_state` runs only when the io_line source has a
        listener, so a store never walks its contents for a line nobody
        records.
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
            self.now = event.time
            event.action()
            self.action_done.fire(self.now)

    def _line_text(self, component_name: str, message: str) -> str:
        stamp = config.format_ticks(self.now)
        return f"[{stamp}] {component_name}: {message}"
