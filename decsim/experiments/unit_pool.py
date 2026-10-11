"""Units run on workers that live for a whole run, each handed on as it ends.

sinter keeps one process per worker for a whole collection and hands a
worker its next batch when it reports back
(sinter/_collection/_collection_manager.py, process_message and
_distribute_unassigned_workers_to_jobs; _collection_worker_state.py
keeps what a worker compiled for its task). This pool does the same
with concurrent.futures: its processes start once, so each keeps its
imports and its own caches (a slicer, a sampler, a compiled backend)
from one piece to the next, and a piece is handed in whenever one
ends, with no wait for the slowest of a batch. With one process the
units run here, in the order handed in.
"""

import concurrent.futures
import itertools
from collections.abc import Callable
from typing import Optional


class UnitPool:
    """At most `processes` units running; the next ended one on request."""

    def __init__(
        self,
        run: Callable[[object, Callable], object],
        measure: Callable,
        processes: int,
    ) -> None:
        self.run = run
        self.measure = measure
        self.processes = processes
        # tag and unit of each running unit, by its future or its ticket
        self.running: dict = {}
        self.tickets = itertools.count()
        self.executor: Optional[concurrent.futures.Executor] = None
        if processes > 1:
            self.executor = concurrent.futures.ProcessPoolExecutor(processes)

    def __enter__(self) -> "UnitPool":
        return self

    def __exit__(self, *_exception) -> None:
        """Running units are let go: a saved piece is never rerun."""
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def has_room(self) -> bool:
        """Whether fewer units run than there are processes."""
        return len(self.running) < self.processes

    def submit(self, tag: object, unit: object) -> None:
        """Start one unit; tag comes back with it when it ends."""
        if self.executor is None:
            ticket = next(self.tickets)
            self.running[ticket] = (tag, unit)
            return
        future = self.executor.submit(self.run, unit, self.measure)
        self.running[future] = (tag, unit)

    def next_ended(self) -> tuple:
        """The tag, unit and outcome of a unit that has ended.

        It waits until one has. Here, it runs the first unit handed in.
        """
        if self.executor is None:
            return self._run_the_first()
        ended, _still_running = concurrent.futures.wait(
            self.running, return_when=concurrent.futures.FIRST_COMPLETED
        )
        future = next(iter(ended))
        tag, unit = self.running.pop(future)
        outcome = future.result()
        return tag, unit, outcome

    def _run_the_first(self) -> tuple:
        ticket = next(iter(self.running))
        tag, unit = self.running.pop(ticket)
        outcome = self.run(unit, self.measure)
        return tag, unit, outcome
