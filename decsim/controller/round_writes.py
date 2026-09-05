"""The rounds held while a store is full, retried in completion order.

The backpressure law of the readout path: a store answers has_room before
any write; a finished round that finds no room waits here, upstream of
the store, and enters when a slot frees, in the order the rounds were
completed. gem5's blocked port keeps the request at the requester and
re-sends it when clearBlocked schedules the retry
(src/mem/cache/base.cc, setBlocked, clearBlocked, processSendRetry);
Ciw's Type I blocking keeps the customer at the upstream node and
releases the longest blocked one when the destination has capacity
(ciw/node.py, block_individual, release_blocked_individual). Nothing is
dropped and nothing is reordered; the QPU never pauses.
"""

from typing import Callable


class HeldRounds:
    """The waiting line in front of the stores."""

    def __init__(self) -> None:
        # (held round, the admission it retries), in completion order
        self.waiting: list = []

    def hold(self, held, admit: Callable[[object], bool]) -> None:
        """Keep one round until a retry admits it."""
        self.waiting.append((held, admit))

    def retry(self) -> None:
        """A slot freed: admit from the head, stop at the first refused."""
        while self.waiting:
            held, admit = self.waiting[0]
            admitted = admit(held)
            if not admitted:
                return
            self.waiting.pop(0)

    @property
    def count(self) -> int:
        """How many rounds wait."""
        return len(self.waiting)
