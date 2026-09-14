"""The ready-queue discipline of a decoder pool: which waiting job is next.

The Scheduler port stays a research knob: Triage (2605.04459) makes the
M-for-N scheduler the thing to vary.
"""

from typing import Protocol, runtime_checkable

import decsim.records.decoding as decoding_records


@runtime_checkable
class Scheduler(Protocol):
    """Which ready job a decoder pool serves next."""

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove and return the next job of one pool's ready queue."""


class FifoScheduler:
    """Dispatch ready jobs in admission order within each decoder pool.

    The manager models each pool as identical non-preemptive service
    units. FIFO is the minimal pool-locally work-conserving baseline: it
    makes no unsupported deadline, cost, or microarchitecture claim.
    """

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove the ready job admitted earliest."""
        return queue.pop(0)
