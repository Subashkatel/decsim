"""The ready-queue discipline of a decoder pool: which waiting job is next.

A scheduler is a class with pop(queue), built by its frozen Settings
record, which the decoder manager's scheduler setting holds. It stays a
research knob: Triage (2605.04459) makes the M-for-N scheduler the
thing to vary.
"""

import dataclasses
from typing import Protocol

import decsim.records.decoding as decoding_records


class Scheduler(Protocol):
    """A ready-queue rule: the waiting job a free unit serves next."""

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove and return the next job of the queue."""


class FifoScheduler:
    """Dispatch ready jobs in admission order within each decoder pool.

    The manager models each pool as identical non-preemptive service
    units. FIFO is the minimal pool-locally work-conserving baseline: it
    makes no unsupported deadline, cost, or microarchitecture claim.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """FIFO has no knob; the record names the rule and builds it."""

        def build(self) -> "FifoScheduler":
            """A fresh scheduler: each manager orders its own queue."""
            return FifoScheduler()

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove the ready job admitted earliest."""
        return queue.pop(0)
