"""Ready-queue discipline for modeled decoder accelerator pools."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..message import DecodeJob


@runtime_checkable
class Scheduler(Protocol):
    """Which ready job a decoder pool serves next."""

    def pop(self, queue: list[DecodeJob]) -> DecodeJob:
        """Remove and return the next job of one pool's ready queue."""


class FifoScheduler:
    """Dispatch ready jobs in admission order within each decoder pool.

    DecoderManager models each pool as identical non-preemptive service units.
    FIFO is the minimal pool-locally work-conserving baseline: it makes no
    unsupported deadline, cost, or accelerator-microarchitecture claim.
    """

    def pop(self, queue: list[DecodeJob]) -> DecodeJob:
        """Remove the ready job admitted earliest."""
        return queue.pop(0)
