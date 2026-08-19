"""Ready-queue discipline for modeled decoder accelerator pools."""

from __future__ import annotations

from ..message import DecodeJob


class FifoScheduler:
    """Dispatch ready jobs in admission order within each decoder pool.

    DecoderManager models each pool as identical non-preemptive service units.
    FIFO is the minimal pool-locally work-conserving baseline: it makes no
    unsupported deadline, cost, or accelerator-microarchitecture claim.
    """

    def pop(self, queue: list[DecodeJob]) -> DecodeJob:
        """Remove the ready job admitted earliest."""
        return queue.pop(0)
