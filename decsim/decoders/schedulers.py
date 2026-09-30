"""The ready-queue discipline of a decoder pool: which waiting job is next.

A scheduler is a class with pop(queue), named by the decoder manager's
scheduler setting. It stays a research knob: Triage (2605.04459) makes
the M-for-N scheduler the thing to vary.
"""

import decsim.records.decoding as decoding_records


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
