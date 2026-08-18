"""Move one decoder request into its assigned unit's memory after an exact delay.

The default transport owns delay and cancellation only. Materialization into
the unit's ``DecoderMemory`` and the upstream hold release belong to the
decoder manager's receiver.
"""

from __future__ import annotations

from typing import Callable

from .message import DecodeJob


class FixedLatencyDecoderMemoryTransfer:
    """Deliver one admitted job to its receiver after a fixed delay.

    The decoder cannot read the input before delivery. One in-flight key per
    request makes cancellation observable: a request cancelled before its
    delivery event never reaches the receiver, and cancelling an unknown or
    already delivered request does nothing.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self._in_flight_keys = set()

    @staticmethod
    def _key(job: DecodeJob):
        return job.request_key if job.request_key is not None else id(job)

    def deliver(
        self, job: DecodeJob, delay_ticks: int,
        receiver: Callable[[DecodeJob], None],
    ) -> None:
        if type(delay_ticks) is not int:
            raise TypeError("delay_ticks must be an exact int")
        if delay_ticks < 0:
            raise ValueError("delay_ticks must be nonnegative")
        if not callable(receiver):
            raise TypeError("decoder input receiver must be callable")
        key = self._key(job)
        if key in self._in_flight_keys:
            raise RuntimeError(
                f"decoder input for {job.label!r} is already in flight")
        self._in_flight_keys.add(key)

        def complete() -> None:
            if key not in self._in_flight_keys:
                return                      # cancelled before this delivery
            self._in_flight_keys.remove(key)
            receiver(job)

        if delay_ticks == 0:
            complete()
        else:
            self.engine.schedule(
                delay_ticks, complete,
                label=f"fixed-latency decoder input {job.label}",
            )

    def cancel(self, job: DecodeJob) -> None:
        """Suppress delivery of a request that has not arrived yet."""
        self._in_flight_keys.discard(self._key(job))
