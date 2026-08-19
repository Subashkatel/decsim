"""Moves a job's rounds from Buffer 0 into the assigned unit's memory: the
transfer delay (a transport), the landing into DecoderMemory, the release,
the cancel. DecoderInputStaging is the sole writer of job.decoder_input,
job.memory and job.input_hold; the transport owns delay and cancellation of
in-flight deliveries only, so a supplied transport cannot bypass decoder
memory.
"""

from __future__ import annotations

from typing import Callable

from ..message import DecodeJob


class DecoderInputStaging:
    def __init__(self, transport):
        self.transport = transport

    def stage(self, job: DecodeJob, memory, on_landed: Callable[[DecodeJob], None]) -> None:
        """Reserve the input link (the job's reserve_transfer), then after the
        transport delay deposit the rounds in the unit's memory, drop the
        Buffer 0 hold and report the landing."""
        delay_ticks = 0 if job.reserve_transfer is None else job.reserve_transfer()
        job.reserve_transfer = None

        def land(_delivered: DecodeJob) -> None:
            job.decoder_input = memory.deposit(job)
            job.payloads = []
            job.memory = memory
            hold = job.input_hold
            if hold is not None:            # Buffer 0 may drop the rounds now
                hold()
                job.input_hold = None
            on_landed(job)

        self.transport.deliver(job, delay_ticks, land)

    def cancel(self, job: DecodeJob) -> None:
        """Drop one job from transport and storage, then free its upstream
        hold; every step is idempotent, so a request in transport, waiting for
        round credits, already stored or already cleared is safe to cancel."""
        self.transport.cancel(job)
        self.release(job)
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None

    def release(self, job: DecodeJob) -> None:
        """Free the job's rounds from its unit's memory; a job holding none is untouched."""
        memory = getattr(job, "memory", None)
        if memory is not None:
            memory.take(job)
            job.memory = None
        job.decoder_input = None

    def release_service_members(self, members) -> None:
        """Return the credits of every request one decode still serves; a
        batch service job holds none of its own. Release is idempotent."""
        released = []
        for member in members:
            if any(done is member for done in released):
                continue
            released.append(member)
            self.release(member)


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
        if delay_ticks < 0:
            raise ValueError("delay_ticks must be nonnegative")
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
