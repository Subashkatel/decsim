"""Moves a job's rounds from Buffer 0 into the assigned unit's memory.

The transport (the job's input link, made cancellable), the landing into
DecoderMemory, the release, the cancel. DecoderInputStaging is the sole
writer of job.decoder_input, job.memory and job.input_hold; the
transport owns the in-flight delivery and its cancellation only, so a
supplied transport cannot bypass decoder memory.
"""

from typing import Callable, Optional, Protocol, runtime_checkable

import decsim.message as message

SendInput = Callable[[Callable[[], None]], int]


@runtime_checkable
class DecoderMemoryTransfer(Protocol):
    """Carries one admitted job to the decoder side when its link delivers.

    send_input(on_landed) sends the job's input over its link, calls
    on_landed once at delivery, and returns the delay the link expects;
    None means the job carries no input and lands now. receiver(job) runs
    exactly once at the landing unless the request is cancelled first;
    cancel is idempotent.
    """

    def deliver(
        self,
        job: message.DecodeJob,
        send_input: Optional[SendInput],
        receiver: Callable[[message.DecodeJob], None],
    ) -> int:
        """Send the input and land the job at delivery; the expected delay."""

    def cancel(self, job: message.DecodeJob) -> None:
        """Suppress the landing of a request that has not landed yet."""


class DecoderInputStaging:
    """Stages a job's input into a unit's memory and frees it again."""

    def __init__(self, transport, engine):
        self.transport = transport
        self.engine = engine

    def stage(
        self,
        job: message.DecodeJob,
        memory,
        on_landed: Callable[[message.DecodeJob], None],
    ) -> None:
        """Send the input over its link, then land it in the unit's memory.

        At the landing the rounds are deposited, the Buffer 0 hold is
        dropped and the landing is reported. Until the landing,
        input_landing_ticks is the tick the link expects.
        """
        send_input = job.send_input
        job.send_input = None

        def land(_delivered: message.DecodeJob) -> None:
            job.input_landing_ticks = self.engine.now
            job.decoder_input = memory.deposit(job)
            job.payloads = []
            job.memory = memory
            hold = job.input_hold
            if hold is not None:  # Buffer 0 may drop the rounds now
                hold()
                job.input_hold = None
            on_landed(job)

        expected_delay_ticks = self.transport.deliver(job, send_input, land)
        job.input_landing_ticks = self.engine.now + expected_delay_ticks

    def cancel(self, job: message.DecodeJob) -> None:
        """Drop one job from transport and storage, then free its hold.

        Every step is idempotent, so a request in transport, waiting for
        round credits, already stored or already cleared is safe to
        cancel.
        """
        self.transport.cancel(job)
        self.release(job)
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None

    def release(self, job: message.DecodeJob) -> None:
        """Free the job's rounds from its unit's memory; none held is fine."""
        memory = job.memory
        if memory is not None:
            memory.take(job)
            job.memory = None
        job.decoder_input = None

    def release_service_members(self, members) -> None:
        """Return the credits of every request one decode still serves.

        A batch service job holds none of its own. Release is idempotent.
        """
        released = []
        for member in members:
            if _is_among(member, released):
                continue
            released.append(member)
            self.release(member)


class CancellableDecoderMemoryTransfer:
    """Delivers one admitted job to its receiver when its link delivers.

    Unless it is cancelled first. The decoder cannot read the input
    before the landing. One in-flight key per request makes cancellation
    observable: a request cancelled before its landing never reaches the
    receiver, and cancelling an unknown or already landed request does
    nothing.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self._in_flight_keys = set()

    def deliver(
        self,
        job: message.DecodeJob,
        send_input: Optional[SendInput],
        receiver: Callable[[message.DecodeJob], None],
    ) -> int:
        """Send the input and land it at delivery; no input lands now.

        Returns the delay the link expects.
        """
        key = _transfer_key(job)
        if key in self._in_flight_keys:
            raise RuntimeError(
                f"decoder input for {job.label!r} is already in flight"
            )
        self._in_flight_keys.add(key)

        def complete() -> None:
            if key not in self._in_flight_keys:
                return  # cancelled before this landing
            self._in_flight_keys.remove(key)
            receiver(job)

        if send_input is None:
            complete()
            return 0
        return send_input(complete)

    def cancel(self, job: message.DecodeJob) -> None:
        """Suppress the landing of a request that has not landed yet."""
        key = _transfer_key(job)
        self._in_flight_keys.discard(key)


def _transfer_key(job: message.DecodeJob):
    """A window job flies under its request key, any other under itself."""
    if job.request_key is not None:
        return job.request_key
    return id(job)


def _is_among(member: message.DecodeJob, released: list) -> bool:
    for done in released:
        if done is member:
            return True
    return False
