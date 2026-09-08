"""Moves a job's rounds from Buffer 0 into the assigned unit's memory.

The transport (the job's input link, made cancellable), the landing into
DecoderMemory, the release, the cancel. DecoderInputStaging is the sole
writer of job.decoder_input, job.memory and job.input_hold; the
transport owns the in-flight delivery and its cancellation only, so a
supplied transport cannot bypass decoder memory.

A unit reads one copy of one input: a job whose rounds this unit holds
reads them, and a job staged while their transfer is still in flight
joins that landing instead of sending them again, which is gem5's MSHR
with several targets on one fill (src/mem/cache/mshr.hh).
"""

import dataclasses
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records

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
        job: decoding_records.DecodeJob,
        send_input: Optional[SendInput],
        receiver: Callable[[decoding_records.DecodeJob], None],
    ) -> int:
        """Send the input and land the job at delivery; the expected delay."""

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Suppress the landing of a request that has not landed yet."""


@dataclasses.dataclass
class _AwaitedLanding:
    """One transfer in flight into a unit, and the jobs joining its landing."""

    memory: Any
    expected_landing_ticks: int
    joined: list


class DecoderInputStaging:
    """Stages a job's input into a unit's memory and frees it again.

    Trace source: copy_made(job, bits, store_name, memory_name) at every
    landing that deposits rounds, the copy out of the store into the
    unit's own memory (data_path.md hops 5 and 9).
    """

    def __init__(self, transport, engine):
        self.transport = transport
        self.engine = engine
        self.copy_made = trace_source.TraceSource()
        # landing key -> the transfer in flight and the jobs joining it
        self.awaited_by_input: dict = {}

    def stage(
        self,
        job: decoding_records.DecodeJob,
        memory,
        on_landed: Callable[[decoding_records.DecodeJob], None],
    ) -> None:
        """Send the input over its link, then land it in the unit's memory.

        At the landing the rounds are deposited, the Buffer 0 hold is
        dropped and the landing is reported. Until the landing,
        input_landing_ticks is the tick the link expects. A job whose
        rounds this unit already holds becomes one more reader of them
        and moves nothing.
        """
        if memory.holds(job):
            self._read_resident(job, memory, on_landed)
            return
        landing_key = memory.landing_key(job)
        awaited = self.awaited_by_input.get(landing_key)
        if awaited is not None:
            self._join_landing(awaited, job, on_landed)
            return
        send_input = job.send_input
        job.send_input = None

        def land(_delivered: decoding_records.DecodeJob) -> None:
            job.input_landing_ticks = self.engine.now
            bits = job.payload_bits()
            job.decoder_input = memory.deposit(job)
            job.payloads = []
            job.memory = memory
            if job.decoder_input.rounds:
                source_name = job.input_source_name
                self.copy_made.fire(job, bits, source_name, memory.name)
            hold = job.input_hold
            if hold is not None:  # Buffer 0 may drop the rounds now
                hold()
                job.input_hold = None
            self._land_joined(landing_key, memory)
            on_landed(job)

        awaited = _AwaitedLanding(memory, self.engine.now, [])
        self.awaited_by_input[landing_key] = awaited
        expected_delay_ticks = self.transport.deliver(job, send_input, land)
        landing_ticks = self.engine.now + expected_delay_ticks
        job.input_landing_ticks = landing_ticks
        awaited.expected_landing_ticks = landing_ticks

    def _join_landing(
        self,
        awaited: _AwaitedLanding,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[decoding_records.DecodeJob], None],
    ) -> None:
        """These rounds are already on their way here: wait for that landing."""
        job.send_input = None
        job.input_landing_ticks = awaited.expected_landing_ticks
        awaited.joined.append((job, on_landed))

    def _land_joined(self, landing_key, memory) -> None:
        """Read the landed rounds into every job that joined the transfer."""
        awaited = self.awaited_by_input.pop(landing_key)
        for job, on_landed in awaited.joined:
            self._read_resident(job, memory, on_landed)

    def _read_resident(
        self,
        job: decoding_records.DecodeJob,
        memory,
        on_landed: Callable[[decoding_records.DecodeJob], None],
    ) -> None:
        """The rounds are here already: read them, move nothing, land now."""
        job.send_input = None
        job.input_landing_ticks = self.engine.now
        job.decoder_input = memory.add_reader(job)
        job.payloads = []
        job.memory = memory
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None
        on_landed(job)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Drop one job from transport and storage, then free its hold.

        Every step is idempotent, so a request in transport, waiting for
        round credits, already stored or already cleared is safe to
        cancel.
        """
        self.transport.cancel(job)
        self._drop_awaited(job)
        self.release(job)
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None

    def _drop_awaited(self, job: decoding_records.DecodeJob) -> None:
        """Forget a landing this job was sending or waiting for.

        The jobs of one attempt are withdrawn together, so a cancelled
        sender takes its joined readers' landing with it and they are
        cancelled in the same pass.
        """
        unit = job.unit
        if unit is None:
            return
        landing_key = unit.memory.landing_key(job)
        awaited = self.awaited_by_input.get(landing_key)
        if awaited is None:
            return
        del self.awaited_by_input[landing_key]

    def release(self, job: decoding_records.DecodeJob) -> None:
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
        job: decoding_records.DecodeJob,
        send_input: Optional[SendInput],
        receiver: Callable[[decoding_records.DecodeJob], None],
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

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Suppress the landing of a request that has not landed yet."""
        key = _transfer_key(job)
        self._in_flight_keys.discard(key)


def _transfer_key(job: decoding_records.DecodeJob):
    """A window job flies under its request key, any other under itself."""
    if job.request_key is not None:
        return job.request_key
    return id(job)


def _is_among(member: decoding_records.DecodeJob, released: list) -> bool:
    for done in released:
        if done is member:
            return True
    return False
