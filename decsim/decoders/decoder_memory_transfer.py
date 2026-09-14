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

That is the copy rule, and it is a tier's setting: with
<tier>.input in_place the unit reads the rounds where the store keeps
them, so nothing is deposited in the unit's memory, nothing crosses the
input link, and the store's hold is kept for the whole decode. AFS's
processing elements "can directly access the data stored on-chip"
(2001.06598 lines 528-531); Collision Clustering's Init unit loads the
syndrome into the storage elements instead (2309.05558 lines 268-271),
which is the copy row, and the strong hop is a transfer of the assigned
data (Toshio 2510.25222 lines 1248-1250).
"""

import dataclasses
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source

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

    Trace sources: copy_made(job, bits, store_name, memory_name) at every
    landing that deposits rounds, the copy out of the store into the
    unit's own memory (data_path.md hops 5 and 9), and at every boundary
    folded into a masked duplicate the unit reads; hold_registered(job,
    round_keys) instead, for a tier whose input is read in place.
    """

    def __init__(
        self,
        transport,
        engine,
        copies_input_by_pool=None,
        formation_by_pool=None,
    ):
        self.transport = transport
        self.engine = engine
        self.trace = _TraceSources()
        # landing key -> the transfer in flight and the jobs joining it
        self.awaited_by_input: dict = {}
        # pool name -> whether that tier copies its input into the unit;
        # a pool this map does not name copies, which is the default
        self.copies_input_by_pool = copies_input_by_pool or {}
        # pool name -> that tier's event-detection logic, for a run whose
        # rounds arrive raw (controller.detection_events_formed_at)
        self.formation_by_pool = formation_by_pool or {}

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
        and moves nothing. A tier that reads its input in place deposits
        nothing and sends nothing.
        """
        if not self.copies_input(job):
            self._read_in_place(job, on_landed)
            return
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
            # the width that crossed the link is the width that left the
            # store, before this tier forms anything out of it
            bits = job.payload_bits()
            self._form_detection_events(job)
            job.decoder_input = memory.deposit(job)
            job.payloads = []
            job.memory = memory
            if job.decoder_input.rounds:
                source_name = job.input_source_name
                self.trace.copy_made.fire(job, bits, source_name, memory.name)
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

    def fold_into_a_copy(
        self, job: decoding_records.DecodeJob, masked_input
    ) -> None:
        """The job reads a masked duplicate; the unit's rounds stay raw.

        The duplicate is the decoder side's own working copy, so it is
        made and booked here (<tier>.boundary_fold copy, cudaq-x keeps
        raw rounds and applies syndrome_mods at window assembly).
        """
        job.decoder_input = masked_input
        bits = _input_bit_count(masked_input)
        source_name = _input_source_name(job)
        self.trace.copy_made.fire(job, bits, source_name, "masked view")

    def fold_in_place(
        self, job: decoding_records.DecodeJob, masked_input
    ) -> None:
        """The mask is written into the unit's own memory, nothing copied.

        The memory that holds the input performs the write and keeps it
        single-writer itself (DecoderMemory.rewrite, Helios 2301.08419
        lines 632-640); the destination takes what it is handed before
        it acts (OMNeT++ csimplemodule.cc:782-783).

        The mask is written once per input, not once per job. The jobs
        that share one landed input are the forced-class solves of one
        window's request (decision D2), so they share one window and one
        boundary: the solve that starts first writes the mask into the
        unit's memory and every solve after it reads exactly those
        rounds, which is what the copy fold gives each of them too. A
        job whose input another has already written finds the memory
        holding rounds it is not reading yet, and takes them.
        """
        memory = job.memory
        if memory is None:
            raise RuntimeError(
                f"{job.label}: boundary_fold in_place needs the unit's own "
                "copy of the rounds, and this tier reads its input in "
                "place (input: in_place); fold into a copy, or copy the "
                "input"
            )
        resident = memory.input_of(job)
        if resident is not job.decoder_input:
            job.decoder_input = resident
            return
        job.decoder_input = memory.rewrite(job, masked_input)

    def copies_input(self, job: decoding_records.DecodeJob) -> bool:
        """Whether this job's tier is given a copy of the rounds it reads."""
        return self.copies_input_by_pool.get(job.pool, True)

    def _form_detection_events(self, job: decoding_records.DecodeJob) -> None:
        """This tier's event-detection logic runs on the rounds it received.

        The landing is the first demand for them. Under controller-side
        formation no tier holds any logic and the rounds arrived formed
        (controller.detection_events_formed_at).
        """
        formation = self.formation_by_pool.get(job.pool)
        if formation is None:
            return
        job.payloads = formation.form(job.payloads)

    def _read_in_place(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[decoding_records.DecodeJob], None],
    ) -> None:
        """The unit reads the rounds where the store keeps them.

        No link move, no deposit, and the store's hold is kept until the
        decode releases the job, because the rounds it reads are the
        store's own (<tier>.input in_place).
        """
        job.send_input = None
        job.input_landing_ticks = self.engine.now
        self._form_detection_events(job)
        decoder_input = decoder_memory_module.materialize_decoder_input(job)
        job.decoder_input = decoder_input
        job.payloads = []
        round_keys = _round_keys(decoder_input)
        self.trace.hold_registered.fire(job, round_keys)
        on_landed(job)

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
        cancel. This is the one place every cancellation passes through,
        so it is also where the job's tier takes back the rounds it had
        claimed to form: a withdrawn or cancelled decode never reached
        the stage that charges them.
        """
        self._release_formation_claim(job)
        self.transport.cancel(job)
        self._drop_awaited(job)
        self.release(job)
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None

    def _release_formation_claim(self, job: decoding_records.DecodeJob) -> None:
        """The job's tier takes back the rounds it claimed, never formed."""
        formation = self.formation_by_pool.get(job.pool)
        if formation is None:
            return
        formation.release(job)

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
        """Free the job's rounds from its unit's memory; none held is fine.

        A job that read its input in place kept the store's hold for the
        whole decode, so this is where that hold ends.
        """
        memory = job.memory
        if memory is not None:
            memory.take(job)
            job.memory = None
        job.decoder_input = None
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None

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


def _round_keys(decoder_input) -> tuple:
    """The (operation, round) identities one input reads."""
    keys = []
    for round_input in decoder_input.rounds:
        keys.append((round_input.operation_id, round_input.round_index))
    return tuple(keys)


def _is_among(member: decoding_records.DecodeJob, released: list) -> bool:
    for done in released:
        if done is member:
            return True
    return False


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decoder input staging reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
    hold_registered: trace_source.TraceSource = trace_source.new_source()


def _input_source_name(job: decoding_records.DecodeJob) -> str:
    """Where the rounds the mask is folded out of sit."""
    memory = job.memory
    if memory is None:
        return job.input_source_name
    return memory.name


def _input_bit_count(decoder_input) -> Optional[int]:
    """The bits of a landed input; None when any fragment has no bits."""
    bit_count = 0
    for fragment in decoder_input.fragments():
        if fragment.bits is None:
            return None
        bit_count += len(fragment.bits)
    return bit_count
