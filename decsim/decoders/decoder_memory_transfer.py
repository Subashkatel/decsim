"""Moves a job's rounds from the weak syndrome buffer into the unit's memory.

The transport (the job's input link, made cancellable), the landing into
DecoderMemory, the release and the cancel. DecoderInputStaging is the
sole writer of job.decoder_input, job.memory and job.input_hold; the
transport owns only the delivery in flight and its cancellation, so a
supplied transport cannot bypass decoder memory.

A unit reads one copy of one input: a job whose rounds the unit holds
reads them, and a job staged while their transfer is in flight joins
that landing, as gem5's MSHR serves several targets with one fill
(src/mem/cache/mshr.hh). With <tier>.input in_place the unit reads the
rounds where the store keeps them: nothing is deposited or crosses the
input link, and the store's hold lasts the whole decode. AFS's
processing elements "can directly access the data stored on-chip"
(2001.06598 lines 528-531); Collision Clustering's Init unit loads the
syndrome into its storage elements instead (2309.05558 lines 268-271),
the copy row.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Optional

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.detection_events as detection_events_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source

SendInput = Callable[[Callable[[], None]], int]


class DecoderInputStaging:
    """Stages each job's input into a unit's memory.

    It frees the input again at the job's release.

    Trace sources: copy_made(job, bits, store_name, memory_name) at
    every landing that deposits rounds and at every boundary folded into
    a masked duplicate; hold_registered(job, round_keys) instead, for a
    tier whose input is read in place.
    """

    def __init__(
        self,
        transport: "CancellableDecoderMemoryTransfer",
        engine: engine_module.Engine,
        copies_input: bool = True,
        formation: Optional[detection_events_module.TierFormation] = None,
    ):
        self.transport = transport
        self.engine = engine
        self.trace = _TraceSources()
        # landing key -> the transfer in flight and the jobs joining it
        self.awaited_by_input: dict = {}
        # whether the pool's tier copies its input into the unit
        # (<tier>.input)
        self.copies_input = copies_input
        # the tier's event-detection logic, for a run that seats the
        # former at the tier's decoder (detection_events.formed_at)
        self.formation = formation

    def stage(
        self,
        job: decoding_records.DecodeJob,
        memory: decoder_memory_module.DecoderMemory,
        on_landed: Callable[[decoding_records.DecodeJob], None],
    ) -> None:
        """Send the input over its link, then land it in the unit's memory.

        At the landing the rounds are deposited, the store's hold is
        dropped and the landing is reported; until then
        input_landing_ticks is the tick the link expects. A job whose
        rounds the unit already holds becomes one more reader and moves
        nothing, and a tier that reads its input in place deposits and
        sends nothing.
        """
        self._claim_formation(job)
        if not self.copies_input:
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
        land = functools.partial(
            self._land, job, memory, landing_key, on_landed
        )
        awaited = _AwaitedLanding(self.engine.now, [])
        self.awaited_by_input[landing_key] = awaited
        expected_delay_ticks = self.transport.deliver(job, send_input, land)
        landing_ticks = self.engine.now + expected_delay_ticks
        job.input_landing_ticks = landing_ticks
        awaited.expected_landing_ticks = landing_ticks

    def is_landing_into(
        self,
        memory: decoder_memory_module.DecoderMemory,
        job: decoding_records.DecodeJob,
    ) -> bool:
        """Whether this job's rounds are already on their way into that memory.

        Such a job joins the landing in flight, so it sends nothing of
        its own.
        """
        landing_key = memory.landing_key(job)
        return landing_key in self.awaited_by_input

    def fold_into_a_copy(
        self,
        job: decoding_records.DecodeJob,
        masked_input: decoding_records.DecoderInput,
    ) -> None:
        """The job reads a masked duplicate; the unit's rounds stay raw.

        The duplicate is the decoder side's own working copy, made and
        booked here (<tier>.boundary_fold copy; cudaq-x keeps raw rounds
        and applies syndrome_mods at window assembly).
        """
        job.decoder_input = masked_input
        bits = _input_bit_count(masked_input)
        source_name = _input_source_name(job)
        self.trace.copy_made.fire(job, bits, source_name, "masked view")

    def fold_in_place(
        self,
        job: decoding_records.DecodeJob,
        masked_input: decoding_records.DecoderInput,
    ) -> None:
        """The mask is written into the unit's own memory, nothing copied.

        The memory keeps itself single-writer (DecoderMemory.rewrite,
        Helios 2301.08419 lines 632-640). The mask is written once per
        input: the jobs that share one landed input are the forced-class
        solves of one window's request, with one boundary final before
        any starts, so the first solve writes the mask and every later
        one reads those rounds, as the copy fold would give it.
        """
        memory = job.memory
        if memory.is_rewritten(job):
            job.decoder_input = memory.input_of(job)
            return
        job.decoder_input = memory.rewrite(job, masked_input)

    def _form_detection_events(self, job: decoding_records.DecodeJob) -> None:
        """This tier's event-detection logic runs on the rounds it received.

        The landing is the first demand for them; the rounds before
        them, when the store sent them, are held by the logic and not
        deposited. A tier whose decoder is not a seat of
        detection_events.formed_at holds no logic, and its rounds
        arrived formed.
        """
        rounds_before = job.rounds_before
        job.rounds_before = ()
        if self.formation is None:
            return
        job.payloads = self.formation.form(job.payloads, rounds_before)

    def _land(
        self,
        job: decoding_records.DecodeJob,
        memory,
        landing_key: tuple,
        on_landed: Callable[[decoding_records.DecodeJob], None],
        delivered: decoding_records.DecodeJob,
    ) -> None:
        """The input's transfer landed: deposit it, drop the store's hold."""
        del delivered
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
        if hold is not None:  # the weak buffer may drop the rounds now
            hold()
            job.input_hold = None
        self._land_joined(landing_key, memory)
        on_landed(job)

    def _read_in_place(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[decoding_records.DecodeJob], None],
    ) -> None:
        """The unit reads the rounds where the store keeps them.

        Nothing is deposited, and the store's hold is kept until the
        decode releases the job (<tier>.input in_place). The store's end
        lands the job at the read's end with no link crossed
        (syndrome_buffer/round_output.py), so a cancel before then
        suppresses the landing as it does a transfer's.
        """
        send_input = job.send_input
        job.send_input = None
        land = functools.partial(self._land_in_place, job, on_landed)
        expected_delay_ticks = self.transport.deliver(job, send_input, land)
        job.input_landing_ticks = self.engine.now + expected_delay_ticks

    def _land_in_place(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[decoding_records.DecodeJob], None],
        delivered: decoding_records.DecodeJob,
    ) -> None:
        """The read completed: the unit reads the store's rounds now."""
        del delivered
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
        awaited: "_AwaitedLanding",
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

        Every step is idempotent, so a request at any stage is safe to
        cancel. Every cancellation passes here, so this is also where
        the tier takes back the rounds it had claimed to form: a
        cancelled decode never reached the stage that charges them.
        """
        self._release_formation_claim(job)
        self.transport.cancel(job)
        self._drop_awaited(job)
        self.release(job)
        hold = job.input_hold
        if hold is not None:
            hold()
            job.input_hold = None

    def _claim_formation(self, job: decoding_records.DecodeJob) -> None:
        """The job's tier claims the rounds it will form, while they are held.

        Its rounds are held in the store until it lands or reads them,
        and a round retires only after, so every job that reads a round
        claims it before the round can retire and the claim can go then.
        """
        if self.formation is None:
            return
        self.formation.rounds_to_form(job)

    def _release_formation_claim(self, job: decoding_records.DecodeJob) -> None:
        """The job's tier takes back the rounds it claimed, never formed."""
        if self.formation is None:
            return
        self.formation.release(job)

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

    def release_service_members(self, members: list) -> None:
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

    A request cancelled before its landing never reaches the receiver,
    and cancelling an unknown or landed request does nothing.
    """

    def __init__(self, engine: engine_module.Engine) -> None:
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


@dataclasses.dataclass
class _AwaitedLanding:
    """One transfer in flight into a unit.

    joined holds the jobs that join its landing.
    """

    expected_landing_ticks: int
    joined: list


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
    """Every event the decoder input staging reports, as one member."""

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
