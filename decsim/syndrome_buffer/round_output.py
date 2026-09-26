"""A syndrome buffer's outgoing port: it sends the rounds that leave the store.

Whoever executes a send is an end of that hop, and the end a round
leaves from is the store that holds it. OMNeT++ enforces the rule at
runtime, that a module may only send a message it owns:
cSimpleModule::send refuses one whose owner is another module
(`omnetpp src/sim/csimplemodule.cc:333-334`, omnetpp-6.1.0,
the diagnostic at 506-508). And gem5 bills a transfer to the port it left
by, never to whoever arranged it (`coherent_xbar.cc:354-357`). Two kinds
of round leave here. A decode job's input: the window side plans the
decode and the decoder manager says when the input moves (the
accelerator's invoke, then DMA into the unit's memory, then compute),
and this port executes the move, so the link a store's rounds ride is
the store's own fact and not the window side's. And a timing-only
feedback-memory round, which the controller packs and asks for: the
store sends it and frees its own slot at the delivery. What lands in
the store is the round receiver's (weak_syndrome_round_receiver.py); this
one sends.

A job's input is read out of the store when its bits leave, at
dispatch: the store is asked when the read of the job's rounds
completes (book_read) and the move starts then. A tier that reads its
input in place (<tier>.input in_place) has its unit read the store's
words where they sit, so the same read is booked and the input lands
at its completion with no link crossed: one read, priced once, by the
memory the bits leave (gem5 prices an access where the memory serves it,
src/mem/simple_mem.cc:154-174).
"""

import functools
from collections.abc import Callable
from typing import Optional

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records


class SyndromeBufferOutput:
    """One store's link to the decoders it feeds, bound once by the root.

    reads_in_place is the input row of the tier this store feeds: True
    when that tier's unit reads the rounds where the store keeps them.
    """

    transfers = ports.Port(ports.WindowTransfers)
    store = ports.Port(ports.SyndromeBuffer)
    # the fabric, asked what a move that starts after its read will pay
    link = ports.Port(ports.Link)
    # the run's former, asked whether the decoder this store feeds needs
    # the round before a job's first; unbound in a bare store
    detection_events = ports.Port(ports.DetectionEventPlacement, optional=True)

    def __init__(
        self,
        engine: engine_module.Engine,
        path: transfer_records.LinkPath,
        name: str,
        reads_in_place: bool = False,
        reader_seat: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.path = path
        # the store this port belongs to, as the data path names it
        self.name = name
        self.reads_in_place = reads_in_place
        # the seat of the decoder this store feeds (detection_events)
        self.reader_seat = reader_seat

    def send_input(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[], None],
    ) -> int:
        """Read one job's rounds, then move them; the delay expected.

        The bits are the job's payloads, which the staging clears when
        the input lands, so they are read here while they are still the
        job's. The strong re-decode calls this send without asking for
        one first, so this names the store too. Under in_place the read
        is the landing: nothing crosses the link.
        """
        self.name_this_store(job)
        self._read_the_round_before(job)
        payload_bits = job.payload_bits()
        carried = job.round_before + tuple(job.payloads)
        round_keys = _payload_round_keys(carried)
        read_tick = self.store.book_read(round_keys)
        if self.reads_in_place:
            return self._land_at(read_tick, on_landed)
        if read_tick == self.engine.now:
            return self._move(job, payload_bits, on_landed)
        return self._move_at(read_tick, job, payload_bits, on_landed)

    def _read_the_round_before(self, job: decoding_records.DecodeJob) -> None:
        """Add the raw round before the job's first, when its reader needs it.

        A decoder that forms the events and has not formed the job's
        first round reads the round before it (needs_the_round_before),
        so that round leaves the store with the job's own and is priced
        with them. A round the store no longer holds is not read, and
        the former says so if it was needed.
        """
        if self.detection_events is None or not job.payloads:
            return
        first = job.payloads[0]
        needs_it = self.detection_events.needs_the_round_before(
            self.reader_seat, first.operation_id, first.round_index
        )
        if not needs_it:
            return
        round_key = (first.operation_id, first.round_index - 1)
        fragments = self.store.retained_fragments(round_key)
        if fragments is None:
            return
        job.round_before = tuple(sorted(fragments, key=_fragment_order))

    def land_held_input(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[], None],
    ) -> int:
        """A job resubmitted after a withdrawal: its rounds never left.

        The input rides no link: nothing is sent and no delay is charged.
        Which inputs ride nothing is this store's own decision (the
        decoder input row, copy against in_place), so the landing happens
        here rather than in the link fabric.
        """
        # the send is bound to its job by input_send_for, which is also
        # where this store names itself on it; the landing reads nothing
        del job
        on_landed()
        return 0

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Send one timing-only round to the decoder side it feeds.

        The controller packs the round and asks; the store executes the
        send, because it is the end the round leaves from. The slot the
        round held is this store's, so the store frees it at the
        delivery and the asker hears of the landing after that.
        """
        landed = functools.partial(
            self._free_memory_round, packed, on_delivered
        )
        self.transfers.send_for_round(
            self.path, packed.packet, packed.wire_bits, landed
        )

    def _free_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """The round landed: its slot is free, then the asker hears."""
        self.store.release_round(packed.round_key)
        on_delivered()

    def input_send_for(
        self, job: decoding_records.DecodeJob, is_input_held: bool
    ) -> Callable[[Callable[[], None]], int]:
        """The send the manager calls at dispatch, bound to this job."""
        self.name_this_store(job)
        if is_input_held:
            return functools.partial(self.land_held_input, job)
        return functools.partial(self.send_input, job)

    def name_this_store(self, job: decoding_records.DecodeJob) -> None:
        """Stamp the job with this store's name, where its rounds sit.

        The naming happens where the job is bound to this store and not
        only where its send runs, because a resubmitted job whose rounds
        never left runs no send (land_held_input) and would otherwise
        reach the observers with no source at all. Whoever records the
        landing then reads where the rounds came from rather than
        deriving it from the job's tier.
        """
        job.input_source_name = self.name

    def _land_at(self, read_tick: int, on_landed: Callable[[], None]) -> int:
        """The unit reads the store's words: it has them at the read's end."""
        now = self.engine.now
        if read_tick == now:
            on_landed()
            return 0
        delay = read_tick - now
        self.engine.schedule(delay, on_landed, label="syndrome buffer read")
        return delay

    def _move(
        self,
        job: decoding_records.DecodeJob,
        payload_bits: Optional[int],
        on_landed: Callable[[], None],
    ) -> int:
        return self.transfers.send_for_job(
            self.path, job, payload_bits=payload_bits, on_delivered=on_landed
        )

    def _move_at(
        self,
        read_tick: int,
        job: decoding_records.DecodeJob,
        payload_bits: Optional[int],
        on_landed: Callable[[], None],
    ) -> int:
        """Start the move when the read completes; the delay expected.

        The link is asked what a send at the read's end would pay, which
        is exact whenever no later request overtakes it there, the same
        estimate a send made now answers (links/channel.py
        expected_delay_ticks).
        """
        move = functools.partial(self._move, job, payload_bits, on_landed)
        read_delay = read_tick - self.engine.now
        self.engine.schedule(read_delay, move, label="syndrome buffer read")
        link_delay = self.link.expected_delay_ticks(
            self.path, payload_bits, read_tick
        )
        return read_delay + link_delay


def _payload_round_keys(payloads) -> tuple:
    """The stored rounds a job's payloads come from, once each, in order."""
    round_keys = []
    for payload in payloads:
        round_keys.append((payload.operation_id, payload.round_index))
    unique = dict.fromkeys(round_keys)
    return tuple(unique)


def _fragment_order(fragment) -> int:
    return fragment.fragment_index
