"""A syndrome buffer's outgoing port: it sends the rounds that leave the store.

The end a round leaves from is the store that holds it, so the store
executes the send: OMNeT++ refuses a module that sends a message it does
not own (src/sim/csimplemodule.cc:333-334), and gem5 bills a transfer to
the port it left by (coherent_xbar.cc:354-357). Two kinds of round leave
here: a decode job's input, which the decoder manager moves at dispatch,
and a timing-only feedback-memory round, whose slot the store frees at
the delivery.

A job's input is read when its bits leave: the store is asked when the
read of the job's rounds completes (book_read), and the move starts
then. A tier that reads in place (<tier>.input in_place) books the same
read and has its input at the read's end with no link crossed (gem5
prices an access where the memory serves it, src/mem/simple_mem.cc
154-174).
"""

import functools
import operator
from collections.abc import Callable
from typing import Optional

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records


class SyndromeBufferOutput:
    """One store's link to the decoders it feeds, bound once by the root.

    reads_in_place is True when the fed tier's unit reads the rounds
    where the store keeps them.
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

        The bits are read here while they are still the job's payloads,
        which the staging clears at the landing. It names the store too,
        since the strong redo sends without asking first. Under in_place
        the read is the landing.
        """
        self.name_this_store(job)
        self._read_the_rounds_before(job)
        payload_bits = job.payload_bits()
        carried = job.rounds_before + tuple(job.payloads)
        round_keys = _payload_round_keys(carried)
        read_tick = self.store.book_read(round_keys)
        job.store_read_ticks = read_tick - self.engine.now
        if self.reads_in_place:
            return self._land_at(read_tick, on_landed)
        if read_tick == self.engine.now:
            return self._move(job, payload_bits, on_landed)
        return self._move_at(read_tick, job, payload_bits, on_landed)

    def _read_the_rounds_before(self, job: decoding_records.DecodeJob) -> None:
        """Add the raw rounds before the job's first that its reader needs.

        A decoder that forms the events needs the rounds before the
        job's first that its recipes read, less those it holds
        (rounds_needed_before). They leave in the job's own read and
        ride its one transfer, as a gem5 DMA request covers its whole
        range (src/dev/dma_device.cc:195-207).
        """
        if self.detection_events is None or not job.payloads:
            return
        first = job.payloads[0]
        operation_id = first.operation_id
        last_round = _last_round_of(job.payloads, operation_id)
        needed_rounds = self.detection_events.rounds_needed_before(
            self.reader_seat, operation_id, first.round_index, last_round
        )
        rounds_before = []
        for round_index in needed_rounds:
            round_key = (operation_id, round_index)
            in_order = self._retained_round_before(round_key)
            rounds_before.extend(in_order)
        job.rounds_before = tuple(rounds_before)

    def _retained_round_before(self, round_key: tuple) -> list:
        """A round the job reads before its first, in fragment order."""
        fragments = self.store.retained_fragments(round_key)
        by_fragment_index = operator.attrgetter("fragment_index")
        return sorted(fragments, key=by_fragment_index)

    def land_held_input(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[], None],
    ) -> int:
        """A job resubmitted after a withdrawal: its rounds never left.

        Nothing is sent and no delay is charged; which inputs ride
        nothing is this store's decision (copy against in_place).
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
        """Read one timing-only round, then send it to the decoder side.

        It leaves by a read of the store like any other round, so it
        waits its turn on the store's ports; the store frees its slot at
        the delivery, then the asker hears.
        """
        landed = functools.partial(
            self._free_memory_round, packed, on_delivered
        )
        send = functools.partial(
            self.transfers.send_for_round,
            self.path,
            packed.packet,
            packed.wire_bits,
            landed,
        )
        round_keys = (packed.round_key,)
        read_tick = self.store.book_read(round_keys)
        if read_tick == self.engine.now:
            send()
            return
        read_delay = read_tick - self.engine.now
        self.engine.schedule(read_delay, send, label="syndrome buffer read")

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

        It happens where the job is bound to this store, not where it
        sends, because a resubmitted job runs no send (land_held_input).
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

        The link is asked what a send at the read's end would pay.
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


def _last_round_of(payloads, operation_id) -> int:
    """The last round of the operation a job's payloads carry."""
    last_round = 0
    for fragment in payloads:
        if fragment.operation_id != operation_id:
            continue
        last_round = max(last_round, fragment.round_index)
    return last_round
