"""A round store's outgoing port: it sends the rounds that leave the store.

Whoever executes a send is an end of that hop, and the end a round
leaves from is the store that holds it. OMNeT++ enforces the rule at
runtime, that a module may only send a message it owns:
cSimpleModule::send refuses one whose owner is another module
(`tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334`, omnetpp-6.1.0,
the diagnostic at 506-508). And gem5 bills a transfer to the port it left
by, never to whoever arranged it (`packet.hh:424-431`). Two kinds
of round leave here. A decode job's input: the window side plans the
decode and the decoder manager says when the input moves (the
accelerator's invoke, then DMA into the unit's memory, then compute),
and this port executes the move, so the link a store's rounds ride is
the store's own fact and not the window side's. And a timing-only
feedback-memory round, which the controller packs and asks for: the
store sends it and frees its own slot at the delivery. What lands in
the store is the incoming port's (round_input.py); this one sends.
"""

import functools
from typing import Callable

import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records


class RoundStoreOutput:
    """One store's link to the decoders it feeds, bound once by the root."""

    def __init__(
        self,
        transfers,
        path: transfer_records.LinkPath,
        name: str,
        store,
    ) -> None:
        self.transfers = transfers
        self.path = path
        # the store this port belongs to, as the data path names it
        self.name = name
        self.store = store

    def send_input(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[], None],
    ) -> int:
        """Move one job's rounds to its unit; the delay the link expects.

        The bits are the job's payloads, which the staging clears when
        the input lands, so they are read here while they are still the
        job's. The strong re-decode calls this send without asking for
        one first, so this names the store too.
        """
        self.name_this_store(job)
        payload_bits = job.payload_bits()
        return self.transfers.send_for_job(
            self.path, job, payload_bits=payload_bits, on_delivered=on_landed
        )

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
        where its send runs, because a tier that reads its input in place
        never runs one (<tier>.input in_place,
        decoders/decoder_memory_transfer.py) and would otherwise reach
        the observers with no source at all. Whoever records the landing
        then reads where the rounds came from rather than deriving it
        from the job's tier.
        """
        job.input_source_name = self.name
