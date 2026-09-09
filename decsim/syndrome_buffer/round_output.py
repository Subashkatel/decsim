"""A round store's outgoing port: it sends the rounds that leave the store.

Whoever executes a send is an end of that hop, and the end a round
leaves from is the store that holds it. OMNeT++ enforces the rule at
runtime, that a module may only send messages it owns
(`cSimpleModule.cc:334-335`), and gem5 bills a transfer to the port it
left by, never to whoever arranged it (`packet.hh:426-428`). Two kinds
of round leave here. A decode job's input: the window side plans the
decode and the decoder manager says when the input moves (the
accelerator's invoke, then DMA into the unit's memory, then compute),
and this port executes the move, so the link a store's rounds ride is
the store's own fact and not the window side's. And a timing-only
feedback-memory round, which the controller packs and asks for: the
store sends it and frees its own slot at the delivery. This port is the
store's face on the data path in both directions, so a round landing
here is stamped through it too.
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
        job's, and the job is stamped with this store's name: whoever
        records the landing reads where the rounds came from rather than
        deriving it from the job's tier.
        """
        job.input_source_name = self.name
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

        The rounds are already this store's, so it names itself here too,
        and the input rides no link: nothing is sent and no delay is
        charged. Which inputs ride nothing is this store's own decision
        (the decoder input row, copy against in_place), so the landing
        happens here rather than in the link fabric.
        """
        job.input_source_name = self.name
        on_landed()
        return 0

    def mark_published(self, round_key: tuple, publication_tick: int) -> None:
        """A round that landed here is readable from this tick on.

        The record is the store's, so the sender that carried the round
        to this end hands the tick over instead of writing it.
        """
        self.store.mark_publication_tick(round_key, publication_tick)

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
        if is_input_held:
            return functools.partial(self.land_held_input, job)
        return functools.partial(self.send_input, job)
