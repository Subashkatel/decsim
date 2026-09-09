"""A round store's outgoing port: it sends the rounds a decode job reads.

Whoever executes a send is an end of that hop, and the end a decoder
input leaves from is the store that holds the rounds. OMNeT++ enforces
the rule at runtime, that a module may only send messages it owns
(`cSimpleModule.cc:334`), and gem5 bills a transfer to the port it left
by, never to whoever arranged it (`coherent_xbar.hh:210-231`,
`packet.hh:426-428`). The window side plans the decode and the decoder
manager says when the input moves (the accelerator's invoke, then DMA
into the unit's memory, then compute); this port executes the move, so
the link a store's rounds ride is the store's own fact and not the
window side's.
"""

import functools
from typing import Callable

import decsim.records.decoding as decoding_records
import decsim.records.transfers as transfer_records


class RoundStoreOutput:
    """One store's link to the decoders it feeds, bound once by the root."""

    def __init__(
        self, transfers, path: transfer_records.LinkPath, name: str
    ) -> None:
        self.transfers = transfers
        self.path = path
        # the store this port belongs to, as the data path names it
        self.name = name

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

    def input_send_for(
        self, job: decoding_records.DecodeJob, is_input_held: bool
    ) -> Callable[[Callable[[], None]], int]:
        """The send the manager calls at dispatch, bound to this job."""
        if is_input_held:
            return functools.partial(self.land_held_input, job)
        return functools.partial(self.send_input, job)
