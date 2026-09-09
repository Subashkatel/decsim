"""The decoder side's outgoing sends: the frame, the strong tier, a peer.

Whoever executes a send is an end of that hop. OMNeT++ enforces the same
rule at runtime, that a module may only send messages it owns
(`cSimpleModule.cc:334`), and gem5 bills a transfer to the ports it
crossed and never to a proxy that arranged it: the crossbar counts a
packet against the CPU-side and memory-side port ids it went between,
and only once it was successfully sent (`coherent_xbar.cc:354-357`,
`xbar.hh:400-411`), and it hands its forwarding latency to "the
neighbouring object that actually makes the packet wait"
(`packet.hh:424-431`). Three hops leave a decoder: the correction to the
Pauli frame, the escalation selection to the strong decoder, and one
window's boundary to the decoder of a dependent window. The window side
decides that they happen, and this decoder-side component executes them,
which is also how the reaction path is booked: Yang et al. 2605.04892
Table I counts the frame update inside the decoder's own subtotal.
"""

import functools
from typing import Callable, Optional

import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

# which output link a tier's result leaves by
FRAME_PATH_BY_TIER = {
    window_records.DecoderTier.WEAK: (
        transfer_records.LinkPath.WEAK_DECODER_TO_FRAME
    ),
    window_records.DecoderTier.STRONG: (
        transfer_records.LinkPath.STRONG_DECODER_TO_FRAME
    ),
}


class DecoderOutput:
    """Sends one decoder's answers where they go, and charges the frame."""

    def __init__(self, transfers, frame) -> None:
        self.transfers = transfers
        self.frame = frame

    def publish(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Send the result on its tier's output link; commit it at delivery.

        A weak result rides weak_decoder_to_frame, a strong one
        strong_decoder_to_frame; the frame's priced write, when the run
        has a frame, gates on_committed.
        """
        output_path = FRAME_PATH_BY_TIER[request_key.tier]
        payload_bits = result_payload_bits(result, operation)
        commit = functools.partial(
            self._commit, window.key, result, request_key, on_committed
        )
        self.transfers.send_for_window(
            output_path, window, operation, request_key, payload_bits, commit
        )

    def send_selection(
        self,
        weak_job: decoding_records.DecodeJob,
        strong_request_key: window_records.DecoderRequestKey,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send one window's escalation to the strong decoder.

        The send is in the weak job's name for the strong request it
        selects; returns the delay the link expects.
        """
        return self.transfers.send_for_job(
            transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            weak_job,
            payload_bits=None,
            request_key=strong_request_key,
            on_delivered=on_delivered,
        )

    def send_boundary(
        self,
        attribution: transfer_records.TransferAttribution,
        payload_bits: Optional[int],
        on_delivered: Callable[[transfer_records.Transfer], None],
    ) -> None:
        """Send one window's boundary to a dependent window's decoder.

        The bits are the ones the window side counted on the seam the
        message updates; None leaves the transfer to the card.
        """
        self.transfers.send_boundary(attribution, payload_bits, on_delivered)

    def _commit(
        self,
        window_key: tuple,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Charge and install one final correction, then call back."""
        if self.frame is None:
            on_committed()
            return
        self.frame.commit_correction(
            window_key=window_key,
            logical_observables=result.logical_observables,
            request_key=request_key,
            on_committed=on_committed,
        )


def result_payload_bits(
    result: decoding_records.DecodeResult, operation: program_records.Operation
) -> Optional[int]:
    """A result reaches the frame as one bit per logical observable.

    A timing-only result stands for one observable per patch.
    """
    if result.logical_observables is not None:
        return len(result.logical_observables)
    patch_count = len(operation.patches)
    return max(1, patch_count)
