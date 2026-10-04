"""The decoder side's outgoing sends: the frame, the strong tier, a peer.

Whoever executes a send is an end of that hop: OMNeT++ refuses a send of
a message another module owns (omnetpp-6.1.0
src/sim/csimplemodule.cc:333-334), and gem5 bills a transfer to the
ports it crossed, never to a proxy that arranged it
(coherent_xbar.cc:354-357, xbar.hh:400-411, packet.hh:424-431). Two hops
leave a decoder: the correction to the Pauli frame, and the escalation
to the strong decoder, the selection of the strong request and then the
rounds of its window read out of the weak syndrome buffer (Toshio et al.
2510.25222 lines 1247-1250). Yang et al. 2605.04892 Table I counts the
frame update inside the decoder's own subtotal. A window's boundary is
the window side's record and leaves from there
(windows/window_boundaries.py).
"""

import functools
from collections.abc import Callable

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

# What a tier's answer carries beside its flips. A strong answer comes
# back across the wall while other requests are open, so it names the
# request it answers, and gem5 sizes a response that carries data as
# that data plus the control size (gem5 src/mem/ruby/network/Network.cc
# MessageSizeType_to_int, Response_Data). CUDA-Q's reply echoes the
# request's id, "enabling
# out-of-order or pipelined verification of responses"
# (cudaq_realtime_message_protocol.md, Request ID Semantics). A weak
# answer stays on the board with its frame and decsim prices no name on
# that hop; a run that frames it sets the hop's
# header_bits_per_transfer.
ANSWER_NAME_BITS_BY_TIER = {
    window_records.DecoderTier.WEAK: 0,
    window_records.DecoderTier.STRONG: window_records.REQUEST_KEY_WIRE_BITS,
}

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

    transfers = ports.Port(ports.WindowTransfers)
    # a run with no frame commits its corrections nowhere
    frame = ports.Port(ports.Frame, optional=True)
    # the store an escalated region's rounds are read out of, and the
    # fabric asked what their move will pay once that read is done
    weak_store = ports.Port(ports.SyndromeBuffer)
    link = ports.Port(ports.Link)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine

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
        flip_bits = result_payload_bits(result, operation)
        name_bits = ANSWER_NAME_BITS_BY_TIER[request_key.tier]
        payload_bits = flip_bits + name_bits
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
        selects; returns the delay the link expects. A selection is the
        request's name and nothing else (records/windows.py
        REQUEST_KEY_WIRE_BITS), so a bounded hop serializes that word
        and its header.
        """
        return self.transfers.send_for_job(
            transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            weak_job,
            payload_bits=window_records.REQUEST_KEY_WIRE_BITS,
            request_key=strong_request_key,
            on_delivered=on_delivered,
        )

    def send_region(
        self,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        """Read a strong window's rounds, then send them to the strong store.

        The rounds leave the weak syndrome buffer here, so that store
        prices the read once (book_read) and the send starts at its end.
        The send carries the request's name and the rounds
        (EscalatedRegion.message_bits); returns the read's time plus the
        link's estimate (links/channel.py expected_delay_ticks).
        """
        round_keys = region.round_keys
        read_tick = self.weak_store.book_read(round_keys)
        if read_tick == self.engine.now:
            return self._send_region(region, on_delivered)
        send = functools.partial(self._send_region, region, on_delivered)
        read_delay = read_tick - self.engine.now
        self.engine.schedule(read_delay, send, label="syndrome buffer read")
        message_bits = region.message_bits()
        link_delay = self.link.expected_delay_ticks(
            transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            message_bits,
            read_tick,
        )
        return read_delay + link_delay

    def _send_region(
        self,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        return self.transfers.send_region(
            transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            region,
            on_delivered,
        )

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
) -> int:
    """A result reaches the frame as one bit per logical observable.

    A timing-only result stands for one observable per patch.
    """
    if result.logical_observables is not None:
        return len(result.logical_observables)
    patch_count = len(operation.patches)
    return max(1, patch_count)
