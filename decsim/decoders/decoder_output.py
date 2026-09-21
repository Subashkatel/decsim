"""The decoder side's outgoing sends: the frame, the strong tier, a peer.

Whoever executes a send is an end of that hop. OMNeT++ enforces the same
rule at runtime, that a module may only send a message it owns:
cSimpleModule::send refuses one whose owner is another module
(omnetpp-6.1.0 `src/sim/csimplemodule.cc:333-334`,
the diagnostic at 506-508). And gem5 bills a transfer to the ports it
crossed and never to a proxy that arranged it: the crossbar counts a
packet against the CPU-side and memory-side port ids it went between,
and only once it was successfully sent (`coherent_xbar.cc:354-357`,
`xbar.hh:400-411`), and it hands its forwarding latency to "the
neighbouring object that actually makes the packet wait"
(`packet.hh:424-431`). Two hops leave a decoder: the correction to the
Pauli frame, and the escalation to the strong decoder, which is the
selection of the strong request and then the rounds of its window,
read out of the weak syndrome buffer (Toshio et al. 2510.25222 lines
1247 to 1250 assign the region's syndrome data to the strong decoder
at the switch). The correction and the selection carry a result the
decoder produced, which is also how the reaction path is booked: Yang
et al. 2605.04892 Table I counts the frame update inside the decoder's
own subtotal. A window's boundary does not leave here: it is the window
side's record and leaves by the object that holds it
(windows/window_boundaries.py, decisions.md D12).
"""

import functools
from collections.abc import Callable
from typing import Optional

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

# A selection names the strong request and carries nothing else, so it
# is a control message, and a control message has a size of its own:
# gem5's network sizes every message with no data at control_msg_size,
# 8 bytes, and a data message at its data plus that size (gem5
# src/mem/ruby/network/Network.cc MessageSizeType_to_int, Network.py
# control_msg_size). The width is one name: CUDA-Q QEC's smallest
# request, the one that names a decoder and nothing else, is one int64
# (cudaqx decoder_rpc_wire_format.h ResetRequestPayload, 8 bytes). The
# path's header frames it like any other transfer of the hop.
SELECTION_PAYLOAD_BITS = 64

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
        selects; returns the delay the link expects. A selection is
        one control word, the name of the request, so a bounded hop
        serializes that word and its header, and a hop with a default
        payload does not price it as a region.
        """
        return self.transfers.send_for_job(
            transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            weak_job,
            payload_bits=SELECTION_PAYLOAD_BITS,
            request_key=strong_request_key,
            on_delivered=on_delivered,
        )

    def send_region(
        self,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        """Read a strong window's rounds, then send them to the strong store.

        The rounds leave the weak syndrome buffer here, so the read is
        priced here, once, by that store (book_read), and the send starts
        at its end; a bit is priced by the memory it leaves, at the tick
        it leaves. The send is in the strong request's name and carries
        the rounds' width; returns the delay expected, the read then the
        link, which is exact whenever no later request overtakes it.
        """
        round_keys = _region_round_keys(region)
        read_tick = self.weak_store.book_read(round_keys)
        if read_tick == self.engine.now:
            return self._send_region(region, on_delivered)
        send = functools.partial(self._send_region, region, on_delivered)
        read_delay = read_tick - self.engine.now
        self.engine.schedule(read_delay, send, label="syndrome buffer read")
        link_delay = self.link.expected_delay_ticks(
            transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            region.wire_bits,
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
) -> Optional[int]:
    """A result reaches the frame as one bit per logical observable.

    A timing-only result stands for one observable per patch.
    """
    if result.logical_observables is not None:
        return len(result.logical_observables)
    patch_count = len(operation.patches)
    return max(1, patch_count)


def _region_round_keys(region: round_records.EscalatedRegion) -> tuple:
    """The (operation, round) key of every round the region carries."""
    round_keys = []
    for packet in region.packets:
        round_keys.append((packet.operation_id, packet.round_index))
    return tuple(round_keys)
