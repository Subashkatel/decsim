"""The window transfers: sends in a window's name over the links.

The fabric adapter the two decoder output ports hold: it turns a window
or a job into one Link send, so a caller names what it moves and never
the fabric. Every send rides the Link port with a TransferAttribution naming the
operation, its patches, the window and the round range, and the request
the transfer serves; the delivery callback runs at the link's delivery.
The rounds a window's send carries are the ones the retention says the
window reads, so a lookahead window's next-operation rounds are named
under their own operation.
Every method here sends: an input that rides no link is the sending
store's own business and never reaches this module.
"""

import functools
from collections.abc import Callable
from typing import Optional

import decsim.engine
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


class WindowTransfers:
    """Sends in a window's or a job's name over the link fabric."""

    link = ports.Port(ports.Link)
    retention = ports.Port(ports.WindowRetention)

    def __init__(self, engine: decsim.engine.Engine) -> None:
        self.engine = engine

    def send_for_window(
        self,
        path: transfer_records.LinkPath,
        window: window_records.Window,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
        payload_bits: Optional[int],
        on_delivered: Callable[[], None],
    ) -> None:
        """Send in a window's name; on_delivered runs at the delivery."""
        round_keys = self._read_keys(window)
        attribution = transfer_records.TransferAttribution.for_window(
            window, operation, request_key, round_keys
        )
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.link.send(
            path, payload_bits, self.engine.now, attribution, delivered
        )

    def send_for_job(
        self,
        path: transfer_records.LinkPath,
        job: decoding_records.DecodeJob,
        *,
        payload_bits: Optional[int],
        request_key: Optional[window_records.DecoderRequestKey] = None,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send in a job's name; on_delivered runs at the delivery.

        Returns the delay the link expects, a scheduler's estimate.
        """
        relation_key = request_key
        if relation_key is None:
            relation_key = job.request_key
        round_keys = self._read_keys(job.window)
        attribution = transfer_records.TransferAttribution.for_job(
            job, relation_key, round_keys
        )
        now_ticks = self.engine.now
        expected_delay_ticks = self.link.expected_delay_ticks(
            path, payload_bits, now_ticks
        )
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.link.send(path, payload_bits, now_ticks, attribution, delivered)
        return expected_delay_ticks

    def send_for_round(
        self,
        path: transfer_records.LinkPath,
        packet: round_records.SyndromeRoundPacket,
        payload_bits: Optional[int],
        on_delivered: Callable[[], None],
    ) -> None:
        """Send in a stored round's name; on_delivered runs at the delivery."""
        attribution = transfer_records.TransferAttribution.for_packet(packet)
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.link.send(
            path, payload_bits, self.engine.now, attribution, delivered
        )

    def send_region(
        self,
        path: transfer_records.LinkPath,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send an escalated region in its request's name.

        Returns the delay the link expects, a scheduler's estimate.
        """
        attribution = transfer_records.TransferAttribution.for_region(region)
        now_ticks = self.engine.now
        message_bits = region.message_bits()
        expected_delay_ticks = self.link.expected_delay_ticks(
            path, message_bits, now_ticks
        )
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.link.send(path, message_bits, now_ticks, attribution, delivered)
        return expected_delay_ticks

    def send_boundary(
        self,
        path: transfer_records.LinkPath,
        attribution: transfer_records.TransferAttribution,
        payload_bits: Optional[int],
        on_delivered: Callable[[transfer_records.Transfer], None],
    ) -> None:
        """Send a boundary in its attribution's name, on the sender's path.

        on_delivered gets the transfer, because a boundary's landing
        reads the delivery's own record.
        """
        self.link.send(
            path,
            payload_bits,
            self.engine.now,
            attribution,
            on_delivered,
        )

    def _read_keys(self, window: window_records.Window) -> tuple:
        read_keys = self.retention.read_keys_for_bounds(
            window.operation_id, window.start_round, window.buffer_hi, window
        )
        return tuple(read_keys)


def _run_at_delivery(
    on_delivered: Callable[[], None], _transfer: transfer_records.Transfer
) -> None:
    on_delivered()
