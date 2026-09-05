"""The window transfers: sends in a window's name over the links.

Every send rides the Link port with a TransferAttribution naming the
operation, its patches, the window and the round range, and the request
the transfer serves; the delivery callback runs at the link's delivery.
An input that rides no link lands now or after a fixed delay.
"""

import functools
from typing import Callable, Optional

import decsim.message as message


class WindowTransfers:
    """Sends in a window's or a job's name over the link fabric."""

    def __init__(self, engine, link) -> None:
        self.engine = engine
        self.link = link

    @property
    def now(self) -> int:
        """The engine's tick."""
        return self.engine.now

    def send_for_window(
        self,
        path: message.LinkPath,
        window: message.Window,
        operation: message.Operation,
        request_key: message.DecoderRequestKey,
        payload_bits: Optional[int],
        on_delivered: Callable[[], None],
    ) -> None:
        """Send in a window's name; on_delivered runs at the delivery."""
        attribution = message.TransferAttribution.for_window(
            window, operation, request_key
        )
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.link.send(
            path, payload_bits, self.engine.now, attribution, delivered
        )

    def send_for_job(
        self,
        path: message.LinkPath,
        job: message.DecodeJob,
        *,
        payload_bits: Optional[int],
        request_key: Optional[message.DecoderRequestKey] = None,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send in a job's name; on_delivered runs at the delivery.

        Returns the delay the link expects, a scheduler's estimate.
        """
        relation_key = request_key
        if relation_key is None:
            relation_key = job.request_key
        attribution = message.TransferAttribution.for_job(job, relation_key)
        now_ticks = self.engine.now
        expected_delay_ticks = self.link.expected_delay_ticks(
            path, payload_bits, now_ticks
        )
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.link.send(path, payload_bits, now_ticks, attribution, delivered)
        return expected_delay_ticks

    def send_boundary(
        self,
        attribution: message.TransferAttribution,
        on_delivered: Callable[[message.Transfer], None],
    ) -> None:
        """Send a boundary over decoder_to_decoder; on_delivered gets it."""
        self.link.send(
            message.LinkPath.DECODER_TO_DECODER,
            None,
            self.engine.now,
            attribution,
            on_delivered,
        )

    def land_after(
        self, delay_ticks: int, on_landed: Callable[[], None]
    ) -> int:
        """Land an input that rides no link: now, or after a fixed delay."""
        if delay_ticks == 0:
            on_landed()
            return 0
        self.engine.schedule(delay_ticks, on_landed, label="held input lands")
        return delay_ticks


def result_payload_bits(
    result: message.DecodeResult, operation: message.Operation
) -> int:
    """A result reaches the frame as one bit per logical observable.

    A timing-only result stands for one observable per patch.
    """
    if result.logical_observables is not None:
        return len(result.logical_observables)
    patch_count = len(operation.patches)
    return max(1, patch_count)


def _run_at_delivery(on_delivered: Callable[[], None], _transfer) -> None:
    on_delivered()
