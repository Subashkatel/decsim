"""Move one decoder request from upstream data to decoder-input store input.

The default transfer waits for a requested delay, materializes the input, then
hands the ready job to the decoder manager.
"""

from __future__ import annotations

from typing import Callable

from .decoder_input_store import DecoderInputStore
from .message import DecodeJob


class FixedLatencyDecoderInputTransfer:
    """Materialize one admitted job after a fixed delay.

    The decoder cannot read the input before completion. On completion, this
    object deposits stored input, releases the upstream hold through the callback,
    and delivers the ready job.
    """

    def __init__(self, engine, *, input_store=None) -> None:
        self.engine = engine
        self.input_store = (
            DecoderInputStore() if input_store is None else input_store
        )
        self._cancelled_pending_keys = set()

    @staticmethod
    def _key(job: DecodeJob):
        return job.request_key if job.request_key is not None else id(job)

    def deliver(
        self, job: DecodeJob, delay_ticks: int,
        receiver: Callable[[DecodeJob], None], *, on_materialized=None,
    ) -> None:
        if type(delay_ticks) is not int:
            raise TypeError("delay_ticks must be an exact int")
        if delay_ticks < 0:
            raise ValueError("delay_ticks must be nonnegative")
        if not callable(receiver):
            raise TypeError("decoder input receiver must be callable")
        if on_materialized is not None and not callable(on_materialized):
            raise TypeError("materialization callback must be callable")
        key = self._key(job)
        self.input_store.reserve(key)

        def complete() -> None:
            if key in self._cancelled_pending_keys:
                self._cancelled_pending_keys.remove(key)
                return
            try:
                decoder_input = self.input_store.deposit(key, job)
                job.decoder_input = decoder_input
                job.payloads = []
                if on_materialized is not None:
                    on_materialized(job)
                receiver(job)
            except BaseException:
                try:
                    self.input_store.discard(key)
                except RuntimeError:
                    pass
                raise

        if delay_ticks == 0:
            complete()
        else:
            self.engine.schedule(
                delay_ticks, complete,
                label=f"fixed-latency decoder input {job.label}",
            )

    def cancel(self, job: DecodeJob) -> None:
        """Cancel an input that has not yet materialized."""
        key = self._key(job)
        if job.decoder_input is not None:
            self.release(job)
            return
        try:
            self.input_store.discard(key)
        except RuntimeError:
            return
        self._cancelled_pending_keys.add(key)

    def release(self, job: DecodeJob) -> None:
        """Release the input-store allocation after decoder service/cancellation."""
        key = self._key(job)
        if job.decoder_input is not None:
            taken = self.input_store.take(key)
            if taken is not job.decoder_input:
                raise RuntimeError("decoder-input store input identity changed")
            job.decoder_input = None
