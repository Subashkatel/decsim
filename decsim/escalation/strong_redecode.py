"""The strong re-decode: the window side of the strong tier.

When the escalation policy escalates a weak result, StrongRedecode asks
the shape for the strong window's job (strong_window_shapes.py), sends
the window's selection over weak_decoder_to_strong_decoder, tells the
decoder side to await the request's result (the DecodeQueue port), and
submits the job now or when the shape's condition fires: the far weak
boundary's commit or the operation's terminal data. When the policy
decodes both tiers at once (Toshio et al. 2510.25222 Sec. III A, Step
1), it builds the strong sibling the requester enqueues beside the weak
job and selects it at the verdict. A strong input landed in its unit
waits for its selection to arrive before it decodes; the submission
uses the per-job law, decode_queue.enqueue(job, send_input, on_decoded)
with the committer's accept_strong_result as the return path.
"""

import functools
from typing import Callable, Optional

import decsim.message as message

LOG_SOURCE = "DecoderCluster"


class StrongRedecode:
    """Selects, submits and lands the strong tier's re-decode of a window."""

    def __init__(
        self,
        engine,
        shape,
        transfers,
        decode_queue,
        on_strong_decoded: Callable[
            [message.DecodeJob, message.DecodeResult], None
        ],
    ) -> None:
        self.engine = engine
        self.shape = shape
        self.transfers = transfers
        self.decode_queue = decode_queue
        # the strong job's return path, the committer's accept_strong_result
        self.on_strong_decoded = on_strong_decoded
        self.selections = _StrongSelections()

    # ---- the two entries

    def parallel_strong_submission(
        self, weak_job: message.DecodeJob
    ) -> message.Submission:
        """The strong sibling started with the weak job (the paper's Step 1).

        Its context is held and its send built now; the verdict selects
        it or cancels it.
        """
        key = (weak_job.op_id, weak_job.window_id)
        assignment = self.shape.plan(weak_job)
        self.selections.remember_sibling(key, assignment.request_key)
        send_input = self._strong_input_send(assignment.job, None)
        return message.Submission(assignment.job, send_input)

    def escalate(self, weak_job: message.DecodeJob) -> None:
        """Ask the strong tier to re-decode the weak job's window.

        The selection rides weak_decoder_to_strong_decoder and the
        decoder side awaits the request's result. A strong sibling
        started with the weak job is selected as it is; otherwise the
        shape assigns the strong window: a job built now is queued
        behind its selection, a held one leaves when its condition fires
        (a terminal window whose tail is already stored leaves now).
        """
        key = (weak_job.op_id, weak_job.window_id)
        sibling_request_key = self.selections.sibling_for(key)
        if sibling_request_key is not None:
            self._send_selection(weak_job, sibling_request_key)
            self.decode_queue.await_strong_result(key, sibling_request_key)
            return
        assignment = self.shape.plan(weak_job)
        request_key = assignment.request_key
        selection_arrival_ticks = self._send_selection(weak_job, request_key)
        if assignment.job is not None:
            self._enqueue(assignment.job, selection_arrival_ticks)
        else:
            self.shape.note_selection_sent(key, selection_arrival_ticks)
            self.submit_if_terminal_data_complete(weak_job.op_id)
        self.decode_queue.await_strong_result(key, request_key)

    # ---- the hooks that release a held job

    def submit_if_far_boundary_committed(self, window_key: tuple) -> None:
        """A weak window committed: the strong window it bounds leaves now.

        The committed window's own strong sibling, selected or cancelled
        by its verdict, is forgotten.
        """
        self.selections.forget_sibling(window_key)
        deferred = self.shape.take_if_far_boundary_committed(window_key)
        if deferred is None:
            return
        self._enqueue(deferred.job, deferred.selection_arrival_ticks)
        self.engine.log(
            LOG_SOURCE,
            f"{deferred.job.label}: far-side weak boundary determined -> "
            "strong window submitted",
        )

    def submit_if_terminal_data_complete(self, operation_id) -> None:
        """A round is stored: a terminal strong window with its tail leaves."""
        deferred = self.shape.take_if_terminal_data_stored(operation_id)
        if deferred is None:
            return
        self._enqueue(deferred.job, deferred.selection_arrival_ticks)
        self.engine.log(
            LOG_SOURCE,
            f"{deferred.job.label}: terminal data complete -> "
            "strong window submitted",
        )

    # ---- observation

    def has_pending(self) -> bool:
        """Whether a strong window is still held for its condition."""
        return self.shape.has_pending()

    def pending_work(self) -> tuple:
        """The held strong windows as (key, phase, rounds)."""
        return self.shape.pending_work()

    # ---- private: submitting a strong job with its input send

    def _enqueue(
        self, strong_job: message.DecodeJob, selection_arrival_ticks: int
    ) -> None:
        """Queue the strong job; its input is sent at dispatch."""
        send_input = self._strong_input_send(
            strong_job, selection_arrival_ticks
        )
        self.decode_queue.enqueue(
            strong_job, send_input, self.on_strong_decoded
        )

    def _strong_input_send(
        self,
        strong_job: message.DecodeJob,
        selection_arrival_ticks: Optional[int],
    ) -> Callable[[Callable[[], None]], int]:
        """The job's input send, its payload bits fixed now.

        The staging clears the job's payloads when the input lands.
        """
        payload_bits = strong_job.payload_bits()
        return functools.partial(
            self._send_strong_input,
            strong_job,
            payload_bits,
            selection_arrival_ticks,
        )

    def _send_strong_input(
        self,
        strong_job: message.DecodeJob,
        payload_bits: Optional[int],
        selection_arrival_ticks: Optional[int],
        on_landed: Callable[[], None],
    ) -> int:
        """Send the input at dispatch; returns the delay the pool expects.

        The unit is assigned first, then the input moves into that unit's
        memory over strong_buffer_to_strong_decoder; a job selected at
        the verdict also waits for its selection to arrive, and the
        pool's estimate is the later of the two.
        """
        landed = on_landed
        if selection_arrival_ticks is not None:
            landed = functools.partial(
                self.selections.land_after_selection,
                strong_job.request_key,
                on_landed,
            )
        expected_delay_ticks = self.transfers.send_for_job(
            message.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
            strong_job,
            payload_bits=payload_bits,
            on_delivered=landed,
        )
        if selection_arrival_ticks is None:
            return expected_delay_ticks
        selection_delay_ticks = selection_arrival_ticks - self.engine.now
        return max(expected_delay_ticks, selection_delay_ticks)

    # ---- private: the selection

    def _send_selection(
        self,
        weak_job: message.DecodeJob,
        strong_request_key: message.DecoderRequestKey,
    ) -> int:
        """Send the window's selection; returns the tick it is expected.

        At the delivery a landed input waiting for it may start, and the
        decoder side accepts the selection.
        """
        key = (weak_job.op_id, weak_job.window_id)
        on_selection_delivered = functools.partial(
            self.decode_queue.accept_selection, key, strong_request_key
        )
        delivered = functools.partial(
            self._selection_delivered,
            strong_request_key,
            on_selection_delivered,
        )
        expected_delay_ticks = self.transfers.send_selection(
            weak_job, strong_request_key, delivered
        )
        return self.engine.now + expected_delay_ticks

    def _selection_delivered(
        self,
        request_key: message.DecoderRequestKey,
        on_selection_delivered: Callable[[], None],
    ) -> None:
        self.selections.note_delivered(request_key)
        on_selection_delivered()


class _StrongSelections:
    """The selection handshake's state.

    Which strong sibling each window may select (the paper's Step 1),
    which selections have arrived over weak_decoder_to_strong_decoder,
    and which landed strong inputs wait for one that has not.
    """

    def __init__(self) -> None:
        self.sibling_key_by_window: dict = {}
        self.delivered_request_keys: set = set()
        self.landing_by_request_key: dict = {}

    def remember_sibling(
        self, window_key: tuple, request_key: message.DecoderRequestKey
    ) -> None:
        self.sibling_key_by_window[window_key] = request_key

    def sibling_for(
        self, window_key: tuple
    ) -> Optional[message.DecoderRequestKey]:
        return self.sibling_key_by_window.get(window_key)

    def forget_sibling(self, window_key: tuple) -> None:
        self.sibling_key_by_window.pop(window_key, None)

    def land_after_selection(
        self,
        request_key: message.DecoderRequestKey,
        on_landed: Callable[[], None],
    ) -> None:
        """A selected strong input lands only once its selection arrived."""
        if request_key in self.delivered_request_keys:
            on_landed()
            return
        self.landing_by_request_key[request_key] = on_landed

    def note_delivered(self, request_key: message.DecoderRequestKey) -> None:
        """The selection arrived: a landed input waiting for it may start."""
        self.delivered_request_keys.add(request_key)
        waiting = self.landing_by_request_key.pop(request_key, None)
        if waiting is not None:
            waiting()
