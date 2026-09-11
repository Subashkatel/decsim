"""The strong re-decode: the window side of the strong tier.

When the escalation policy escalates a weak result, StrongRedecode asks
the shape for the strong window's job (strong_window_shapes.py), sends
the window's selection over weak_decoder_to_strong_decoder, tells the
decoder side to await the request's result (the DecodeQueue port), and
submits the job now or when the conditions the shape declared fire: the
commits of the weak windows it named, or the stored rounds of an
operation (pending_strong_windows.py). The redecode holds that index,
so a new shape row names its own condition rather than adding a hook.
When the policy
decodes both tiers at once (Toshio et al. 2510.25222 Sec. III A, Step
1), it builds the strong sibling the requester enqueues beside the weak
job and selects it at the verdict. A strong input landed in its unit
waits for its selection to arrive before it decodes; the submission
uses the per-job law, decode_queue.enqueue(job, send_input, on_decoded)
with the committer's accept_strong_result as the return path.
"""

import dataclasses
import functools
from typing import Callable, Optional

import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.windows as window_records
import decsim.trace_source as trace_source


class StrongRedecode:
    """Selects, submits and lands the strong tier's re-decode of a window."""

    def __init__(
        self,
        engine,
        shape,
        decoder_output,
        strong_output,
        decode_queue,
        on_strong_decoded: Callable[
            [decoding_records.DecodeJob, decoding_records.DecodeResult], None
        ],
    ) -> None:
        self.engine = engine
        self.shape = shape
        # the two ends that execute this tier's sends: the weak
        # decoder's selection leaves by the decoder output, and syndrome
        # buffer 1 sends the strong input
        self.decoder_output = decoder_output
        self.strong_output = strong_output
        self.decode_queue = decode_queue
        # the strong job's return path, the committer's accept_strong_result
        self.on_strong_decoded = on_strong_decoded
        self.selections = _StrongSelections()
        self.pending = pending_strong_windows.PendingStrongWindows()
        # trace sources: the wait a held strong window sits in, from the
        # hold to the condition that ends it
        self.trace = _TraceSources()

    # ---- the two entries

    def parallel_strong_submission(
        self, weak_job: decoding_records.DecodeJob
    ) -> Optional[decoding_records.Submission]:
        """The strong sibling started with the weak job (the paper's Step 1).

        Its context is held and its send built now; the verdict selects
        it or cancels it. A sibling whose input has not landed yet is
        held instead and enqueued when its condition fires, so there is
        no submission for the requester to make: Step 1 feeds both
        decoders the same data (2510.25222 lines 598-601), and in a
        model that prices transport the strong decoder starts when its
        copy has arrived.
        """
        key = (weak_job.operation_id, weak_job.window_id)
        assignment = self.shape.plan(weak_job)
        self.selections.remember_sibling(key, assignment.request_key)
        if assignment.job is None:
            self._hold(key, assignment, None)
            self.submit_if_stored_data_releases(weak_job.operation_id)
            return None
        send_input = self._strong_input_send(assignment.job, None)
        return decoding_records.Submission(assignment.job, send_input)

    def escalate(self, weak_job: decoding_records.DecodeJob) -> None:
        """Ask the strong tier to re-decode the weak job's window.

        The selection rides weak_decoder_to_strong_decoder and the
        decoder side awaits the request's result. A strong sibling
        started with the weak job is selected as it is; otherwise the
        shape assigns the strong window: a job built now is queued
        behind its selection, a held one is registered under the
        conditions it declared and leaves when they fire (a window
        waiting on data already stored leaves now).
        """
        key = (weak_job.operation_id, weak_job.window_id)
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
            self._hold(key, assignment, selection_arrival_ticks)
            self.submit_if_stored_data_releases(weak_job.operation_id)
        self.decode_queue.await_strong_result(key, request_key)

    # ---- the two events that meet a declared condition

    def submit_if_commit_releases(self, window_key: tuple) -> None:
        """A weak window committed: a strong window waiting on it leaves.

        The committed window's own strong sibling, selected or cancelled
        by its verdict, is forgotten.
        """
        self.selections.forget_sibling(window_key)
        released = self.pending.released_by_commit(window_key)
        self._submit_released(released)

    def submit_if_stored_data_releases(self, operation_id) -> None:
        """A round is stored: a strong window waiting on that data leaves."""
        released = self.pending.released_by_stored_data(operation_id)
        self._submit_released(released)

    def cancel_held_sibling(self, window_key: tuple) -> None:
        """A kept weak result: its held sibling never decodes.

        The parallel sibling is speculative (Toshio et al. 2510.25222
        Sec. III A, Step 1), so a confident weak result ends it. One
        already enqueued is cancelled on the decoder side; one still
        held for its input is dropped here, before the window's final
        commit frees the rounds it would have read.
        """
        held = self.pending.held_for(window_key)
        if held is None:
            return
        self.pending.take(held)
        self.selections.forget_sibling(window_key)
        self.trace.strong_window_left.fire(
            held.assignment.request_key,
            window_key,
            "cancelled: the weak result is confident",
            False,
        )
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"strong sibling for {window_key} cancelled while held for "
            f"its input (the weak result is confident)",
        )

    # ---- observation

    def has_pending(self) -> bool:
        """Whether a strong window is still held for its conditions."""
        return self.pending.any_held()

    def pending_work(self) -> tuple:
        """The held strong windows as (key, wait name, rounds)."""
        return self.pending.work()

    # ---- private: holding and releasing an assignment

    def _hold(
        self,
        key: tuple,
        assignment,
        selection_arrival_ticks: Optional[int],
    ) -> None:
        """Register the assignment under the conditions its row declared."""
        conditions = self.shape.release_conditions(assignment)
        held = pending_strong_windows.PendingStrongWindow(
            key=key,
            assignment=assignment,
            conditions=conditions,
            selection_arrival_ticks=selection_arrival_ticks,
        )
        self.pending.register(held)
        waits_for = pending_strong_windows.waiting_text(conditions)
        self.trace.strong_window_held.fire(
            assignment.request_key, key, waits_for, assignment.round_count
        )

    def _submit_released(self, released: tuple) -> None:
        """Ask each released row for its job; submit the ones it builds."""
        for held in released:
            job = self.shape.held_job(held.assignment)
            if job is None:
                continue
            self.pending.take(held)
            self.trace.strong_window_left.fire(
                job.request_key,
                held.key,
                held.conditions.released_description,
                True,
            )
            self._enqueue(job, held.selection_arrival_ticks)
            self.engine.log(
                log_sources.DECODER_MANAGER,
                f"{job.label}: {held.conditions.released_description} -> "
                "strong window submitted",
            )

    # ---- private: submitting a strong job with its input send

    def _enqueue(
        self,
        strong_job: decoding_records.DecodeJob,
        selection_arrival_ticks: int,
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
        strong_job: decoding_records.DecodeJob,
        selection_arrival_ticks: Optional[int],
    ) -> Callable[[Callable[[], None]], int]:
        """The send the manager calls at dispatch, bound to this job."""
        return functools.partial(
            self._send_strong_input, strong_job, selection_arrival_ticks
        )

    def _send_strong_input(
        self,
        strong_job: decoding_records.DecodeJob,
        selection_arrival_ticks: Optional[int],
        on_landed: Callable[[], None],
    ) -> int:
        """Send the input at dispatch; returns the delay the pool expects.

        The unit is assigned first, then syndrome buffer 1 moves the
        input into that unit's memory; a job selected at
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
        expected_delay_ticks = self.strong_output.send_input(strong_job, landed)
        if selection_arrival_ticks is None:
            return expected_delay_ticks
        selection_delay_ticks = selection_arrival_ticks - self.engine.now
        return max(expected_delay_ticks, selection_delay_ticks)

    # ---- private: the selection

    def _send_selection(
        self,
        weak_job: decoding_records.DecodeJob,
        strong_request_key: window_records.DecoderRequestKey,
    ) -> int:
        """Send the window's selection; returns the tick it is expected.

        At the delivery a landed input waiting for it may start, and the
        decoder side accepts the selection.
        """
        key = (weak_job.operation_id, weak_job.window_id)
        on_selection_delivered = functools.partial(
            self.decode_queue.accept_selection, key, strong_request_key
        )
        delivered = functools.partial(
            self._selection_delivered,
            strong_request_key,
            on_selection_delivered,
        )
        expected_delay_ticks = self.decoder_output.send_selection(
            weak_job, strong_request_key, delivered
        )
        return self.engine.now + expected_delay_ticks

    def _selection_delivered(
        self,
        request_key: window_records.DecoderRequestKey,
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
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        self.sibling_key_by_window[window_key] = request_key

    def sibling_for(
        self, window_key: tuple
    ) -> Optional[window_records.DecoderRequestKey]:
        return self.sibling_key_by_window.get(window_key)

    def forget_sibling(self, window_key: tuple) -> None:
        self.sibling_key_by_window.pop(window_key, None)

    def land_after_selection(
        self,
        request_key: window_records.DecoderRequestKey,
        on_landed: Callable[[], None],
    ) -> None:
        """A selected strong input lands only once its selection arrived."""
        if request_key in self.delivered_request_keys:
            on_landed()
            return
        self.landing_by_request_key[request_key] = on_landed

    def note_delivered(
        self, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection arrived: a landed input waiting for it may start."""
        self.delivered_request_keys.add(request_key)
        waiting = self.landing_by_request_key.pop(request_key, None)
        if waiting is not None:
            waiting()


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the strong redecode reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    strong_window_held: trace_source.TraceSource = trace_source.new_source()
    strong_window_left: trace_source.TraceSource = trace_source.new_source()
