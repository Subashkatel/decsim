"""The window commits: a result rides home and commits its window once.

The boundary is decoder state (the residual defects at the commit edge),
so it leaves for the dependent windows at decode done, the way Skoric's
blocks pass their artificial defects on (2209.08552 lines 275-278),
LILLIPUT's state register and qLDPC's net_error do
(qldpc/decoders/sinter.py decode_shots_to_error); the frame commit
downstream never gates the next window. The publisher rides the result
over its tier's output link and charges the frame's write; the
committer is every window job's on_decoded.
"""

import functools
from typing import Callable

import decsim.message as message
import decsim.observe.trace_source as trace_source
import decsim.records.windows as window_records
import decsim.windows.window_transfers as window_transfers


class CorrectionPublisher:
    """Rides a result over its output link to the frame, then calls back."""

    def __init__(self, transfers, frame) -> None:
        self.transfers = transfers
        self.frame = frame

    def publish(
        self,
        window: window_records.Window,
        operation: message.Operation,
        result: message.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Send the result on its tier's output link; commit it at delivery.

        A weak result rides weak_decoder_to_frame, a strong one
        strong_decoder_to_frame; the frame's priced write, when the run
        has a frame, gates on_committed.
        """
        output_path = message.LinkPath.WEAK_DECODER_TO_FRAME
        if request_key.tier is not window_records.DecoderTier.WEAK:
            output_path = message.LinkPath.STRONG_DECODER_TO_FRAME
        payload_bits = window_transfers.result_payload_bits(result, operation)
        commit = functools.partial(
            self._commit, window.key, result, request_key, on_committed
        )
        self.transfers.send_for_window(
            output_path, window, operation, request_key, payload_bits, commit
        )

    def _commit(
        self,
        window_key: tuple,
        result: message.DecodeResult,
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


class WindowCommitter:
    """Commits one window's result once; the job's on_decoded.

    Seven attributes: the engine stamps and logs, the planner and the
    tracker name the window and its operation, and the courier, the
    publisher, the strong redecode and the results are the four the
    commit hands to; the strong redecode (None when the run never
    escalates) hears an escalated result before its provisional commit
    and every weak commit after it. Trace source: window_committed(
    window, contribution), the window's record and the contribution
    that owns its rounds, at every commit.
    """

    def __init__(
        self,
        engine,
        planner,
        tracker,
        courier,
        publisher: CorrectionPublisher,
        strong_redecode,
        results,
    ) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.courier = courier
        self.publisher = publisher
        self.strong_redecode = strong_redecode
        self.results = results
        self.window_committed = trace_source.TraceSource()

    def accept_result(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """A primary decode finished: hand the boundary on, publish, commit.

        A result the policy escalated asks the strong tier for the
        window first (its selection, and a serial strong job, leave
        now), then commits provisionally, its boundary leaving with the
        commit.
        """
        key = (job.op_id, job.window_id)
        window = self.planner.windows_by_key[key]
        window.t_done = self.engine.now
        operation = self.tracker.operation_by_id[job.op_id]
        is_final = not job.awaiting_strong_result
        if not is_final:
            self.strong_redecode.escalate(job)
            self.commit(window, operation, result, job.request_key, False)
            return
        self.courier.hand_on(window, operation, result, job.request_key, True)
        commit = functools.partial(
            self.commit, window, operation, result, job.request_key, True
        )
        self.publisher.publish(
            window, operation, result, job.request_key, commit
        )

    def accept_strong_result(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """A strong decode finished: publish it, then finalize the window."""
        key = (job.request_key.operation_id, job.request_key.window_id)
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[window.op_id]
        finish = functools.partial(
            self.finish_strong, window, operation, result, job.request_key
        )
        self.publisher.publish(
            window, operation, result, job.request_key, finish
        )

    def commit(
        self,
        window: window_records.Window,
        operation: message.Operation,
        result: message.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        is_final: bool,
    ) -> None:
        """Commit the window: its contribution, its status, what it wakes."""
        window.committed = True
        if window.t_done is None:
            window.t_done = self.engine.now
        status = result.decode_status
        window.decode_status = None
        status_note = ""
        if status is not None:
            window.decode_status = status.value
            status_note = f" best effort: {status.value}"
        self.engine.log(
            "DecoderCluster",
            f"DECODE DONE {operation.name} W{window.k} "
            f"[commit {window.commit_lo}-{window.commit_hi}]{status_note}",
        )
        contribution = self.results.install_window_contribution(
            window, result.logical_observables
        )
        if is_final:
            window.published_request_key = request_key
        self.window_committed.fire(window, contribution)
        self.results.note_window_committed(window, is_final)
        if not is_final:
            # provisional: the boundary leaves with the commit
            self.courier.hand_on(window, operation, result, request_key, False)
        if self.strong_redecode is not None:
            self.strong_redecode.submit_if_far_boundary_committed(window.key)
        self.results.deliver_if_final(operation)

    def finish_strong(
        self,
        window: window_records.Window,
        operation: message.Operation,
        result: message.DecodeResult,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """The strong result is the window's final one.

        Its prediction replaces the provisional one, the held boundary
        ships now that it is final, and the operation may complete.
        """
        if result.logical_observables is not None:
            self.results.replace_prediction(
                window.key, result.logical_observables
            )
        # nothing of the operation waits on the window any more
        window.published_request_key = request_key
        self.courier.ship_held(window, result, request_key)
        self.results.release_committed_segments(operation.id)
        self.results.deliver_if_final(operation)
