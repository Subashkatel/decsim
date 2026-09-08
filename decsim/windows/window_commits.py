"""The window commits: a result rides home and commits its window once.

The boundary is decoder state (the residual defects at the commit edge),
so it leaves for the dependent windows at decode done, the way Skoric's
blocks pass their artificial defects on (2209.08552 lines 275-278),
LILLIPUT's state register and qLDPC's net_error do
(qldpc/decoders/sinter.py decode_shots_to_error); the frame commit
downstream never gates the next window.

Two classes, because a window's result meets two decisions. WindowVerdict
is every window job's on_decoded: it applies the threshold, escalates or
cancels, and tells the decode queue what it decided. WindowCommitter
then commits the window and hands the correction on; the decoder side
executes the send that carries it to the frame
(decoders/decoder_output.py).
"""

import dataclasses
import functools
from typing import Callable

import decsim.decoders.decode_queue as decode_queue_module
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records


class WindowVerdict:
    """Applies the threshold to one window's result; the job's on_decoded.

    The planner and the tracker name the window and its operation, the
    escalation policy answers the window's confidence, the decode queue
    hears that answer and, at the commit, that the result was read, the
    strong redecode (None when the run never escalates) hears an
    escalated result before its provisional commit and every weak commit
    after it, and the committer performs whichever commit the verdict
    asks for.
    """

    def __init__(
        self,
        planner,
        tracker,
        escalation_policy,
        strong_redecode,
        decode_queue,
        committer: "WindowCommitter",
    ) -> None:
        self.planner = planner
        self.tracker = tracker
        self.escalation_policy = escalation_policy
        self.strong_redecode = strong_redecode
        self.decode_queue = decode_queue
        self.committer = committer

    def accept_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """The window's answer: apply the threshold, publish or escalate.

        The result carries the window's confidence, so the threshold is
        applied here (Toshio et al. 2510.25222 Sec. III A, step 3): a
        kept result rides its output link to the frame and commits as
        final; a result below the threshold asks the strong tier for
        the window first (its selection, and a serial strong job, leave
        now), then commits provisionally, its boundary leaving with the
        commit.
        """
        key = (job.operation_id, job.window_id)
        window = self.planner.windows_by_key[key]
        window.t_done = self.committer.engine.now
        operation = self.tracker.operation_by_id[job.operation_id]
        verdict = self.escalation_policy.verdict_for_weak_result(job, result)
        is_final = verdict is decoding_records.Verdict.KEEP
        if not is_final:
            self.strong_redecode.escalate(job)
        elif self.strong_redecode is not None:
            self.strong_redecode.cancel_held_sibling(key)
        self.decode_queue.resolve_weak_request(job, result, verdict)
        read_result = functools.partial(self.decode_queue.read_result, job)
        self.committer.commit_or_publish(
            window, operation, result, job.request_key, is_final, read_result
        )

    def accept_strong_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A strong decode finished: publish it, then finalize the window."""
        key = (job.request_key.operation_id, job.request_key.window_id)
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[window.operation_id]
        self.committer.publish_strong(
            window, operation, result, job.request_key
        )


class WindowCommitter:
    """Commits one window once and hands its correction on.

    The engine stamps and logs; the courier, the decoder output and the
    results are the three the commit hands to; the strong redecode
    (None when the run never escalates) hears every weak commit, because
    a commit can release a held strong window. The commit is also where
    the result is in the window side's hands, so the caller's
    on_result_read runs there. Trace source: window_committed(window,
    contribution), the window's record and the contribution that owns
    its rounds, at every commit.
    """

    def __init__(
        self, engine, courier, decoder_output, results, strong_redecode
    ) -> None:
        self.engine = engine
        self.courier = courier
        self.decoder_output = decoder_output
        self.results = results
        self.strong_redecode = strong_redecode
        self.trace = _TraceSources()

    def commit_or_publish(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        is_final: bool,
        on_result_read: Callable[[], None],
    ) -> None:
        """A provisional result commits now; a final one commits on its send.

        The commit is the moment the window side has the result in hand,
        so it is where the decoder side hears that its output was read
        (<tier>.result_blocks_unit, decoders/settings.py).
        """
        if not is_final:
            self.commit(window, operation, result, request_key, False)
            on_result_read()
            return
        self.courier.hand_on(window, operation, result, request_key, True)
        commit = functools.partial(
            self._commit_the_final_result,
            window,
            operation,
            result,
            request_key,
            on_result_read,
        )
        self.decoder_output.publish(
            window, operation, result, request_key, commit
        )

    def _commit_the_final_result(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_result_read: Callable[[], None],
    ) -> None:
        """Commit the landed result, then report that it was read."""
        self.commit(window, operation, result, request_key, True)
        on_result_read()

    def publish_strong(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """Send the strong result home; it finalizes the window on landing."""
        finish = functools.partial(
            self.finish_strong, window, operation, result, request_key
        )
        self.decoder_output.publish(
            window, operation, result, request_key, finish
        )

    def commit(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
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
            decode_queue_module.LOG_SOURCE,
            f"DECODE DONE {operation.name} W{window.window_index} "
            f"[commit {window.commit_lo}-{window.commit_hi}]{status_note}",
        )
        contribution = self.results.install_window_contribution(
            window, result.logical_observables
        )
        if is_final:
            window.published_request_key = request_key
        self.trace.window_committed.fire(window, contribution)
        self.results.note_window_committed(window, is_final)
        if not is_final:
            # provisional: the boundary leaves with the commit
            self.courier.hand_on(window, operation, result, request_key, False)
        if self.strong_redecode is not None:
            self.strong_redecode.submit_if_commit_releases(window.key)
        self.results.deliver_if_final(operation)

    def finish_strong(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
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


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the window committer reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    window_committed: trace_source.TraceSource = trace_source.new_source()
