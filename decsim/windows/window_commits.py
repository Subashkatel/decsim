"""The window commits: a result rides home and commits its window once.

The boundary is decoder state (the residual defects at the commit edge),
so it leaves for the dependent windows at decode done, the way Skoric's
blocks pass their artificial defects on (2209.08552 lines 275-278),
LILLIPUT's state register and qLDPC's net_error do
(qldpc/decoders/sinter.py decode_shots_to_error); the frame commit
downstream never gates the next window. The committer is every window
job's on_decoded: it decides, and the decoder side executes the send
that carries the correction to the frame (decoders/decoder_output.py).
"""

import functools

import decsim.decoders.decode_queue as decode_queue_module
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records


class WindowCommitter:
    """Commits one window's result once; the job's on_decoded.

    The engine stamps and logs, the planner and the tracker name the
    window and its operation, the escalation policy answers the
    window's confidence, the decode queue hears that answer, and the
    courier, the decoder output, the strong redecode and the results are
    the four the commit hands to; the strong redecode (None when the run
    never escalates) hears an escalated result before its provisional
    commit and every weak commit after it. Trace source:
    window_committed(window, contribution), the window's record and the
    contribution that owns its rounds, at every commit.
    """

    def __init__(
        self,
        engine,
        planner,
        tracker,
        courier,
        decoder_output,
        strong_redecode,
        results,
        escalation_policy,
        decode_queue,
    ) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.courier = courier
        self.decoder_output = decoder_output
        self.strong_redecode = strong_redecode
        self.results = results
        self.escalation_policy = escalation_policy
        self.decode_queue = decode_queue
        self.window_committed = trace_source.TraceSource()

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
        window.t_done = self.engine.now
        operation = self.tracker.operation_by_id[job.operation_id]
        verdict = self.escalation_policy.verdict_for_weak_result(job, result)
        is_final = verdict is decoding_records.Verdict.KEEP
        if not is_final:
            self.strong_redecode.escalate(job)
        self.decode_queue.resolve_weak_request(job, result, verdict)
        if not is_final:
            self.commit(window, operation, result, job.request_key, False)
            return
        self.courier.hand_on(window, operation, result, job.request_key, True)
        commit = functools.partial(
            self.commit, window, operation, result, job.request_key, True
        )
        self.decoder_output.publish(
            window, operation, result, job.request_key, commit
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
        finish = functools.partial(
            self.finish_strong, window, operation, result, job.request_key
        )
        self.decoder_output.publish(
            window, operation, result, job.request_key, finish
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
        self.window_committed.fire(window, contribution)
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
