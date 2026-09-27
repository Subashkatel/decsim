"""The window commits: a result rides home and commits its window once.

The boundary is decoder state (the residual defects at the commit edge),
so it leaves for dependent windows once the verdict is ready, as Skoric's
blocks pass their artificial defects on: the corrections crossing out of
the commit region become artificial defects (2209.08552 lines 268-269),
and the next window decodes them with the buffer region's unresolved
defects and the new rounds (lines 272-275). LILLIPUT's state register
and qLDPC's net_error do the same
(qldpc/decoders/sinter.py decode_shots_to_error); the frame commit
downstream never gates the next window.

Two classes, because a window's result meets two decisions. WindowVerdict
is every window job's on_decoded: it applies the threshold, escalates or
cancels, and tells the decode queue what it decided. WindowCommitter
then commits the window and hands the correction on; the decoder side
executes the send that carries it to the frame
(decoders/decoder_output.py). The committer is written first, because
the verdict declares it as a port and a port carries the class it names.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
import decsim.windows.operation_results as operation_results
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_planner as window_planner


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

    courier = ports.Port(window_boundaries.BoundaryCourier)
    decoder_output = ports.Port(ports.DecoderOutput)
    results = ports.Port(operation_results.OperationResults)
    # the strong tier's window side; None when the run never escalates
    strong_redecode = ports.Port(ports.StrongRedecode, optional=True)

    def __init__(self, engine) -> None:
        self.engine = engine
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
        """Send the strong result home; it finalizes the window on landing.

        What the weak decode committed of the faults behind the window
        joins the result before it leaves, so the frame and the
        prediction receive the same bits.
        """
        result = _with_the_crossing_commit(window, result)
        finish = functools.partial(
            self.finish_strong, window, operation, result, request_key
        )
        self.decoder_output.publish(
            window, operation, result, request_key, finish
        )

    def note_decode_finished(self, window: window_records.Window) -> None:
        """Stamp the window with the tick its decode answered."""
        window.t_done = self.engine.now

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
        window.crossing_commit = result.crossing_commit
        if window.t_done is None:
            self.note_decode_finished(window)
        _record_the_decode(window, result)
        status_note = ""
        if window.decode_status is not None:
            status_note = f" best effort: {window.decode_status}"
        self.engine.log(
            log_sources.DECODER_MANAGER,
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
            # provisional: the boundary leaves with the commit, and what
            # it fed forward outlives the strong answer
            window.provisional_no_correction_reason = (
                window.no_correction_reason
            )
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

        Its prediction, its status and its no-correction reason replace
        the provisional ones, the held boundary ships now that it is
        final, and the operation may complete.
        """
        _record_the_decode(window, result)
        if result.logical_observables is not None:
            self.results.replace_prediction(
                window.key, result.logical_observables
            )
        # nothing of the operation waits on the window any more
        window.published_request_key = request_key
        self.courier.ship_held(window, result, request_key)
        self.results.release_committed_segments(operation.id)
        self.results.deliver_if_final(operation)


class WindowVerdict:
    """Applies the threshold to one window's result; the job's on_decoded.

    The planner and the tracker name the window and its operation, the
    escalation policy answers the window's confidence, the decode queue
    hears that answer and, at the commit, that the result was read, the
    strong redecode (None when the run never escalates) hears an
    escalated result before its provisional commit and every weak commit
    after it, and the committer stamps the window with the tick the
    decode answered and performs whichever commit the verdict asks for.
    The engine and the cycle costs price this handoff's threshold and
    switch logic on its own clock, like gem5's frontend and forward
    latencies (src/mem/XBar.py). Zero costs keep the handoff synchronous.
    """

    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker_module.RoundTracker)
    escalation_policy = ports.Port(ports.EscalationPolicy)
    # the strong tier's window side; None when the run never escalates
    strong_redecode = ports.Port(ports.StrongRedecode, optional=True)
    decode_queue = ports.Port(ports.DecodeQueue)
    committer = ports.Port(WindowCommitter)

    def __init__(
        self,
        engine: engine_module.Engine,
        clock: Optional[config.Clock] = None,
        threshold_cycles: int = 0,
        switch_cycles: int = 0,
    ) -> None:
        self.engine = engine
        self.clock = clock
        self.threshold_cycles = threshold_cycles
        self.switch_cycles = switch_cycles

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
        the window first after the switch cost, then commits
        provisionally, its boundary leaving with the commit.
        """
        key = (job.operation_id, job.window_id)
        window = self.planner.windows_by_key[key]
        self.committer.note_decode_finished(window)
        cycles = self.threshold_cycles
        if cycles == 0 or result.soft_output is None:
            self._form_verdict(job, result)
            return
        if not self.escalation_policy.decides_on_a_confidence:
            self._form_verdict(job, result)
            return
        edge = self.clock.edge(cycles, self.engine.now)
        delay = edge - self.engine.now
        decide = functools.partial(self._form_verdict, job, result)
        self.engine.schedule(delay, decide, label="escalation threshold")

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

    def _form_verdict(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        verdict = self.escalation_policy.verdict_for_weak_result(job, result)
        if verdict is decoding_records.Verdict.KEEP or self.switch_cycles == 0:
            self._apply_verdict(job, result, verdict)
            return
        edge = self.clock.edge(self.switch_cycles, self.engine.now)
        delay = edge - self.engine.now
        apply = functools.partial(self._apply_verdict, job, result, verdict)
        self.engine.schedule(delay, apply, label="escalation switch")

    def _apply_verdict(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
    ) -> None:
        key = (job.operation_id, job.window_id)
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[job.operation_id]
        is_final = verdict is decoding_records.Verdict.KEEP
        if not is_final:
            self.strong_redecode.escalate(job)
        elif self.strong_redecode is not None:
            self.strong_redecode.cancel_strong_request(key)
        self.decode_queue.resolve_weak_request(job, result, verdict)
        read_result = functools.partial(self.decode_queue.read_result, job)
        self.committer.commit_or_publish(
            window, operation, result, job.request_key, is_final, read_result
        )


def _record_the_decode(
    window: window_records.Window, result: decoding_records.DecodeResult
) -> None:
    """The committed decode's status and no-correction reason, as values."""
    window.decode_status = _value_of(result.decode_status)
    window.no_correction_reason = _value_of(result.no_correction_reason)


def _value_of(member: Optional[enum.Enum]) -> Optional[str]:
    """An enum member's value, None for no member."""
    if member is None:
        return None
    return member.value


def _with_the_crossing_commit(
    window: window_records.Window, result: decoding_records.DecodeResult
) -> decoding_records.DecodeResult:
    """The strong result, plus the weak commit the region cannot own.

    A strong window owns no fault touching a round before its commit
    region, so what the weak decode committed of those faults stays: at
    a back-to-back seam that commit is the boundary condition the region
    before it and this one are both pinned on (Toshio et al. 2510.25222
    lines 1248-1250).
    """
    crossing = window.crossing_commit
    strong_flips = result.logical_observables
    if crossing is None or strong_flips is None:
        return result
    kept_flips = crossing.logical_observables
    prediction = tuple(
        strong_flip ^ kept_flip
        for strong_flip, kept_flip in zip(strong_flips, kept_flips, strict=True)
    )
    return dataclasses.replace(result, logical_observables=prediction)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the window committer reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    window_committed: trace_source.TraceSource = trace_source.new_source()
