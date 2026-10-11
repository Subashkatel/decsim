"""The window commits: a result rides home and commits its window once.

The boundary is decoder state, the residual defects at the commit edge,
so it leaves for the dependent windows at the verdict, as Skoric's
blocks pass their artificial defects on (2209.08552 lines 268-275) and
qLDPC its net_error (qldpc/decoders/sinter.py decode_shots_to_error).
The frame commit downstream never gates the next window.

Every decode of a window is a revision: the first is the window pool's,
and each escalate edge the window crosses adds one on its target pool.
WindowVerdict is every job's on_decoded: a revision whose pool an edge
leaves meets that edge's verdict, which escalates or cancels, and tells
the pool's decode queue; a revision no edge leaves is final.
WindowCommitter then commits the revision and hands the correction on;
the decoder side sends the final one to the frame
(decoders/decoder_output.py), which takes one write per window.
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
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.operation_results as operation_results
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_planner as window_planner


@dataclasses.dataclass(frozen=True)
class VerdictCosts:
    """An edge's verdict, priced on its own clock.

    threshold_cycles prices the threshold on a result with a soft
    output; switch_cycles prices the switch on an escalation. Zero
    costs keep the handoff synchronous.
    """

    clock: Optional[config.Clock] = None
    threshold_cycles: int = 0
    switch_cycles: int = 0


@dataclasses.dataclass(frozen=True)
class Revision:
    """One decode of a window: the first, or one along an escalate edge.

    is_first is the window pool's decode, which commits the window; a
    later revision replaces what the earlier ones committed.
    """

    window: window_records.Window
    operation: program_records.Operation
    result: decoding_records.DecodeResult
    request_key: window_records.DecoderRequestKey
    is_first: bool


class WindowCommitter:
    """Commits each window once, then each later revision over it.

    Every escalate edge hears every boundary as the courier takes it,
    because a held strong window waits for the boundaries the weak
    decoder determined (Toshio et al. 2510.25222 lines 1248-1250). The
    caller's on_result_read runs at the first revision's commit, where
    the window side has the result; a target pool never blocks its unit
    on a result (build/decoders.py _blocks_unit). Trace source:
    window_committed(window, contribution) at every commit.
    """

    courier = ports.Port(window_boundaries.BoundaryCourier)
    decoder_output = ports.Port(ports.DecoderOutput)
    results = ports.Port(operation_results.OperationResults)
    edges = ports.Port(escalate_edges.EscalateEdges)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.trace = _TraceSources()

    def commit_revision(
        self,
        revision: Revision,
        is_final: bool,
        on_result_read: Callable[[], None],
    ) -> None:
        """Commit the first revision, or replace the window's with a later one.

        A later final revision goes home and finalizes the window; a
        later provisional one replaces the prediction and waits for the
        next edge's answer.
        """
        window = revision.window
        operation = revision.operation
        result = revision.result
        request_key = revision.request_key
        if revision.is_first:
            self.commit_or_publish(
                window, operation, result, request_key, is_final, on_result_read
            )
            return
        if is_final:
            self.publish_strong(window, operation, result, request_key)
            return
        self.replace_provisionally(window, result)

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

        The commit is where the decoder side hears that its output was
        read (<tier>.result_blocks_unit, decoders/settings.py).
        """
        if not is_final:
            self.commit(window, operation, result, request_key, False)
            on_result_read()
            return
        self.courier.hand_on(window, operation, result, request_key, True)
        self._release_held_strong_windows(window.key)
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
        prediction receive the same bits. The answer's route prices
        where that join happens (records/routes.py
        ANSWER_ROUTE_NAMES). A boundary held for the window leaves
        now, from the decoder that computed it, as a final weak result's
        does (commit_or_publish), and not after the answer reaches the
        frame.
        """
        result = _with_the_crossing_commit(window, result)
        self.courier.ship_held(window, result, request_key)
        finish = functools.partial(
            self.finish_strong, window, operation, result, request_key
        )
        self.decoder_output.publish(
            window, operation, result, request_key, finish
        )

    def replace_provisionally(
        self,
        window: window_records.Window,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A later revision that escalates again replaces the prediction.

        The window stays awaiting its final revision: nothing goes to
        the frame and no boundary leaves.
        """
        result = _with_the_crossing_commit(window, result)
        _record_the_decode(window, result)
        if result.logical_observables is None:
            return
        self.results.replace_prediction(window.key, result.logical_observables)

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
            self._release_held_strong_windows(window.key)
        self.results.deliver_if_final(operation)

    def finish_strong(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """The strong result, landed, is the window's final one.

        Its prediction, status and no-correction reason replace the
        provisional ones.
        """
        _record_the_decode(window, result)
        if result.logical_observables is not None:
            self.results.replace_prediction(
                window.key, result.logical_observables
            )
        # nothing of the operation waits on the window any more
        window.published_request_key = request_key
        self.results.release_committed_segments(operation.id)
        self.results.deliver_if_final(operation)

    def _release_held_strong_windows(self, window_key: tuple) -> None:
        """The weak boundary is handed on: a strong window held on it leaves."""
        for edge in self.edges.edges:
            edge.submit_if_commit_releases(window_key)


class WindowVerdict:
    """Gives each revision its verdict; every job's on_decoded.

    A revision of a pool an escalate edge leaves meets that edge's
    policy, priced on the edge's own clock, like gem5's frontend and
    forward latencies (src/mem/XBar.py); zero costs keep the handoff
    synchronous. A revision no edge leaves is final. window_pool
    decodes the planner's windows, so its revision is the first.
    """

    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker_module.RoundTracker)
    # the window pool's manager; an edge's target pool is its edge's
    decode_queue = ports.Port(ports.DecodeQueue)
    committer = ports.Port(WindowCommitter)
    edges = ports.Port(escalate_edges.EscalateEdges)

    def __init__(
        self,
        engine: engine_module.Engine,
        window_pool: window_records.PoolId = window_records.WEAK_POOL_ID,
    ) -> None:
        self.engine = engine
        self.window_pool = window_pool

    def accept_revision(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A decode of the window finished: its verdict, or its final commit.

        Toshio et al. 2510.25222 Sec. III A, step 3: a kept result rides
        its output link to the frame and commits as final; a result
        below the threshold asks the next pool for the window after the
        switch cost, then commits provisionally, its boundary leaving
        with the first commit. A strong result no edge leaves is final.
        """
        pool = job.request_key.pool
        edge = self.edges.leaving(pool)
        is_first = pool == self.window_pool
        if not is_first and edge is None:
            self._commit_final_revision(job, result)
            return
        if is_first:
            window = self._window_of(job)
            self.committer.note_decode_finished(window)
        self._price_the_threshold(job, result, edge)

    def _price_the_threshold(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        edge: Optional[ports.EscalateEdge],
    ) -> None:
        if edge is None or result.soft_output is None:
            self._form_verdict(job, result, edge)
            return
        costs = edge.costs
        if costs.threshold_cycles == 0:
            self._form_verdict(job, result, edge)
            return
        tick = costs.clock.edge(costs.threshold_cycles, self.engine.now)
        delay = tick - self.engine.now
        decide = functools.partial(self._form_verdict, job, result, edge)
        self.engine.schedule(delay, decide, label="escalation threshold")

    def _commit_final_revision(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A later revision no edge leaves: publish it, finalize the window."""
        revision = self._revision(job, result)
        self.committer.commit_revision(revision, True, _nothing_to_read)

    def _form_verdict(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        edge: Optional[ports.EscalateEdge],
    ) -> None:
        verdict = _verdict_for(job, result, edge)
        if verdict is decoding_records.Verdict.KEEP:
            self._apply_verdict(job, result, verdict, edge)
            return
        costs = edge.costs
        if costs.switch_cycles == 0:
            self._apply_verdict(job, result, verdict, edge)
            return
        tick = costs.clock.edge(costs.switch_cycles, self.engine.now)
        delay = tick - self.engine.now
        apply = functools.partial(
            self._apply_verdict, job, result, verdict, edge
        )
        self.engine.schedule(delay, apply, label="escalation switch")

    def _apply_verdict(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
        edge: Optional[ports.EscalateEdge],
    ) -> None:
        revision = self._revision(job, result)
        is_final = verdict is decoding_records.Verdict.KEEP
        if not is_final:
            edge.escalate(job)
        elif edge is not None:
            edge.cancel_strong_request(revision.window.key)
        queue = self._queue_of(job.request_key.pool)
        queue.resolve_weak_request(job, result, verdict)
        read_result = functools.partial(queue.read_result, job)
        self.committer.commit_revision(revision, is_final, read_result)

    def _revision(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> Revision:
        window = self._window_of(job)
        operation = self.tracker.operation_by_id[window.operation_id]
        request_key = job.request_key
        is_first = request_key.pool == self.window_pool
        return Revision(window, operation, result, request_key, is_first)

    def _window_of(
        self, job: decoding_records.DecodeJob
    ) -> window_records.Window:
        request_key = job.request_key
        key = (request_key.operation_id, request_key.window_id)
        return self.planner.windows_by_key[key]

    def _queue_of(self, pool: window_records.PoolId) -> ports.DecodeQueue:
        """The manager that served a revision of pool."""
        if pool == self.window_pool:
            return self.decode_queue
        edge = self.edges.entering(pool)
        return edge.decode_queue


def _verdict_for(
    job: decoding_records.DecodeJob,
    result: decoding_records.DecodeResult,
    edge: Optional[ports.EscalateEdge],
) -> decoding_records.Verdict:
    """The edge's verdict; a revision no edge leaves is kept."""
    if edge is None:
        return decoding_records.Verdict.KEEP
    return edge.policy.verdict_for_weak_result(job, result)


def _nothing_to_read() -> None:
    """A target pool's result left its unit when its decode ended."""


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
    """Every event the window committer reports, as one member."""

    window_committed: trace_source.TraceSource = trace_source.new_source()
