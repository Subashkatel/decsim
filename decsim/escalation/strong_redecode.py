"""The strong re-decode: the window side of one escalate edge.

One StrongRedecode serves each edge, from its source pool to its target
pool, with its own policy, verdict costs, shape and held windows; a
legacy machine has one, from the weak pool to the strong one. When the
policy escalates a source result, StrongRedecode asks the shape
for the strong window's job, sends the window's selection over
weak_decoder_to_strong_decoder, tells the decoder side to await the
result, and submits the job now or when the conditions the shape
declared fire. A held window whose rounds the strong syndrome buffer
lacks has them read out of the weak syndrome buffer and carried over the
same hop (Toshio et al. 2510.25222 lines 1247-1250), in one transfer or
several (CUDA-Q QEC's enqueue_syndromes, realtime_decoding.rst lines
52-56). When both tiers decode at once (Step 1), it builds the
speculative strong decode beside the weak job and selects it at the
verdict; a landed strong input waits for its selection before it
decodes.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Optional

import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.records.routes as route_records
import decsim.records.windows as window_records
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry
import decsim.trace_source as trace_source
import decsim.windows.revision_rules as revision_rules
import decsim.windows.round_retention as round_retention


class StrongRedecode:
    """Drives each window this edge escalates through its target's decode.

    hops names the edge: its source, the pool whose results meet its
    verdict; its target, the pool that re-decodes the windows it
    escalates; and the forward hop its selections and regions ride.
    rules says how the window side takes its verdicts and revisions,
    the verdict priced on its own clock (windows/revision_rules.py).
    """

    # the rule that keeps or escalates a source result
    policy = ports.Port(ports.EscalationPolicy)
    shape = ports.Port(strong_window_shapes.StrongWindowShape)
    # the ledger of held strong windows and what releases each
    pending = ports.Port(pending_strong_windows.PendingStrongWindows)
    # where the rounds a strong window reads sit on the chip
    retention = ports.Port(ports.WindowRetention)
    # the ends that execute this tier's sends: the weak decoder's selection
    # and the escalated region leave by the decoder output, and the strong
    # syndrome buffer sends the strong input
    decoder_output = ports.Port(ports.DecoderOutput)
    # the store a held window's missing rounds are read out of
    carried_from = ports.Port(ports.SyndromeBuffer)
    # the strong syndrome buffer's receiving end, where the region lands
    strong_receiver = ports.Port(ports.SyndromeRoundReceiver)
    strong_output = ports.Port(ports.SyndromeBufferOutput)
    # the target pool's manager
    decode_queue = ports.Port(ports.DecodeQueue)
    # the strong job's return path
    verdict = ports.Port(ports.WindowVerdict)
    # the join of the edge that leaves the target pool, which reads the
    # target's results for that edge's signal; none when no edge leaves
    target_join = ports.Port(ports.WindowGapJoin, optional=True)
    # the edge that leaves the target pool: it may start beside a target
    # job, and a keep here ends its work on the window too
    next_edge = ports.Port(ports.EscalateEdge, optional=True)

    # the windows escalated, those whose speculative strong decode was
    # selected as it ran, and the strong windows held for their
    # conditions and dropped held when the weak result was kept
    STATISTICS = (
        statistics_registry.Statistic(
            "escalated", statistic_kinds.Count, "windows escalated", "windows"
        ),
        statistics_registry.Statistic(
            "speculative_selected",
            statistic_kinds.Count,
            "escalations that took the running speculative decode",
            "windows",
        ),
        statistics_registry.Statistic(
            "held",
            statistic_kinds.Count,
            "strong windows held for their release conditions",
            "windows",
        ),
        statistics_registry.Statistic(
            "held_cancelled",
            statistic_kinds.Count,
            "held strong windows dropped for a kept weak result",
            "windows",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    def __init__(
        self,
        engine: engine_module.Engine,
        hops: route_records.EdgeHops,
        rules: revision_rules.EdgeRules = revision_rules.DEFAULT_RULES,
    ) -> None:
        self.engine = engine
        self.hops = hops
        self.rules = rules
        self.selections = _StrongSelections()
        # the rounds sent up and not landed yet, so a wake-up while a
        # region crosses does not carry them twice
        self.carried_round_keys: set = set()
        # trace sources: the wait a held strong window sits in, from the
        # hold to the condition that ends it
        self.trace = _TraceSources()

    @property
    def source(self) -> window_records.PoolId:
        """The pool whose results meet this edge's verdict."""
        return self.hops.source

    @property
    def target(self) -> window_records.PoolId:
        """The pool that re-decodes the windows this edge escalates."""
        return self.hops.target

    def accept_target_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A target decode ended: to the next edge's join, else the verdict.

        A target pool that is the source of an edge of its own reads its
        results for that edge's signal first, as the entry pool's solves
        reach the requester's join.
        """
        if self.target_join is None:
            self.verdict.accept_revision(job, result)
            return
        self.target_join.accept_result(job, result)

    def enqueue_target(
        self,
        job: decoding_records.DecodeJob,
        selection_arrival_ticks: Optional[int],
    ) -> None:
        """Queue a target job and its companions; the next edge may start.

        The edge out of the target starts beside the job itself, which
        names the attempt; its signal joins the companions' solves.
        """
        attempt = self._attempt_of(job)
        for attempt_job in attempt:
            send_input = self._strong_input_send(
                attempt_job, selection_arrival_ticks
            )
            self.decode_queue.enqueue(
                attempt_job, send_input, self.accept_target_result
            )
        if self.next_edge is None:
            return
        self.next_edge.start_beside_source(job)

    def start_beside_source(
        self, source_job: decoding_records.DecodeJob
    ) -> None:
        """A source job past the entry was queued: decode it here too.

        The speculative decode starts with the source pool's job (Toshio
        et al. 2510.25222 lines 598-601), one depth down. The source job
        was built with its faces pinned, so nothing defers it.
        """
        if not self.policy.decodes_both_at_once(source_job.window):
            return
        submission = self._planned_speculative_decode(source_job)
        if submission is None:
            return
        self.enqueue_target(submission.job, None)

    # ---- the two entries

    def parallel_strong_submission(
        self, weak_job: decoding_records.DecodeJob
    ) -> Optional[decoding_records.Submission]:
        """The speculative strong decode, started with the weak job (Step 1).

        Step 1 starts both decoders on the same window (2510.25222 lines
        598-601). A weak job whose window still owes a boundary waits in
        the decode queue until it lands, so its speculative decode is
        planned at that release (released_submission): a pinned face
        reads the neighbour's final commit, which is what releases the
        weak job. One whose input has not landed is held and enqueued
        when its condition fires, so there is no submission.
        """
        key = (weak_job.operation_id, weak_job.window_id)
        if weak_job.window.deps_remaining > 0:
            self.selections.defer_speculative_decode(key, weak_job)
            return None
        return self._planned_speculative_decode(weak_job)

    def released_submission(
        self, window_key: tuple
    ) -> Optional[decoding_records.Submission]:
        """The weak job's last boundary arrived: plan its speculative decode."""
        weak_job = self.selections.take_deferred_speculative_decode(window_key)
        if weak_job is None:
            return None
        return self._planned_speculative_decode(weak_job)

    def _planned_speculative_decode(
        self, weak_job: decoding_records.DecodeJob
    ) -> Optional[decoding_records.Submission]:
        key = (weak_job.operation_id, weak_job.window_id)
        assignment = self.shape.plan(weak_job)
        self.selections.remember_speculative_decode(key, assignment.request_key)
        if assignment.job is None:
            self._hold(key, assignment, None)
            self.submit_if_stored_data_releases(weak_job.operation_id)
            return None
        send_input = self._strong_input_send(assignment.job, None)
        return decoding_records.Submission(assignment.job, send_input)

    def escalate(self, weak_job: decoding_records.DecodeJob) -> None:
        """Ask the strong tier to re-decode the weak job's window.

        A speculative strong decode started with the weak job is
        selected as it is; otherwise the shape assigns the strong
        window, whose job is queued behind its selection now or held
        under its conditions.
        """
        self.statistics.escalated.add()
        key = (weak_job.operation_id, weak_job.window_id)
        speculative_request_key = self.selections.take_speculative_decode(key)
        if speculative_request_key is not None:
            self.statistics.speculative_selected.add()
            self._send_selection(weak_job, speculative_request_key)
            self.decode_queue.await_strong_result(key, speculative_request_key)
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
        """A weak window committed: a strong window waiting on it leaves."""
        released = self.pending.released_by_commit(window_key)
        self._submit_released(released)

    def submit_if_stored_data_releases(
        self,
        operation_id: Any,  # an opaque identity
    ) -> None:
        """A round is stored: a strong window waiting on that data leaves."""
        released = self.pending.released_by_stored_data(operation_id)
        self._submit_released(released)

    def cancel_strong_request(self, window_key: tuple) -> None:
        """A kept weak result: its strong request ends, held or submitted.

        A confident weak result halts the speculative strong decode
        (Toshio et al. 2510.25222 Sec. III A, Step 3, lines 606-614).
        One still held is dropped here, before the window's final commit
        frees the rounds it would read; one submitted is cancelled by
        the strong side's manager.
        """
        held = self.pending.held_for(window_key)
        if held is not None:
            self._drop_held(window_key, held)
        self.selections.forget_speculative_decode(window_key)
        self.decode_queue.cancel_strong(window_key)
        if self.next_edge is not None:
            self.next_edge.cancel_strong_request(window_key)

    def abandon_strong_request(self, window_key: tuple) -> None:
        """A kept weak result: a held request ends, a submitted one runs out.

        A held request was never sent, so it is dropped as a cancel drops
        it; one the strong side holds runs to its end there, and its
        result is dropped (revision_rules.NoCancel).
        """
        held = self.pending.held_for(window_key)
        if held is not None:
            self._drop_held(window_key, held)
        self.selections.forget_speculative_decode(window_key)
        self.decode_queue.abandon_strong(window_key)
        if self.next_edge is not None:
            self.next_edge.abandon_strong_request(window_key)

    def _drop_held(self, window_key: tuple, held) -> None:
        self.statistics.held_cancelled.add()
        self.pending.take(held)
        self.trace.strong_window_left.fire(
            held.assignment.request_key,
            window_key,
            "cancelled: the weak result is confident",
            False,
        )
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"speculative strong decode for {window_key} cancelled while "
            f"held for its input (the weak result is confident)",
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
        self.statistics.held.add()
        waits_for = pending_strong_windows.waiting_text(conditions)
        self.trace.strong_window_held.fire(
            assignment.request_key, key, waits_for, assignment.round_count
        )

    def _submit_released(self, released: tuple) -> None:
        """Ask each released row for its job; submit the ones it builds.

        A row with no job yet lacks rounds on the strong side, so the
        ones the chip has and has not sent go up now.
        """
        for held in released:
            job = self.shape.held_job(held.assignment)
            if job is None:
                self._carry_missing_rounds(held.assignment)
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

    # ---- private: carrying the strong window's rounds up

    def _carry_missing_rounds(self, assignment) -> None:
        """Send the rounds the strong side lacks, once each."""
        wanted = self.shape.rounds_to_carry(assignment)
        missing = []
        for round_key in wanted:
            if round_key not in self.carried_round_keys:
                missing.append(round_key)
        if not missing:
            return
        target = self.target
        packets = self.retention.escalated_rounds(missing, target)
        region = round_records.EscalatedRegion.of(
            assignment.request_key, packets, assignment.first_round
        )
        self.strong_receiver.reserve_region(region)
        self.carried_round_keys.update(missing)
        landed = functools.partial(self._region_landed, region)
        self.decoder_output.send_region(
            self.hops.forward, self.carried_from, region, landed
        )
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"strong request {assignment.request_key}: rounds "
            f"{missing[0][1]}-{missing[-1][1]} of op {missing[0][0]} "
            f"carried up with the escalation ({region.wire_bits} bits)",
        )

    def _region_landed(self, region: round_records.EscalatedRegion) -> None:
        """The region reached the strong side: its store takes every round.

        The rounds count as carried until the store has all of them, so
        a wake-up between the landing and the store sends nothing twice.
        """
        stored = functools.partial(self._region_stored, region)
        self.strong_receiver.receive_region(region, stored)

    def _region_stored(self, region: round_records.EscalatedRegion) -> None:
        self.carried_round_keys.difference_update(region.round_keys)

    # ---- private: submitting a strong job with its input send

    def _enqueue(
        self,
        strong_job: decoding_records.DecodeJob,
        selection_arrival_ticks: int,
    ) -> None:
        """Queue the strong job; its input is sent at dispatch."""
        self.enqueue_target(strong_job, selection_arrival_ticks)

    def _attempt_of(self, job: decoding_records.DecodeJob) -> list:
        """The job, and a companion per other class the next edge forces.

        A complementary gap on the edge out of the target takes the
        lightest correction in each logical class (Toshio et al.
        2510.25222 Sec. II C; Smith et al. 2405.03766 Sec. II), solved
        at once (Mishima et al. 2605.14637 App. B). The solves read one
        input, kept until the last of them has it.
        """
        forced_classes = self._forced_classes()
        if not forced_classes:
            return [job]
        job.forced_logical_class = forced_classes[0]
        attempt = [job]
        for forced_class in forced_classes[1:]:
            companion = strong_window_shapes.forced_class_companion(
                self.shape, job, forced_class
            )
            attempt.append(companion)
        shared = round_retention.SharedInputHold(job.input_hold, len(attempt))
        for attempt_job in attempt:
            attempt_job.input_hold = shared.release
        return attempt

    def _forced_classes(self) -> tuple:
        """The classes the next edge's signal pins its source's solves to."""
        if self.target_join is None:
            return ()
        signal = self.target_join.signal
        return signal.forced_logical_classes

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

        A job selected at the verdict also waits for its selection to
        arrive, and the estimate is the later of the two.
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
        """Send the window's selection; returns the tick it is expected."""
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
            self.hops.forward, weak_job, strong_request_key, delivered
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
    """The selection handshake: deferred and speculative decodes, arrivals."""

    def __init__(self) -> None:
        self.speculative_key_by_window: dict = {}
        self.deferred_weak_job_by_window: dict = {}
        self.delivered_request_keys: set = set()
        self.landing_by_request_key: dict = {}

    def defer_speculative_decode(
        self, window_key: tuple, weak_job: decoding_records.DecodeJob
    ) -> None:
        self.deferred_weak_job_by_window[window_key] = weak_job

    def take_deferred_speculative_decode(
        self, window_key: tuple
    ) -> Optional[decoding_records.DecodeJob]:
        return self.deferred_weak_job_by_window.pop(window_key, None)

    def remember_speculative_decode(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        self.speculative_key_by_window[window_key] = request_key

    def take_speculative_decode(
        self, window_key: tuple
    ) -> Optional[window_records.DecoderRequestKey]:
        return self.speculative_key_by_window.pop(window_key, None)

    def forget_speculative_decode(self, window_key: tuple) -> None:
        self.speculative_key_by_window.pop(window_key, None)

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
    """Every event the redecode reports, as one member (gem5's stats Group)."""

    strong_window_held: trace_source.TraceSource = trace_source.new_source()
    strong_window_left: trace_source.TraceSource = trace_source.new_source()
