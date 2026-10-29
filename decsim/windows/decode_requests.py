"""The decode requests: a job per complete window, asked for once.

A complete window pays its store read and readiness decision before
submission. A window that still owes a boundary is masked when its
decode starts (qLDPC folds the net error into the next window's
syndrome, qldpc/decoders/sinter.py decode_shots_to_error; cudaq-x keeps
raw rounds and applies syndrome_mods at assembly). The requester builds
the primary tier's job, asks the escalation policy which tiers decode
the window now, and enqueues one submission per tier on the DecodeQueue
port.
"""

import dataclasses
import functools
import operator
from typing import Any, Optional

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.input_checks as input_checks_module
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.routers as routers
import decsim.windows.window_commits as window_commits
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_planner as window_planner


class WindowInputGate:
    """The boundary gate a window's decode job passes through.

    It says what the job's input reads when its decode starts. The
    decoder side calls it through job.gate. It decides the mask and the
    row the tier declares; the write is the decoder side's, which owns
    the memory (<tier>.boundary_fold, decoders/settings.py).
    """

    interaction = ports.Port(window_interactions.WindowInteraction)
    # the decoder side's input, which installs what this hands it
    input_fold = ports.Port(ports.DecoderInputFold)
    # what each decode reads, kept while a boundary it read is not final
    input_checks = ports.Port(input_checks_module.InputChecks)

    def __init__(self, copies_fold_by_pool: dict):
        # each pool's copies_boundary_fold: a copy duplicates the landed
        # input, in place XORs the mask into the unit's own memory. A
        # job is folded as the pool that decodes it folds, an entry
        # pool's window or any target's re-decode alike
        self.copies_fold_by_pool = copies_fold_by_pool

    def mask_input(self, job: decoding_records.DecodeJob) -> None:
        """XOR the window's boundary mask into the input the decode reads.

        The fold runs when a boundary arrived, not when a bit is set:
        cuda-q QEC applies the accumulated syndrome mods to every window
        past the first (sliding_window.cpp:287-293), so the seam's cost
        does not follow the noise. With the copy fold the unit's stored
        rounds stay raw and the job reads a masked duplicate, as cudaq-x
        does; with the in-place fold the mask goes into the unit's own
        memory.
        """
        window = job.window
        if window is None:
            return
        self.input_checks.note_read(job)
        state = window.boundary_in
        if job.decoder_input is None:
            return
        if not self.interaction.boundary_arrived(state):
            return
        window_info = window_records.WindowInfo.from_window(window)
        masked_rounds = []
        for round_input in job.decoder_input.rounds:
            masked = self._masked_round(state, window_info, round_input)
            masked_rounds.append(masked)
        masked_input = dataclasses.replace(
            job.decoder_input, rounds=tuple(masked_rounds)
        )
        if self._copies_the_fold(job):
            self.input_fold.fold_into_a_copy(job, masked_input)
            return
        self.input_fold.fold_in_place(job, masked_input)

    # ---- private

    def _copies_the_fold(self, job: decoding_records.DecodeJob) -> bool:
        """The fold of the pool that decodes this job."""
        pool = job.request_key.pool
        return self.copies_fold_by_pool[pool]

    def _masked_round(self, state, window_info, round_input):
        fragments = []
        for fragment in round_input.fragments:
            masked = self.interaction.apply_boundary(
                state, window_info, fragment, round_input.round_index
            )
            fragments.append(masked)
        return dataclasses.replace(round_input, fragments=tuple(fragments))


class DecodeRequestBuilder:
    """Builds one decode job from a complete window, its gate stamped on.

    Trace source: window_data_complete(window, read_keys) when the last
    round the window reads is readable in its store.
    """

    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker_module.RoundTracker)
    interaction = ports.Port(window_interactions.WindowInteraction)
    gate = ports.Port(WindowInputGate)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.next_request_sequence = 0
        self.trace = _TraceSources()

    # ---- building a request

    def new_request_key(
        self,
        stream_id: Any,  # an opaque identity
        window_id: int,
        tier: window_records.PoolId,
    ) -> window_records.DecoderRequestKey:
        """The next request identity, run-wide ordinal included."""
        request_key = window_records.DecoderRequestKey(
            stream_id, window_id, tier, self.next_request_sequence
        )
        self.next_request_sequence += 1
        return request_key

    def stamp_first_round(
        self, window: window_records.Window, store: ports.RetainedRounds
    ) -> None:
        """Retain arrival provenance for latency accounting."""
        if window.t_first_round is not None:
            return
        first_round_key = (window.stream_id, window.start_round)
        window.t_first_round = store.publication_tick(first_round_key)

    def note_data_complete(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        read_keys: list,
    ) -> None:
        """Stamp the window's data-complete tick the first time it is seen.

        A trailing buffer satisfied by memory rounds alone releases on
        time with no syndrome content behind it; the log line keeps that
        approximation visible.
        """
        if window.t_data_complete is not None:
            return
        window.t_data_complete = self.engine.now
        self.trace.window_data_complete.fire(window, read_keys)
        if not self.tracker.is_buffer_filled_by_memory(window):
            return
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"{operation.name} W{window.window_index} buffer filled by "
            f"memory rounds (time-only, no syndrome content)",
        )

    def build(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        tier: window_records.PoolId,
        store: ports.RetainedRounds,
        forced_logical_class: Optional[int] = None,
    ) -> decoding_records.DecodeJob:
        """The tier's decode job for one complete window, read from store.

        forced_logical_class is the class the job's solve is pinned to,
        None for the window's own unconstrained decode.
        """
        request_key = self.new_request_key(
            window.stream_id, window.window_index, tier
        )
        payloads = self.assemble_payloads(window, store)
        payload_round_count = decoding_records.distinct_round_count(payloads)
        round_count = (
            payload_round_count + window.batched_preceding_idle_round_count
        )
        model = self.planner.models.model_by_window.get(window.key)
        label = self._job_label(window, operation)
        return decoding_records.DecodeJob(
            stream_id=window.stream_id,
            window_id=window.window_index,
            round_count=round_count,
            ready_time=self.engine.now,
            payloads=payloads,
            detector_error_model=model,
            window=window,
            label=label,
            request_key=request_key,
            input_key=request_key,
            request_created_ticks=self.engine.now,
            gate=self.gate,
            forced_logical_class=forced_logical_class,
        )

    def build_forced_companion(
        self, job: decoding_records.DecodeJob, forced_logical_class: int
    ) -> decoding_records.DecodeJob:
        """The window's other forced-class job over the same rounds.

        Its own request, so both transfers are priced and recorded
        separately, and the first request's key as its input identity,
        so a unit that already holds the rounds reads them instead of
        receiving them again.
        """
        request_key = self.new_request_key(
            job.stream_id, job.window_id, job.request_key.pool
        )
        label = f"{job.label} class {forced_logical_class}"
        payloads = list(job.payloads)
        return decoding_records.DecodeJob(
            stream_id=job.stream_id,
            window_id=job.window_id,
            round_count=job.round_count,
            ready_time=self.engine.now,
            payloads=payloads,
            detector_error_model=job.detector_error_model,
            window=job.window,
            label=label,
            request_key=request_key,
            input_key=job.input_key,
            request_created_ticks=self.engine.now,
            gate=self.gate,
            forced_logical_class=forced_logical_class,
        )

    def assemble_payloads(
        self, window: window_records.Window, store: ports.RetainedRounds
    ) -> list:
        """Collect this window's raw payloads, with the rounds it reads besides.

        Those are its successors' overflow rounds and each region it
        reads on another stream.

        The boundary is never folded here: the mask is XORed into the
        landed input at the decoder when the decode starts.
        """
        operation_rounds = self.tracker.effective_round_count_for_window(
            window.stream_id, window
        )
        end_round = min(window.buffer_hi, operation_rounds)
        window_info = window_records.WindowInfo.from_window(window)
        payloads = []
        stop_round = end_round + 1
        for round_index in range(window.start_round, stop_round):
            round_key = (window.stream_id, round_index)
            fragments = store.retained_fragments(round_key)
            self._append_round_payloads(
                payloads, fragments, window_info, round_index
            )
        for span in window.spatial_reads:
            self._append_span_payloads(payloads, store, span, window_info)
        overflow = window.buffer_hi - operation_rounds
        if overflow <= 0:
            return payloads
        successor_ids = self.planner.successors_by_stream.get(
            window.stream_id, []
        )
        for successor_id in successor_ids:
            self._append_overflow_payloads(
                payloads,
                store,
                successor_id,
                overflow,
                operation_rounds,
                window_info,
            )
        return payloads

    # ---- private

    def _job_label(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
    ) -> str:
        if self.planner.is_windowed(window.stream_id):
            return (
                f"{operation.name} W{window.window_index} "
                f"[commit {window.commit_lo}-{window.commit_hi}]"
            )
        body_rounds = self.tracker.round_count_for_window(operation.id, window)
        idle_rounds = window.batched_preceding_idle_round_count
        if idle_rounds:
            effective_rounds = window.round_count + idle_rounds
            return (
                f"{operation.name} [whole op, {effective_rounds} rounds: "
                f"{idle_rounds} idle + {body_rounds} body]"
            )
        return f"{operation.name} [whole op, {window.round_count} rounds]"

    def _append_overflow_payloads(
        self,
        payloads,
        store,
        successor_id,
        overflow,
        operation_rounds,
        window_info,
    ) -> None:
        stop_round = overflow + 1
        for round_index in range(1, stop_round):
            round_key = (successor_id, round_index)
            fragments = store.retained_fragments(round_key)
            shifted_round_index = operation_rounds + round_index
            self._append_round_payloads(
                payloads, fragments, window_info, shifted_round_index
            )

    def _append_span_payloads(self, payloads, store, span, window_info) -> None:
        """Every round of a region of another stream, at its own index."""
        stop_round = span.last_round + 1
        for round_index in range(span.first_round, stop_round):
            round_key = (span.stream_id, round_index)
            fragments = store.retained_fragments(round_key)
            self._append_round_payloads(
                payloads, fragments, window_info, round_index
            )

    def _append_round_payloads(
        self, payloads, fragments, window_info, round_index
    ) -> None:
        """One round's fragments in measurement order, no boundary folded."""
        if fragments is None:
            return
        by_fragment_index = operator.attrgetter("fragment_index")
        ordered = sorted(fragments, key=by_fragment_index)
        for fragment in ordered:
            payload = self.interaction.apply_boundary(
                None, window_info, fragment, round_index
            )
            payloads.append(payload)


class DecodeRequester:
    """Requests a decode for every complete window, once.

    The router names each window's entry pool and holds its side: its
    manager, the store end that sends to it, and its confidence join,
    which is both forced-class solves' on_decoded and None when the
    pool reports its own soft output from one decode
    (windows/routers.py).
    """

    tracker = ports.Port(round_tracker_module.RoundTracker)
    retention = ports.Port(round_retention_module.RoundRetention)
    builder = ports.Port(DecodeRequestBuilder)
    # the run's escalate edges: the edge leaving the primary tier builds
    # the speculative decode a window submits beside its weak job, and
    # each edge's target manager serves its own; a run with no
    # switching has none, and decodes its windows on the primary alone
    edges = ports.Port(escalate_edges.EscalateEdges)
    verdict = ports.Port(window_commits.WindowVerdict)
    # which entry pool decodes each window, and that pool's side
    router = ports.Port(routers.Router)

    def __init__(
        self,
        clock: Optional[config.Clock] = None,
        decision_cycles: int = 0,
    ) -> None:
        self.clock = clock
        self.decision_cycles = decision_cycles
        # the windows whose decision has not ended, by window key
        self.deciding_by_window: dict = {}
        # the keys of windows withdrawn and open again, until the window
        # manager's cursor has gone back to them
        self.withdrawn_keys: set = set()

    def request_ready_windows(self, windows: list) -> None:
        """Request each window that has its data, in the given order."""
        for window in windows:
            self.request_if_ready(window)

    def request_if_ready(self, window: window_records.Window) -> None:
        """If the window has its data, submit it through the policy."""
        if _is_left_alone(window):
            return
        if window.t_first_round is None and self.tracker.has_first_round(
            window
        ):
            store = self.retention.store_of_stream(window.stream_id)
            self.builder.stamp_first_round(window, store)
        if not self._has_its_rounds(window):
            return
        operation = self.tracker.operation_by_id[window.stream_id]
        read_keys = self.retention.read_keys_for_bounds(
            window.stream_id, window.start_round, window.buffer_hi, window
        )
        self.builder.note_data_complete(window, operation, read_keys)
        # a window still owed a boundary is queued now and takes a unit
        # when the boundary arrives; the boundary is XORed into the landed
        # input at the decode's start (qLDPC net_error / cudaq-x
        # syndrome_mods / LILLIPUT's state register)
        self.request(window, operation)

    def request(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
    ) -> None:
        """Build the primary jobs; the window's requests leave at its decision.

        The window side decides once per window, the pre-decode step a
        syndrome passes before any decoder has it (RISC-Q 2603.16203
        lines 895-898), and Step 1 feeds the syndrome to both decoders
        at once (Toshio et al. 2510.25222 lines 598-601), so a
        speculative strong decode is planned and every request admitted
        when the decision ends. The store is read at dispatch
        (syndrome_buffer/round_output.py).
        """
        store = self.retention.store_of_stream(window.stream_id)
        self.builder.stamp_first_round(window, store)
        entry_pool = self._pool_for(window)
        forced_classes = self._forced_logical_classes(entry_pool)
        first_class = _first_forced_class(forced_classes)
        job = self.builder.build(
            window, operation, entry_pool, store, first_class
        )
        window.queued = True
        primary_jobs = self._primary_jobs(job, forced_classes)
        window_reads = decoding_records.WindowReads(window.key)
        is_input_held = self._bind_input_hold(primary_jobs, window_reads)
        primary = self._primary_submissions(primary_jobs, is_input_held)
        deciding = _DecidingWindow(job, primary)
        if self.decision_cycles == 0:
            self._issue(deciding)
            return
        self.deciding_by_window[window.key] = deciding
        engine = self.builder.engine
        edge = self.clock.edge(self.decision_cycles, engine.now)
        delay = edge - engine.now
        decide = functools.partial(self._decide, window.key)
        engine.schedule(delay, decide, label="window decision")

    def _has_its_rounds(self, window: window_records.Window) -> bool:
        """Whether every round the window reads is in its pool's store.

        A round of another pool's stream is there once its copy landed
        (controller/round_destinations.py).
        """
        if not self.tracker.is_data_complete(window):
            return False
        return self.retention.copied_reads_landed(window)

    def _decide(self, window_key: tuple) -> None:
        """The window's decision ended: issue its requests, unless withdrawn."""
        deciding = self.deciding_by_window.pop(window_key, None)
        if deciding is None:
            return
        self._issue(deciding)

    def _issue(self, deciding: "_DecidingWindow") -> None:
        """Admit the primary jobs and the speculative decode beside them."""
        submissions = list(deciding.primary_submissions)
        job = deciding.job
        edge = self.edges.leaving(job.request_key.pool)
        if edge is not None:
            # the queue is read before the job joins it (deadlines.py)
            edge.arm_deadline(job)
        if edge is not None and edge.policy.decodes_both_at_once(job.window):
            speculative = edge.parallel_strong_submission(job)
            started = _speculative_submissions(speculative)
            submissions.extend(started)
        for submission in submissions:
            self._admit(submission)

    def _primary_submissions(
        self, primary_jobs: list, is_input_held: bool
    ) -> list:
        """One submission per primary job, each with its own input send."""
        submissions = []
        for job in primary_jobs:
            submission = self._primary_submission(job, is_input_held)
            submissions.append(submission)
        return submissions

    def _forced_logical_classes(self, pool: window_records.PoolId) -> tuple:
        """The classes the pool's confidence has each window pinned to."""
        gap_join = self._join_of(pool)
        if gap_join is None:
            return ()
        return gap_join.signal.forced_logical_classes

    # ---- private: the entry pool a window is decoded on

    def _pool_for(self, window: window_records.Window) -> window_records.PoolId:
        """The entry pool that decodes the window."""
        return self.router.pool_for(window)

    def _queue_of(self, pool: window_records.PoolId) -> ports.DecodeQueue:
        """The manager of an entry pool."""
        side = self.router.side(pool)
        return side.decode_queue

    def _join_of(
        self, pool: window_records.PoolId
    ) -> Optional[ports.WindowGapJoin]:
        """The join of an entry pool's solves; None when it has none."""
        side = self.router.side(pool)
        return side.gap_join

    def _store_output_of(
        self, pool: window_records.PoolId
    ) -> ports.SyndromeBufferOutput:
        """The end of the store an entry pool reads that sends to it."""
        side = self.router.side(pool)
        return side.store_output

    def _primary_jobs(
        self, job: decoding_records.DecodeJob, forced_classes: tuple
    ) -> list:
        """The primary tier's jobs: the built one, and one per other class.

        A confidence from forced-class solves needs one decode per
        class, all asked for at one instant, as a CUDA grid launches in
        one call (CUDA C++ Programming Guide section 5.1).
        """
        jobs = [job]
        for forced_class in forced_classes[1:]:
            companion = self.builder.build_forced_companion(job, forced_class)
            jobs.append(companion)
        return jobs

    def _primary_submission(
        self, job: decoding_records.DecodeJob, is_input_held: bool
    ) -> decoding_records.Submission:
        """One primary job with its input send; a held input lands at once.

        The send is the store's own (syndrome_buffer/round_output.py);
        the decoder manager calls it at dispatch.
        """
        store_output = self._store_output_of(job.request_key.pool)
        send_input = store_output.input_send_for(job, is_input_held)
        return decoding_records.Submission(job, send_input)

    def _bind_input_hold(self, primary_jobs: list, window_reads) -> bool:
        """Move the window's hold to the attempt; whether it was held already.

        The jobs of one attempt read the same rounds, so they share one
        hold and the weak syndrome buffer may drop the rounds when the last of
        them has its input; a job whose rounds this side already holds keeps
        them and moves nothing.
        """
        first = primary_jobs[0]
        if first.submitted or self.retention.holds_input(first):
            return True
        store = self.retention.store_of_stream(first.request_key.stream_id)
        self.retention.bind_input_hold(first, window_reads, store)
        reader_count = len(primary_jobs)
        if reader_count == 1:
            return False
        shared = round_retention_module.SharedInputHold(
            first.input_hold, reader_count
        )
        for job in primary_jobs:
            job.input_hold = shared.release
        return False

    def check_settled(self) -> None:
        """At the end of a run no window may still hold an unjoined solve."""
        for side in self.router.sides():
            _check_joined(side)

    def decode_again(self, window: window_records.Window) -> None:
        """Decode the window again, on the boundary it holds now.

        A decode of it read a boundary that changed (input_checks.py).
        Its rounds are still stored: a window's potential strong hold
        keeps them until the window is final
        (round_retention.register_window), and the fresh request takes
        its own hold on them. The window keeps its first queue entry,
        and the final boundary releases it now, so the dropped decode
        is part of its wait for that boundary (experiments/measure.py
        dep_block).
        """
        window.t_released = self.builder.engine.now
        operation = self.tracker.operation_by_id[window.stream_id]
        self.request(window, operation)

    def withdraw(self, window: window_records.Window) -> None:
        """Withdraw one window's unstarted decode.

        Its submission bookkeeping is reset first, so it can be
        resubmitted fresh: taking the decode back may free a slot, and
        the manager dispatches at once. gem5's IEW takes a squash before
        it dispatches in the same order (src/cpu/o3/iew.cc 1451-1452).
        """
        window.queued = False
        self.withdrawn_keys.add(window.key)
        window.t_queued = None
        window.t_dispatch = None
        window.service_began = False
        has_dropped_decision = self._withdraw_decision(window.key)
        if not has_dropped_decision:
            entry_pool = self._pool_for(window)
            decode_queue = self._queue_of(entry_pool)
            decode_queue.withdraw_window(window.key)

    def release_window(self, window_key: tuple) -> None:
        """The window's last boundary arrived: its waiting decodes may start.

        The window's first decode waits in its entry pool's manager; a
        speculative strong decode or a re-decode of the same window waits
        in its edge's target manager. A speculative strong decode the
        edge out of the entry pool waited to plan until this boundary is
        submitted first; then every manager releases the window, each
        entry pool's in build order, then each edge's target in graph
        order.
        """
        stream_id, _window_index = window_key
        entry = self.router.pool_of_stream(stream_id)
        edge = self.edges.leaving(entry)
        if edge is not None:
            speculative = edge.released_submission(window_key)
            started = _speculative_submissions(speculative)
            for submission in started:
                self._admit(submission)
        for side in self.router.sides():
            side.decode_queue.release_window(window_key)
        for decode_queue in self.edges.target_queues():
            decode_queue.release_window(window_key)

    def _admit(self, submission: decoding_records.Submission) -> None:
        job = submission.job
        if job.strong_decode_for is not None:
            # the speculative strong decode started with the weak job (Toshio
            # 2510.25222 lines 598-601) is its edge's target's to serve
            entry = self._pool_for(job.window)
            edge = self.edges.entering(job.request_key.pool, entry)
            edge.enqueue_target(job, None)
            return
        # a window decoded again keeps its first entry (decode_again)
        if job.window.t_queued is None:
            job.window.t_queued = self.builder.engine.now
        entry_pool = job.request_key.pool
        on_decoded = self.verdict.accept_revision
        gap_join = self._join_of(entry_pool)
        if gap_join is not None:
            on_decoded = gap_join.accept_result
        decode_queue = self._queue_of(entry_pool)
        decode_queue.enqueue(job, submission.send_input, on_decoded)

    def _withdraw_decision(self, window_key: tuple) -> bool:
        """Drop a window still in its decision and release its input hold."""
        deciding = self.deciding_by_window.pop(window_key, None)
        if deciding is None:
            return False
        for submission in deciding.primary_submissions:
            job = submission.job
            if job.input_hold is not None:
                job.input_hold()
                job.input_hold = None
        return True


@dataclasses.dataclass(frozen=True)
class _DecidingWindow:
    """What one window issues when its decision ends."""

    job: decoding_records.DecodeJob
    primary_submissions: list


def _is_left_alone(window: window_records.Window) -> bool:
    """Queued, committed, or covered by a strong region (window_planner.py)."""
    if window.is_absorbed:
        return True
    return window.queued or window.committed


def _speculative_submissions(speculative) -> list:
    """The speculative decode's submission, or none while its side holds it."""
    if speculative is None:
        return []
    return [speculative]


def _first_forced_class(forced_classes: tuple) -> Optional[int]:
    """The class the window's own job is built with; None when free."""
    if not forced_classes:
        return None
    return forced_classes[0]


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decode request builder reports, as one member."""

    window_data_complete: trace_source.TraceSource = trace_source.new_source()


def _check_joined(side: routers.EntrySide) -> None:
    """No window of the entry pool still holds an unjoined solve."""
    if side.gap_join is None:
        return
    unresolved = side.gap_join.unresolved_windows()
    if not unresolved:
        return
    raise RuntimeError(
        f"the run ended with windows of {side.pool.word} holding an "
        f"unjoined solve: {unresolved}"
    )
