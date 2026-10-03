"""The decode requests: a job per complete window, asked for once.

A complete window pays its store read and readiness decision before submission;
a window that still owes a boundary is masked when its decode starts (qLDPC
folds the net error into the syndrome of the next window,
qldpc/decoders/sinter.py decode_shots_to_error; cudaq-x keeps raw rounds and
applies syndrome_mods at assembly). WindowInputGate is what the decoder side
calls back into through job.gate: may_stage says whether a blocked job may take
an input slot yet, may_start whether the landed job may decode, mask_input folds
the boundary into the landed input once. The requester builds the primary tier's
job, asks the escalation policy which tiers decode the window now, and enqueues
one submission per tier on the DecodeQueue port with each job's input send; a
speculative strong decode's submission is the strong tier's window side's, and a
speculative strong decode that side holds for its input has no submission to
make here.
"""

import dataclasses
import functools
import operator
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.window_commits as window_commits
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_planner as window_planner


class WindowInputGate:
    """Whether a job may take a slot, may start, and what its input reads.

    The decoder side calls back into this through job.gate: may_stage
    says whether a boundary-blocked job may occupy an input slot yet,
    may_start whether the landed job may decode, mask_input says what the
    landed input must read once the window's boundary is folded in. It is
    decoder-side policy about one window, so it is its own class rather
    than a second face of the builder. What the mask is, and which row
    the tier declares, are decided here; the write itself is the decoder
    side's, which owns the memory and the working copy it lands in
    (<tier>.boundary_fold, decoders/settings.py, and
    decoders/decoder_memory_transfer.py).
    """

    planner = ports.Port(window_planner.WindowPlanner)
    interaction = ports.Port(window_interactions.WindowInteraction)
    # the decoder side's input, which installs what this hands it
    input_fold = ports.Port(ports.DecoderInputFold)

    def __init__(self, copies_the_fold: bool = True):
        # weak_decoder.boundary_fold: copy duplicates the landed input,
        # in_place XORs the mask into the unit's own memory
        self.copies_the_fold = copies_the_fold

    def may_stage(self, job: decoding_records.DecodeJob) -> bool:
        """May this boundary-blocked job occupy an input slot yet?

        Only when every unmet dependency is already resolving without
        needing a slot of its own: decoded, decoding, or itself dispatched
        with resolving dependencies all the way down. Admitted earlier, a
        job like the Tan seam (which reads both neighbors) squats a slot
        against the very decode that must release it.
        """
        window = job.window
        if window is None or window.deps_remaining <= 0:
            return True
        visiting = {(window.operation_id, window.window_index)}
        return self._every_dependency_resolving(window.deps, visiting)

    def may_start(self, job: decoding_records.DecodeJob) -> bool:
        """May this landed job start its decode?

        False parks the job in its slot until its window's last boundary
        arrives. Pure check: the mask itself is applied by mask_input at
        the actual start, so a re-check can never fold the boundary twice.
        """
        window = job.window
        if window is None:
            return True
        return window.deps_remaining <= 0

    def mask_input(self, job: decoding_records.DecodeJob) -> None:
        """XOR the window's boundary mask into the input the decode reads.

        The fold's condition is that a boundary arrived, which the window
        interaction answers, and not that a bit is set in it: cuda-q QEC
        applies the accumulated syndrome mods to every window past the
        first, whatever they hold (sliding_window.cpp:287-293, the
        `w > 0` branch), and a fold skipped on an all-zero mask would
        make the work the seam costs follow the noise, a cost priced
        independent of the noise so that a sweep can read it. The mask
        is this side's: what a boundary is and how it lands on a round
        layer are the window interaction's. Where the masked input is
        written is the decoder side's, which is handed it here: with the
        copy fold the unit's
        stored rounds stay raw (cudaq-x keeps raw rounds and applies
        syndrome_mods at window assembly) and the job reads a masked
        duplicate; with the in-place fold the mask goes into the unit's
        own memory and nothing is duplicated.
        """
        window = job.window
        if window is None:
            return
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
        if self.copies_the_fold:
            self.input_fold.fold_into_a_copy(job, masked_input)
            return
        self.input_fold.fold_in_place(job, masked_input)

    # ---- private

    def _every_dependency_resolving(self, dependencies, visiting: set) -> bool:
        for dependency in dependencies:
            if not self._resolving_without_new_slots(dependency, visiting):
                return False
        return True

    def _resolving_without_new_slots(self, key: tuple, visiting: set) -> bool:
        if key in visiting:
            return False
        window = self.planner.windows_by_key.get(key)
        if _needs_no_slot(window):
            return True
        if window.t_dispatch is None:
            return False
        visited = visiting | {key}
        return self._every_dependency_resolving(window.deps, visited)

    def _masked_round(self, state, window_info, round_input):
        fragments = []
        for fragment in round_input.fragments:
            masked = self.interaction.apply_boundary(
                state, window_info, fragment, round_input.round_index
            )
            fragments.append(masked)
        return dataclasses.replace(round_input, fragments=tuple(fragments))


class DecodeRequestBuilder:
    """Builds one decode job from a complete window.

    It stamps the gate on every job it builds, so the decoder side has
    the window's input policy without knowing the window package. Trace
    source: window_data_complete(window, read_keys) when the last round
    the window reads is readable in its store, which is the moment the
    window may be requested.
    """

    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker_module.RoundTracker)
    interaction = ports.Port(window_interactions.WindowInteraction)
    gate = ports.Port(WindowInputGate)

    def __init__(self, engine) -> None:
        self.engine = engine
        self.next_request_sequence = 0
        self.trace = _TraceSources()

    # ---- building a request

    def new_request_key(
        self, operation_id, window_id: int, tier: window_records.DecoderTier
    ) -> window_records.DecoderRequestKey:
        """The next request identity, run-wide ordinal included."""
        request_key = window_records.DecoderRequestKey(
            operation_id, window_id, tier, self.next_request_sequence
        )
        self.next_request_sequence += 1
        return request_key

    def stamp_first_round(self, window: window_records.Window, store) -> None:
        """Retain arrival provenance for latency accounting."""
        if window.t_first_round is not None:
            return
        first_round_key = (window.operation_id, window.start_round)
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
        tier: window_records.DecoderTier,
        store,
        forced_logical_class: Optional[int] = None,
    ) -> decoding_records.DecodeJob:
        """The tier's decode job for one complete window, read from store.

        forced_logical_class is the class the job's solve is pinned to,
        None for the window's own unconstrained decode.
        """
        request_key = self.new_request_key(
            window.operation_id, window.window_index, tier
        )
        payloads = self.assemble_payloads(window, store)
        payload_round_count = decoding_records.distinct_round_count(payloads)
        round_count = (
            payload_round_count + window.batched_preceding_idle_round_count
        )
        spatial_nodes = self.planner.spatial_node_count_of(operation.id)
        geometry = self.planner.code_geometry_of(operation.id)
        model = self.planner.models.model_by_window.get(window.key)
        label = self._job_label(window, operation)
        return decoding_records.DecodeJob(
            operation_id=window.operation_id,
            window_id=window.window_index,
            round_count=round_count,
            ready_time=self.engine.now,
            spatial_nodes=spatial_nodes,
            payloads=payloads,
            detector_error_model=model,
            code=geometry.code_name,
            window=window,
            label=label,
            strong_label=f"strong({operation.name} W{window.window_index})",
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
            job.operation_id, job.window_id, job.request_key.tier
        )
        label = f"{job.label} class {forced_logical_class}"
        payloads = list(job.payloads)
        return decoding_records.DecodeJob(
            operation_id=job.operation_id,
            window_id=job.window_id,
            round_count=job.round_count,
            ready_time=self.engine.now,
            spatial_nodes=job.spatial_nodes,
            payloads=payloads,
            detector_error_model=job.detector_error_model,
            code=job.code,
            window=job.window,
            label=label,
            strong_label=job.strong_label,
            request_key=request_key,
            input_key=job.input_key,
            request_created_ticks=self.engine.now,
            gate=self.gate,
            forced_logical_class=forced_logical_class,
        )

    def assemble_payloads(self, window: window_records.Window, store) -> list:
        """Collect this window's raw payloads, with successor overflow rounds.

        The boundary is never folded here: the mask is XORed into the
        landed input at the decoder when the decode starts.
        """
        operation_rounds = self.tracker.effective_round_count_for_window(
            window.operation_id, window
        )
        end_round = min(window.buffer_hi, operation_rounds)
        window_info = window_records.WindowInfo.from_window(window)
        payloads = []
        stop_round = end_round + 1
        for round_index in range(window.start_round, stop_round):
            round_key = (window.operation_id, round_index)
            fragments = store.retained_fragments(round_key)
            self._append_round_payloads(
                payloads, fragments, window_info, round_index
            )
        overflow = window.buffer_hi - operation_rounds
        if overflow <= 0:
            return payloads
        successor_ids = self.planner.successors_by_operation.get(
            window.operation_id, []
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
        if self.planner.is_windowed(window.operation_id):
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
    """Requests a decode for every complete, unblocked window, once.

    gap_join is the confidence join of a run whose gap is two
    forced-class solves; it is both jobs' on_decoded and None when the
    run's weak decoder reports its own soft output from one decode.
    """

    tracker = ports.Port(round_tracker_module.RoundTracker)
    retention = ports.Port(round_retention_module.RoundRetention)
    builder = ports.Port(DecodeRequestBuilder)
    decode_queue = ports.Port(ports.DecodeQueue)
    # the strong side's manager, which serves the speculative decode a window
    # submits beside its weak job; a run that never escalates has none
    strong_decode_queue = ports.Port(ports.DecodeQueue, optional=True)
    # the switching policy; None on a run with no switching, whose
    # windows are decoded on the primary tier alone
    escalation_policy = ports.Port(ports.EscalationPolicy, optional=True)
    verdict = ports.Port(window_commits.WindowVerdict)
    # the primary store's outgoing port; it executes the input send
    store_output = ports.Port(ports.SyndromeBufferOutput)
    # a run whose weak decoder reports its own soft output has no join
    gap_join = ports.Port(ports.WindowGapJoin, optional=True)

    def __init__(
        self,
        clock: Optional[config.Clock] = None,
        decision_cycles: int = 0,
    ) -> None:
        self.clock = clock
        self.decision_cycles = decision_cycles
        # the windows whose decision has not ended, by window key
        self.deciding_by_window: dict = {}

    def request_ready_windows(self, windows, strong_redecode) -> None:
        """Request each window that has its data, in the given order.

        strong_redecode is the strong tier's window side, which builds the
        speculative strong decode when the policy decodes both tiers at once;
        None when the run never escalates.
        """
        for window in windows:
            self.request_if_ready(window, strong_redecode)

    def request_if_ready(
        self, window: window_records.Window, strong_redecode
    ) -> None:
        """If the window has its data, submit it through the policy."""
        if window.queued or window.committed:
            return
        if window.t_first_round is None and self.tracker.has_first_round(
            window
        ):
            store = self.retention.store_for(None)
            self.builder.stamp_first_round(window, store)
        if not self.tracker.is_data_complete(window):
            return
        operation = self.tracker.operation_by_id[window.operation_id]
        read_keys = self.retention.read_keys_for_bounds(
            window.operation_id, window.start_round, window.buffer_hi, window
        )
        self.builder.note_data_complete(window, operation, read_keys)
        # a window still owed a boundary ships its raw rounds now; the
        # boundary is XORed into the landed input at the decoder when it
        # arrives (qLDPC net_error / cudaq-x syndrome_mods / LILLIPUT's
        # state register)
        self.request(window, operation, strong_redecode)

    def request(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        strong_redecode,
    ) -> None:
        """Build the primary jobs; the window's requests leave at its decision.

        The primary tier's jobs are built and their input held here; a strong
        tier the policy names too gets its speculative strong decode from the
        strong redecode. The window side decides once per window, the pre-decode
        step a syndrome passes before any decoder has it (RISC-Q 2603.16203
        lines 895-898), and Step 1 feeds that syndrome to both decoders at once
        (Toshio et al. 2510.25222 lines 598-601), so the speculative strong
        decode is planned and every request admitted when it ends. The store is
        read when the input leaves it, at dispatch
        (syndrome_buffer/round_output.py), not here.
        """
        store = self.retention.primary_store
        self.builder.stamp_first_round(window, store)
        primary_tier = self.retention.primary_tier
        forced_classes = self._forced_logical_classes()
        first_class = _first_forced_class(forced_classes)
        job = self.builder.build(
            window, operation, primary_tier, store, first_class
        )
        window.queued = True
        tiers = self._tiers_for(window, primary_tier)
        primary_jobs = self._primary_jobs(job, forced_classes)
        window_reads = decoding_records.WindowReads(window.key)
        is_input_held = self._bind_input_hold(primary_jobs, window_reads)
        primary = self._primary_submissions(primary_jobs, is_input_held)
        deciding = _DecidingWindow(job, tiers, primary, strong_redecode)
        if self.decision_cycles == 0:
            self._issue(deciding)
            return
        self.deciding_by_window[window.key] = deciding
        engine = self.builder.engine
        edge = self.clock.edge(self.decision_cycles, engine.now)
        delay = edge - engine.now
        decide = functools.partial(self._decide, window.key)
        engine.schedule(delay, decide, label="window decision")

    def _tiers_for(self, window, primary_tier) -> tuple:
        """The tiers decoding the window now: the policy's, or the primary."""
        if self.escalation_policy is None:
            return (primary_tier,)
        return self.escalation_policy.tiers_for_ready_window(window)

    def _decide(self, window_key: tuple) -> None:
        """The window's decision ended: issue its requests, unless withdrawn."""
        deciding = self.deciding_by_window.pop(window_key, None)
        if deciding is None:
            return
        self._issue(deciding)

    def _issue(self, deciding: "_DecidingWindow") -> None:
        """Admit the primary jobs and the speculative decode beside them."""
        primary_tier = self.retention.primary_tier
        submissions = []
        for tier in deciding.tiers:
            if tier is primary_tier:
                submissions.extend(deciding.primary_submissions)
                continue
            strong_redecode = deciding.strong_redecode
            speculative = strong_redecode.parallel_strong_submission(
                deciding.job
            )
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

    def _forced_logical_classes(self) -> tuple:
        """The classes this run's confidence has each window pinned to."""
        if self.gap_join is None:
            return ()
        return self.gap_join.signal.forced_logical_classes

    def _primary_jobs(
        self, job: decoding_records.DecodeJob, forced_classes: tuple
    ) -> list:
        """The primary tier's jobs: the built one, and one per other class.

        A confidence built from forced-class solves needs the window
        decoded once per class, and all of them are asked for at one instant
        (CUDA launches a whole grid in one call, CUDA C++ Programming
        Guide section 5.1, Kernels; OpenMP's primary thread "creates a
        team of itself and zero or more additional threads", OpenMP 5.2
        specification section 1.3, Execution Model).
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

        The send is the store's own (syndrome_buffer/round_output.py):
        this side asks for it and the decoder manager calls it at
        dispatch, and neither of them executes it.
        """
        send_input = self.store_output.input_send_for(job, is_input_held)
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
        store = self.retention.primary_store
        self.retention.bind_input_hold(first, window_reads, store)
        reader_count = len(primary_jobs)
        if reader_count == 1:
            return False
        shared = _SharedInputHold(first.input_hold, reader_count)
        for job in primary_jobs:
            job.input_hold = shared.release
        return False

    def check_settled(self) -> None:
        """At the end of a run no window may still hold an unjoined solve."""
        if self.gap_join is None:
            return
        unresolved = self.gap_join.unresolved_windows()
        if not unresolved:
            return
        raise RuntimeError(
            f"the run ended with windows holding an unjoined solve: "
            f"{unresolved}"
        )

    def withdraw(self, window: window_records.Window) -> None:
        """Withdraw one window's early-shipped, unstarted decode.

        Its submission bookkeeping is reset so it can be resubmitted fresh
        (a strong window that absorbs the window owns its rounds from then
        on). The reset comes first: taking the decode back frees its slot
        and the manager dispatches at once, and a later window staged then
        must not read this one as still dispatched, the order gem5's IEW
        keeps by taking a squash before it dispatches (src/cpu/o3/iew.cc
        1451-1452).
        """
        window.queued = False
        window.t_queued = None
        window.t_dispatch = None
        window.service_began = False
        has_dropped_decision = self._withdraw_decision(window.key)
        if not has_dropped_decision:
            self.decode_queue.withdraw_window(window.key)

    def release_parked(self, window_key: tuple, strong_redecode) -> None:
        """The window's last boundary arrived: its parked decodes may start.

        The window's weak decode parks on the chip's manager; a speculative
        strong decode or a re-decode of the same window parks on the host's. A
        speculative strong decode the strong side waited to plan until the weak
        job left its park is submitted first, so the two start together.
        """
        if strong_redecode is not None:
            speculative = strong_redecode.unparked_submission(window_key)
            started = _speculative_submissions(speculative)
            for submission in started:
                self._admit(submission)
        self.decode_queue.release_parked(window_key)
        if self.strong_decode_queue is not None:
            self.strong_decode_queue.release_parked(window_key)

    def _admit(self, submission: decoding_records.Submission) -> None:
        job = submission.job
        if job.strong_decode_for is None:
            job.window.t_queued = self.builder.engine.now
        queue = self.decode_queue
        on_decoded = self.verdict.accept_result
        if job.strong_decode_for is not None:
            # the speculative strong decode started with the weak job (Toshio
            # 2510.25222 lines 598-601) is the strong side's to serve
            queue = self.strong_decode_queue
            on_decoded = self.verdict.accept_strong_result
        elif self.gap_join is not None:
            on_decoded = self.gap_join.accept_result
        queue.enqueue(job, submission.send_input, on_decoded)

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


class _SharedInputHold:
    """One store hold released when the last job that reads it has landed.

    The forced-class solves of one window read the same rounds, so
    the weak syndrome buffer keeps them until every one of them has its input,
    the lifetime a zero-copy send owes its source (the kernel's dmaengine
    client rule that a mapping lives until the transfer completes).
    """

    def __init__(self, release: Callable[[], None], reader_count: int) -> None:
        self.hold_release = release
        self.readers_left = reader_count

    def release(self) -> None:
        """One reader has its input; the rounds go when the last one does."""
        self.readers_left -= 1
        if self.readers_left > 0:
            return
        self.hold_release()


@dataclasses.dataclass(frozen=True)
class _DecidingWindow:
    """A window in its decision: its primary job, tiers and submissions."""

    job: decoding_records.DecodeJob
    tiers: tuple
    primary_submissions: list
    # the strong tier's window side; None on a run that never escalates
    strong_redecode: Optional[ports.StrongRedecode]


def _needs_no_slot(window: Optional[window_records.Window]) -> bool:
    """Unplanned, absorbed, decoded or decoding: it asks for no slot."""
    if window is None:
        return True
    if window.is_absorbed:
        return True
    if window.t_done is not None:
        return True
    return window.service_began


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
    """Every event the decode request builder reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    window_data_complete: trace_source.TraceSource = trace_source.new_source()
