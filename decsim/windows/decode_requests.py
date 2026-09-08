"""The decode requests: a job per complete window, asked for once.

A window's raw rounds ship as soon as its data is complete; a window
that still owes a boundary is masked at the decoder when its decode
starts (qLDPC folds the net error into the syndrome of the next window,
qldpc/decoders/sinter.py decode_shots_to_error; cudaq-x keeps raw
rounds and applies syndrome_mods at assembly). WindowInputGate is what
the decoder side calls back into through job.gate: may_stage says
whether a blocked job may take an input slot yet, may_start whether the
landed job may decode, mask_input folds the boundary into the landed
input once. The requester builds the
primary tier's job, asks the escalation policy which tiers decode the
window now, and enqueues one submission per tier on the DecodeQueue
port with each job's input send; a strong sibling's submission is the
strong tier's window side's, and a sibling that side holds for its
input has no submission to make here.
"""

import dataclasses
from typing import Callable, Optional

import decsim.decoders.decode_queue as decode_queue_module
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source


class WindowInputGate:
    """Whether a job may take a slot, may start, and what its input reads.

    The decoder side calls back into this through job.gate: may_stage
    says whether a boundary-blocked job may occupy an input slot yet,
    may_start whether the landed job may decode, mask_input folds the
    window's boundary into the landed input once. It is decoder-side
    policy about one window, so it is its own class rather than a second
    face of the builder. Trace source: copy_made(job, bits, memory_name,
    "masked view") when the mask is folded into a second copy of the
    landed input (data_path.md, correction 2 moves that copy behind the
    port); a tier that folds in place edits the unit's own memory and
    copies nothing (<tier>.boundary_fold, decoders/settings.py).
    """

    def __init__(self, planner, interaction, copies_the_fold: bool = True):
        self.planner = planner
        self.interaction = interaction
        # weak_decoder.boundary_fold: copy duplicates the landed input,
        # in_place XORs the mask into the unit's own memory
        self.copies_the_fold = copies_the_fold
        self.trace = _GateTraceSources()

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
        for dependency in window.deps:
            if not self._resolving_without_new_slots(dependency, visiting):
                return False
        return True

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

        With the copy fold the unit's stored rounds stay raw (cudaq-x
        keeps raw rounds and applies syndrome_mods at window assembly)
        and the job reads a masked duplicate; with the in-place fold the
        mask is written into the unit's own memory and nothing is
        duplicated.
        """
        window = job.window
        if window is None:
            return
        state = window.boundary_in
        if not state or job.decoder_input is None:
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
            self._fold_into_a_copy(job, masked_input)
            return
        self._fold_in_place(job, masked_input)

    def _fold_into_a_copy(
        self, job: decoding_records.DecodeJob, masked_input
    ) -> None:
        """The job reads a masked duplicate; the unit's rounds stay raw."""
        job.decoder_input = masked_input
        bits = _input_bit_count(masked_input)
        source_name = _input_source_name(job)
        self.trace.copy_made.fire(job, bits, source_name, "masked view")

    def _fold_in_place(
        self, job: decoding_records.DecodeJob, masked_input
    ) -> None:
        """The mask is written into the unit's own memory, for one reader.

        An input two jobs read has no single writer, so folding into it
        would change what the other job decodes; Helios keeps its shared
        memory single-writer (2301.08419 lines 632-640) and this refuses
        rather than rewriting the sibling's window.
        """
        memory = job.memory
        if memory is None:
            raise RuntimeError(
                f"{job.label}: boundary_fold in_place needs the unit's own "
                "copy of the rounds, and this tier reads its input in "
                "place (input: in_place); fold into a copy, or copy the "
                "input"
            )
        readers = memory.reader_count(job)
        if readers > 1:
            raise RuntimeError(
                f"{job.label}: boundary_fold in_place would rewrite an "
                f"input {readers} jobs read; one input has one writer "
                "(Helios 2301.08419 lines 632-640), so give this tier "
                "boundary_fold copy or one job per input"
            )
        job.decoder_input = memory.rewrite(job, masked_input)

    # ---- private

    def _resolving_without_new_slots(self, key: tuple, visiting: set) -> bool:
        if key in visiting:
            return False
        window = self.planner.windows_by_key.get(key)
        if window is None:
            return True
        if window.is_absorbed:
            return True
        if window.t_done is not None or window.service_began:
            return True
        if window.t_dispatch is None:
            return False
        visited = visiting | {key}
        for dependency in window.deps:
            if not self._resolving_without_new_slots(dependency, visited):
                return False
        return True

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
    source: window_data_complete(window) when the last round the window
    reads is readable in its store, which is the moment the window may
    be requested.
    """

    def __init__(self, engine, planner, tracker, interaction, gate) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.interaction = interaction
        self.gate = gate
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
    ) -> None:
        """Stamp the window's data-complete tick the first time it is seen.

        A trailing buffer satisfied by memory rounds alone releases on
        time with no syndrome content behind it; the log line keeps that
        approximation visible.
        """
        if window.t_data_complete is not None:
            return
        window.t_data_complete = self.engine.now
        self.trace.window_data_complete.fire(window)
        if not self.tracker.is_buffer_filled_by_memory(window):
            return
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
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
        receiving them again (design audit note 12 section 4.5).
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
        """One round's fragments in stable patch order, no boundary folded."""
        if fragments is None:
            return
        ordered = sorted(fragments, key=_fragment_patch_order)
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

    def __init__(
        self,
        tracker,
        retention,
        builder: DecodeRequestBuilder,
        decode_queue,
        escalation_policy,
        verdict,
        store_output,
        gap_join=None,
    ) -> None:
        self.tracker = tracker
        self.retention = retention
        self.builder = builder
        self.decode_queue = decode_queue
        self.escalation_policy = escalation_policy
        self.verdict = verdict
        # the primary store's outgoing port; it executes the input send
        self.store_output = store_output
        self.gap_join = gap_join

    def request_ready_windows(self, windows, strong_redecode) -> None:
        """Request each window that has its data, in the given order.

        strong_redecode is the strong tier's window side, which builds
        the strong sibling when the policy decodes both tiers at once;
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
        self.builder.note_data_complete(window, operation)
        if window.deps_remaining > 0 and not window.blocked_logged:
            # raw rounds ship now; the boundary is XORed into the landed
            # input at the decoder when it arrives (qLDPC net_error /
            # cudaq-x syndrome_mods / LILLIPUT's state register)
            window.blocked_logged = True
        self.request(window, operation, strong_redecode)

    def request(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        strong_redecode,
    ) -> None:
        """Build the primary job, ask the policy its tiers, enqueue each.

        The primary tier's job is the one built here; a strong tier the
        policy names too gets its sibling from the strong redecode (the
        paper's Step 1, both decoders started on the same window).
        """
        store = self.retention.primary_store
        self.builder.stamp_first_round(window, store)
        window.t_queued = self.builder.engine.now
        primary_tier = self.retention.primary_tier
        forced_classes = self._forced_logical_classes()
        first_class = _first_forced_class(forced_classes)
        job = self.builder.build(
            window, operation, primary_tier, store, first_class
        )
        window.queued = True
        tiers = self.escalation_policy.tiers_for_ready_window(window)
        primary_jobs = self._primary_jobs(job, forced_classes)
        is_input_held = self._bind_input_hold(primary_jobs, window.key)
        submissions = []
        for tier in tiers:
            if tier is primary_tier:
                primary = self._primary_submissions(primary_jobs, is_input_held)
                submissions.extend(primary)
            else:
                sibling = strong_redecode.parallel_strong_submission(job)
                started = _sibling_submissions(sibling)
                submissions.extend(started)
        for submission in submissions:
            self.enqueue(submission)

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
        (CUDA launches a whole grid in one call, cuda_guide.txt:
        1888-1892; OpenMP's primary thread creates the whole team,
        openmp_spec_5_2.txt:1400-1407).
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

    def _bind_input_hold(self, primary_jobs: list, key: tuple) -> bool:
        """Move the window's hold to the attempt; whether it was held already.

        The jobs of one attempt read the same rounds, so they share one
        hold and Buffer 0 may drop the rounds when the last of them has
        its input; a job whose rounds this side already holds keeps
        them and moves nothing.
        """
        first = primary_jobs[0]
        if first.submitted or self.retention.holds_input(first):
            return True
        store = self.retention.primary_store
        self.retention.bind_input_hold(first, key, store)
        reader_count = len(primary_jobs)
        if reader_count == 1:
            return False
        shared = _SharedInputHold(first.input_hold, reader_count)
        for job in primary_jobs:
            job.input_hold = shared.release
        return False

    def enqueue(self, submission: decoding_records.Submission) -> None:
        """Admit one submission with its input send and its return path.

        A strong job's result finalizes its window; a primary job's
        result commits it, through the confidence join when the window's
        answer takes both forced-class solves.
        """
        job = submission.job
        on_decoded = self.verdict.accept_result
        if job.strong_decode_for is not None:
            on_decoded = self.verdict.accept_strong_result
        elif self.gap_join is not None:
            on_decoded = self.gap_join.accept_result
        self.decode_queue.enqueue(job, submission.send_input, on_decoded)

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
        on).
        """
        self.decode_queue.withdraw_window(window.key)
        window.queued = False
        window.blocked_logged = False
        window.t_queued = None
        window.t_dispatch = None
        window.service_began = False

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: its parked decode may start."""
        self.decode_queue.release_parked(window_key)


class _SharedInputHold:
    """One store hold released when the last job that reads it has landed.

    The forced-class solves of one window read the same rounds, so
    Buffer 0 keeps them until every one of them has its input, the
    lifetime a zero-copy send owes its source (the kernel's dmaengine
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


def _sibling_submissions(sibling) -> list:
    """The strong sibling's submission, or none while its side holds it."""
    if sibling is None:
        return []
    return [sibling]


def _first_forced_class(forced_classes: tuple) -> Optional[int]:
    """The class the window's own job is built with; None when free."""
    if not forced_classes:
        return None
    return forced_classes[0]


def _input_source_name(job: decoding_records.DecodeJob) -> str:
    """Where the rounds the mask is folded out of sit."""
    memory = job.memory
    if memory is None:
        return job.input_source_name
    return memory.name


def _input_bit_count(decoder_input) -> Optional[int]:
    """The bits of a landed input; None when any fragment has no bits."""
    bit_count = 0
    for fragment in decoder_input.fragments():
        if fragment.bits is None:
            return None
        bit_count += len(fragment.bits)
    return bit_count


def _fragment_patch_order(fragment):
    return identity_records.stable_identity_order_key(fragment.patch_id)


@dataclasses.dataclass(frozen=True)
class _GateTraceSources:
    """Every event the window input gate reports, as one member."""

    copy_made: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decode request builder reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    window_data_complete: trace_source.TraceSource = trace_source.new_source()
