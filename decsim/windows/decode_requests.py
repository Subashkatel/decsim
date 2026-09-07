"""The decode requests: a job per complete window, asked for once.

A window's raw rounds ship as soon as its data is complete; a window
that still owes a boundary is masked at the decoder when its decode
starts (qLDPC folds the net error into the syndrome of the next window,
qldpc/decoders/sinter.py decode_shots_to_error; cudaq-x keeps raw
rounds and applies syndrome_mods at assembly). The builder is the job's
WindowInputGate: may_stage says whether a blocked job may take an input
slot yet, may_start whether the landed job may decode, mask_input folds
the boundary into the landed input once. The requester builds the
primary tier's job, asks the escalation policy which tiers decode the
window now, and enqueues one submission per tier on the DecodeQueue
port with each job's input send; a strong sibling's submission is the
strong tier's window side's.
"""

import dataclasses
import functools
from typing import Callable, Optional

import decsim.message as message
import decsim.observe.trace_source as trace_source
import decsim.records.identity as identity_records
import decsim.records.windows as window_records


class DecodeRequestBuilder:
    """Builds one request from a complete window; the job's input gate.

    Trace sources: copy_made(job, bits, memory_name, "masked view") when
    the boundary mask is folded into a second copy of the landed input
    (data_path.md, correction 2 moves that copy behind the port);
    window_data_complete(window) when the last round the window reads is
    readable in its store, which is the moment the window may be
    requested.
    """

    def __init__(
        self, engine, planner, tracker, interaction, transfers
    ) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.interaction = interaction
        self.transfers = transfers
        self.next_request_sequence = 0
        self.copy_made = trace_source.TraceSource()
        self.window_data_complete = trace_source.TraceSource()

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
        first_round_key = (window.op_id, window.start_round)
        window.t_first_round = store.publication_tick(first_round_key)

    def note_data_complete(
        self, window: window_records.Window, operation: message.Operation
    ) -> None:
        """Stamp the window's data-complete tick the first time it is seen.

        A trailing buffer satisfied by memory rounds alone releases on
        time with no syndrome content behind it; the log line keeps that
        approximation visible.
        """
        if window.t_data_complete is not None:
            return
        window.t_data_complete = self.engine.now
        self.window_data_complete.fire(window)
        if not self.tracker.is_buffer_filled_by_memory(window):
            return
        self.engine.log(
            "DecoderCluster",
            f"{operation.name} W{window.k} buffer filled by memory rounds "
            f"(time-only, no syndrome content)",
        )

    def build(
        self,
        window: window_records.Window,
        operation: message.Operation,
        tier: window_records.DecoderTier,
        store,
    ) -> message.DecodeJob:
        """The tier's decode job for one complete window, read from store."""
        request_key = self.new_request_key(window.op_id, window.k, tier)
        payloads = self.assemble_payloads(window, store)
        payload_round_count = message.distinct_round_count(payloads)
        round_count = (
            payload_round_count + window.batched_preceding_idle_round_count
        )
        spatial_nodes = self.planner.spatial_node_count_of(operation.id)
        geometry = self.planner.code_geometry_of(operation.id)
        model = self.planner.model_by_window.get(window.key)
        label = self._job_label(window, operation)
        return message.DecodeJob(
            op_id=window.op_id,
            window_id=window.k,
            n_rounds=round_count,
            ready_time=self.engine.now,
            spatial_nodes=spatial_nodes,
            payloads=payloads,
            dem=model,
            code=geometry.code_name,
            window=window,
            label=label,
            strong_label=f"strong({operation.name} W{window.k})",
            request_key=request_key,
            request_created_ticks=self.engine.now,
            gate=self,
        )

    def assemble_payloads(self, window: window_records.Window, store) -> list:
        """Collect this window's raw payloads, with successor overflow rounds.

        The boundary is never folded here: the mask is XORed into the
        landed input at the decoder when the decode starts.
        """
        operation_rounds = self.tracker.effective_round_count_for_window(
            window.op_id, window
        )
        end_round = min(window.buffer_hi, operation_rounds)
        window_info = window_records.WindowInfo.from_window(window)
        payloads = []
        stop_round = end_round + 1
        for round_index in range(window.start_round, stop_round):
            round_key = (window.op_id, round_index)
            fragments = store.retained_fragments(round_key)
            self._append_round_payloads(
                payloads, fragments, window_info, round_index
            )
        overflow = window.buffer_hi - operation_rounds
        if overflow <= 0:
            return payloads
        successor_ids = self.planner.successors_by_operation.get(
            window.op_id, []
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

    def input_send(
        self, job: message.DecodeJob, path: message.LinkPath
    ) -> Callable[[Callable[[], None]], int]:
        """The job's input send: at dispatch it rides the path into the unit.

        The payload bits are fixed now: the staging clears the job's
        payloads when the input lands.
        """
        payload_bits = job.payload_bits()
        return functools.partial(self._send_input, job, path, payload_bits)

    def resend_held_input(self) -> Callable[[Callable[[], None]], int]:
        """A job resubmitted after a withdrawal: its input lands at once."""
        return self._land_held_input

    # ---- the WindowInputGate port

    def may_stage(self, job: message.DecodeJob) -> bool:
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
        visiting = {(window.op_id, window.k)}
        for dependency in window.deps:
            if not self._resolving_without_new_slots(dependency, visiting):
                return False
        return True

    def may_start(self, job: message.DecodeJob) -> bool:
        """May this landed job start its decode?

        False parks the job in its slot until its window's last boundary
        arrives. Pure check: the mask itself is applied by mask_input at
        the actual start, so a re-check can never fold the boundary twice.
        """
        window = job.window
        if window is None:
            return True
        return window.deps_remaining <= 0

    def mask_input(self, job: message.DecodeJob) -> None:
        """XOR the window's boundary mask into the landed decoder input.

        The unit's stored copy stays raw (cudaq-x keeps raw rounds and
        applies syndrome_mods at window assembly); the job's input view is
        replaced with the masked rounds the decode will read.
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
        job.decoder_input = dataclasses.replace(
            job.decoder_input, rounds=tuple(masked_rounds)
        )
        bits = _input_bit_count(job.decoder_input)
        self.copy_made.fire(job, bits, job.memory.name, "masked view")

    # ---- private

    def _job_label(
        self, window: window_records.Window, operation: message.Operation
    ) -> str:
        if self.planner.is_windowed(window.op_id):
            return (
                f"{operation.name} W{window.k} "
                f"[commit {window.commit_lo}-{window.commit_hi}]"
            )
        body_rounds = self.tracker.round_count_for_window(operation.id, window)
        idle_rounds = window.batched_preceding_idle_round_count
        if idle_rounds:
            effective_rounds = window.n_rounds + idle_rounds
            return (
                f"{operation.name} [whole op, {effective_rounds} rounds: "
                f"{idle_rounds} idle + {body_rounds} body]"
            )
        return f"{operation.name} [whole op, {window.n_rounds} rounds]"

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

    def _send_input(
        self,
        job: message.DecodeJob,
        path: message.LinkPath,
        payload_bits: Optional[int],
        on_landed: Callable[[], None],
    ) -> int:
        return self.transfers.send_for_job(
            path, job, payload_bits=payload_bits, on_delivered=on_landed
        )

    def _land_held_input(self, on_landed: Callable[[], None]) -> int:
        return self.transfers.land_after(0, on_landed)

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


class DecodeRequester:
    """Requests a decode for every complete, unblocked window, once."""

    def __init__(
        self,
        tracker,
        retention,
        builder: DecodeRequestBuilder,
        decode_queue,
        escalation_policy,
        committer,
    ) -> None:
        self.tracker = tracker
        self.retention = retention
        self.builder = builder
        self.decode_queue = decode_queue
        self.escalation_policy = escalation_policy
        self.committer = committer

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
            self.builder.stamp_first_round(window, self.retention.weak_store)
        if not self.tracker.is_data_complete(window):
            return
        operation = self.tracker.operation_by_id[window.op_id]
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
        operation: message.Operation,
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
        job = self.builder.build(window, operation, primary_tier, store)
        window.queued = True
        tiers = self.escalation_policy.tiers_for_ready_window(window)
        submissions = []
        for tier in tiers:
            if tier is primary_tier:
                primary = message.Submission(job)
                submissions.append(primary)
            else:
                sibling = strong_redecode.parallel_strong_submission(job)
                submissions.append(sibling)
        for submission in submissions:
            self._submit(submission, window.key)

    def enqueue(self, submission: message.Submission) -> None:
        """Admit one submission with its input send and its return path.

        A strong job's result finalizes its window; a primary job's
        result commits it.
        """
        job = submission.job
        on_decoded = self.committer.accept_result
        if job.strong_decode_for is not None:
            on_decoded = self.committer.accept_strong_result
        self.decode_queue.enqueue(job, submission.send_input, on_decoded)

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

    def _submit(self, submission: message.Submission, key: tuple) -> None:
        """A strong submission carries its send; a weak one gets its input."""
        job = submission.job
        if job.strong_decode_for is not None:
            self.enqueue(submission)
            return
        if job.submitted or self.retention.holds_input(job):
            resend = self.builder.resend_held_input()
            submission = message.Submission(job, resend)
            self.enqueue(submission)
            return
        store = self.retention.primary_store
        self.retention.bind_input_hold(job, key, store)
        send_input = self.builder.input_send(
            job, self.retention.primary_input_path
        )
        submission = message.Submission(job, send_input)
        self.enqueue(submission)


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
