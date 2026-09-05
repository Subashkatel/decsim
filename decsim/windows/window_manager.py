"""The windows facade: the window life cycle of every operation.

Which windows exist is the WindowPlanner's, which rounds arrived and
whether a window has its data is the RoundTracker's; the facade
receives rounds, keeps the holds, requests decodes, commits results and
delivers each operation's result, the shape of gem5's cache (BaseCache
owns its MSHR queue, write buffer and tags, each one job, and implements
the ports: src/mem/cache/base.hh). Boundaries between windows are the
BoundaryCourier's, ownership of committed rounds is the LogicalLedger's,
the strong tier is StrongEscalation's (NoStrongTier when the policy
never escalates). One round reads as accept_window_input, check_window,
_submit_window_decode, on_decode_done, _commit_window.

Wide state recorded for the structural commits that split the rest into
retention, requester, committer and results (slice 5 design note,
section 3).
"""

import dataclasses
import functools
import types
from typing import Callable, Optional, Protocol, runtime_checkable

import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.strong_escalation as strong_escalation
import decsim.message as message
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.windowing_schemes as windowing_schemes


@runtime_checkable
class BoundaryPolicy(Protocol):
    """When a committed window may ship its boundary defects to dependents.

    Eager ships at every weak commit (the default); Held ships only once
    the result is final. A provisional boundary that ships is never
    revised, so serial switching (which can revise a result) needs Held.
    """

    def on_commit(self, window: message.Window, *, final: bool) -> bool:
        """Whether to ship the boundary now."""


@runtime_checkable
class ErrorModelProvider(Protocol):
    """Builds the decoder-facing models of windows and streams."""

    def register_dynamic_stream(
        self, stream_operation, round_count: int, *, fault_model_requirement
    ):
        """Note a dynamic stream; its window models come per window."""

    def validate_stream_length(
        self, stream_operation, stream_round_count: int
    ) -> None:
        """Refuse a stream longer than the source can supply."""

    def window_models_for_operation(
        self,
        operation,
        windows: list,
        round_count: int,
        *,
        fault_model_requirement,
        fault_exclusion_ranges: tuple,
        window_protocol,
    ) -> list:
        """One model per window of the operation."""

    def window_model_for_stream(self, stream_id, window):
        """The model of one window of a dynamic stream."""

    def strong_window_model_for_operation(
        self,
        operation,
        window,
        round_count: int,
        *,
        fault_model_requirement,
        exclude_faults_touching=None,
    ):
        """The model of one strong window of the operation."""


class WindowManager:
    """The windows facade: receives rounds, closes windows, commits results."""

    def __init__(
        self,
        engine,
        *,
        planner,
        tracker,
        links,
        conditional_release,
        boundary_policy,
        window_interaction,
        feedback_boundary_mode: str = "trailing_buffer",
        retain_strong_context: bool,
        syndrome_buffer: Optional[round_store_module.RoundStore] = None,
        syndrome_buffer_1: Optional[round_store_module.RoundStore] = None,
        pauli_frame=None,
        escalation_policy,
        submit_fn: Callable,
        check_strong_route: Callable,
        on_workload_complete: Callable[[], None],
    ):
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.links = links
        self.conditional_release = conditional_release
        self.pauli_frame = pauli_frame
        self.boundary_policy = boundary_policy
        self.window_interaction = window_interaction
        self.feedback_boundary_mode = feedback_boundary_mode
        self.retain_strong_context = retain_strong_context
        # jobs whose windows still owe a boundary at the decoder; the
        # boundary receive path releases them (wired by the root)
        self.release_service: Optional[Callable] = None
        self._next_decoder_request_sequence = 0
        self.escalation_policy = escalation_policy
        self.primary_tier = escalation_policy.primary_tier
        self._idle_decode_demand_receiver = None
        self.withdraw_decode = None  # wired to the decoder manager
        self.submit_fn = submit_fn  # (job, send_input) -> None
        self.escalation = strong_escalation.NoStrongTier()
        if retain_strong_context:
            self.escalation = strong_escalation.StrongEscalation(
                self, check_strong_route
            )
        self.on_workload_complete = on_workload_complete
        if syndrome_buffer is None:
            settings = round_store_settings.RoundStoreSettings()
            syndrome_buffer = round_store_module.RoundStore(settings)
        self.syndrome_buffer = syndrome_buffer
        strong_is_primary = self.primary_tier is message.DecoderTier.STRONG
        uses_strong_store = retain_strong_context or strong_is_primary
        if syndrome_buffer_1 is None and uses_strong_store:
            settings = round_store_settings.RoundStoreSettings()
            syndrome_buffer_1 = round_store_module.RoundStore(settings)
        self.syndrome_buffer_1 = syndrome_buffer_1
        self.result_by_operation: dict[int, tuple[int, ...]] = {}
        self.segment_results_sent: set = set()
        self._required_stream_end_by_operation_id: dict[int, int] = {}
        self._stream_binding_by_operation_id: dict[int, tuple] = {}
        self.ledger = committed_rounds.LogicalLedger()
        self.courier = window_boundaries.BoundaryCourier(self)
        self._finished_operation_ids: set[int] = set()
        self._workload_complete_sent = False
        # the committed prefix of each stream, so segments release once
        self._committed_round_count_by_stream: dict = {}
        for window in self.planner.windows_by_key.values():
            window_info = message.WindowInfo.from_window(window)
            window.boundary_in = self.window_interaction.initial_boundary_state(
                window_info
            )

    @property
    def windows(self) -> dict:
        """Every window by key; the planner's table."""
        return self.planner.windows_by_key

    # ---- the plan: operations, streams, windows and their holds

    def register_operation(self, operation: message.Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        is_new = self.tracker.register_operation(operation)
        if is_new:
            self.syndrome_buffer.open_operation(operation.id)

    def register_stream(self, stream_operation: message.Operation) -> None:
        """Register a stream whose windows are created at runtime."""
        resolved_feedback_mode = stream_operation.feedback_boundary_mode
        if resolved_feedback_mode is None:
            resolved_feedback_mode = self.feedback_boundary_mode
        stream_operation = dataclasses.replace(
            stream_operation, feedback_boundary_mode=resolved_feedback_mode
        )
        self.syndrome_buffer.open_operation(stream_operation.id)
        source_round_limit = self.planner.register_stream(stream_operation)
        self.tracker.register_stream(stream_operation, source_round_limit)

    def install_planned_holds(self, buffering_plan) -> None:
        """Check the stores against the plan and place its holds."""
        self._check_store_capacities(buffering_plan)
        self._register_planned_holds(buffering_plan)

    def _check_store_capacities(self, buffering_plan) -> None:
        """Refuse a store the yaml sized below the plan's longest hold."""
        strong_is_primary = self.primary_tier is message.DecoderTier.STRONG
        capacity = self.syndrome_buffer.settings.rounds
        minimum = buffering_plan.minimum_live_rounds
        if strong_is_primary:
            minimum = ()
        if capacity is not None and capacity < len(minimum):
            raise ValueError(
                f"upstream syndrome buffer needs {len(minimum)} packet slots, "
                f"got {capacity}"
            )
        if self.syndrome_buffer_1 is None:
            return
        strong_capacity = self.syndrome_buffer_1.settings.rounds
        strong_minimum = buffering_plan.sb1_minimum_live_rounds
        if strong_is_primary:
            # the plan's window reads live on the room-side store
            strong_minimum = buffering_plan.minimum_live_rounds
        if strong_capacity is not None and strong_capacity < len(
            strong_minimum
        ):
            raise ValueError(
                f"syndrome buffer 1 needs {len(strong_minimum)} packet "
                f"slots, got {strong_capacity}"
            )

    @property
    def primary_store(self):
        """The store the primary tier reads.

        Buffer 0 for the weak lane, syndrome buffer 1 for a strong-primary
        plan.
        """
        if self.primary_tier is message.DecoderTier.STRONG:
            return self.syndrome_buffer_1
        return self.syndrome_buffer

    def _register_planned_holds(self, plan) -> None:
        for owner, identities in plan.weak_holds:
            self.primary_store.register_hold(owner, identities)
        for owner, identities in plan.potential_holds:
            self.syndrome_buffer_1.register_hold(owner, identities)

    # ---- retention: which rounds each window holds in the two stores

    def _transfer_retention_hold(
        self, previous, replacement, store=None
    ) -> tuple:
        if store is None:
            store = self.syndrome_buffer
        keys = store.hold_round_identities(previous)
        store.transfer_hold(previous, replacement)
        return keys

    def _release_hold_if_live(self, owner, store=None) -> None:
        if store is None:
            store = self.syndrome_buffer
        if store.has_hold(owner):
            store.release_hold(owner)

    def _transfer_potential_to_pending(self, window_key, request_key) -> tuple:
        potential = message.PotentialStrong(window_key)
        pending = message.PendingStrong(request_key)
        return self._transfer_retention_hold(
            potential, pending, self.syndrome_buffer_1
        )

    def _add_window_read_refs(self, key: tuple, window: message.Window) -> None:
        """Register typed weak and possible-strong owners for a new window."""
        weak = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window
        )
        strong = self._strong_context_read_keys(window, weak)
        self.syndrome_buffer.register_hold(key, weak)
        if self.retain_strong_context:
            potential = message.PotentialStrong(key)
            held = weak + strong
            self.syndrome_buffer_1.register_hold(potential, held)

    def _read_keys_for_bounds(
        self,
        operation_id,
        start_round: int,
        buffer_hi: int,
        window: Optional[message.Window] = None,
    ) -> list:
        """Retained payload round keys for a possibly cross-operation range."""
        operation_rounds = self.tracker.effective_round_count_for_window(
            operation_id, window
        )
        last_local_round = min(buffer_hi, operation_rounds)
        stop_round = last_local_round + 1
        reads = []
        for round_index in range(start_round, stop_round):
            reads.append((operation_id, round_index))
        overflow = buffer_hi - operation_rounds
        if overflow <= 0:
            return reads
        successor_ids = self.planner.successors_by_operation.get(
            operation_id, []
        )
        overflow_stop = overflow + 1
        for successor_id in successor_ids:
            for round_index in range(1, overflow_stop):
                reads.append((successor_id, round_index))
        return reads

    def _strong_context_read_keys(
        self, window: message.Window, weak_reads: list
    ) -> list:
        """Rounds kept until the strong decoder is known to need them."""
        if not self.retain_strong_context:
            return []
        context_lo, _commit_lo, _commit_hi, context_hi = (
            self._strong_context_bounds(window)
        )
        weak = set(weak_reads)
        strong = self._read_keys_for_bounds(
            window.op_id, context_lo, context_hi, window
        )
        return [round_key for round_key in strong if round_key not in weak]

    def _replace_window_read_refs(
        self, key: tuple, window: message.Window
    ) -> None:
        """Move shrinking weak reads into strong retention before release."""
        weak = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window
        )
        strong = self._strong_context_read_keys(window, weak)
        potential = message.PotentialStrong(key)
        if (
            self.syndrome_buffer_1 is not None
            and self.syndrome_buffer_1.has_hold(potential)
        ):
            held = weak + strong
            self.syndrome_buffer_1.replace_hold(potential, held)
        self.syndrome_buffer.replace_hold(key, weak)

    def withdraw_window_decode(self, key: tuple) -> None:
        """Withdraw one window's early-shipped, unstarted decode.

        Its submission bookkeeping is reset so it can be resubmitted fresh
        (a strong window that absorbs the window owns its rounds from then
        on).
        """
        window = self.planner.windows_by_key[key]
        self.withdraw_decode(key)
        window.queued = False
        window.blocked_logged = False
        window.t_queued = None
        window.t_dispatch = None
        window.service_began = False

    # ---- the decoder manager's gates on a boundary-blocked job

    def stage_admission(self, job: message.DecodeJob) -> bool:
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

    def begin_service_gate(self, job: message.DecodeJob) -> bool:
        """May this landed job start its decode?

        False parks the job in its slot until its window's last boundary
        arrives. Pure check: the mask itself is applied by
        apply_service_boundary at the actual start, so a re-check can
        never fold the boundary twice.
        """
        window = job.window
        if window is None:
            return True
        return window.deps_remaining <= 0

    def apply_service_boundary(self, job: message.DecodeJob) -> None:
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
        window_info = message.WindowInfo.from_window(window)
        masked_rounds = []
        for round_input in job.decoder_input.rounds:
            masked = self._masked_round(state, window_info, round_input)
            masked_rounds.append(masked)
        job.decoder_input = dataclasses.replace(
            job.decoder_input, rounds=tuple(masked_rounds)
        )

    def _masked_round(self, state, window_info, round_input):
        fragments = []
        for fragment in round_input.fragments:
            masked = self.window_interaction.apply_boundary(
                state, window_info, fragment, round_input.round_index
            )
            fragments.append(masked)
        return dataclasses.replace(round_input, fragments=tuple(fragments))

    def _require_retained_payloads(
        self, round_keys: list, purpose: str, store=None
    ) -> None:
        """Reject a new consumer if any already-arrived input was released."""
        if store is None:
            store = self.syndrome_buffer
        arrived_for = self.tracker.rounds_arrived
        if store is not self.syndrome_buffer:
            arrived_for = self.tracker.strong_rounds_arrived
        missing = []
        for round_key in round_keys:
            if _is_released(store, arrived_for, round_key):
                missing.append(round_key)
        if missing:
            raise RuntimeError(
                f"{purpose} requires retained payload rounds that are no "
                f"longer available: {missing}"
            )

    # ---- dynamic streams: their windows are planned as rounds arrive

    def _update_stream(self, operation_id) -> None:
        """Grow (and maybe seal) the stream as one more round arrives."""
        if not self.planner.has_stream(operation_id):
            return
        highest_known_round = self.tracker.rounds_arrived(operation_id)
        self._grow_stream(operation_id, highest_known_round, None)
        if self.tracker.is_sealed(operation_id):
            return
        if self.tracker.reaches_source_limit(operation_id):
            limit = self.tracker.source_round_limit(operation_id)
            self.seal_stream(operation_id, limit)

    def _grow_stream(
        self, stream_id, highest_known_round: int, round_cap: Optional[int]
    ) -> None:
        """Plan every window whose commit region has begun, and admit it."""
        if self.tracker.is_sealed(stream_id):
            return
        created = self.planner.grow_stream(
            stream_id, highest_known_round, round_cap
        )
        for window in created:
            self._admit_stream_window(window)

    def _admit_stream_window(self, window: message.Window) -> None:
        """Connect one new stream window: its boundary, its model, its holds.

        If the previous boundary already arrived, apply it immediately.
        """
        window_info = message.WindowInfo.from_window(window)
        window.boundary_in = self.window_interaction.initial_boundary_state(
            window_info
        )
        if window.k > 0:
            previous_key = (window.op_id, window.k - 1)
            self._link_to_previous_window(previous_key, window.key, window)
        self.planner.attach_stream_model(window)
        self._add_window_read_refs(window.key, window)

    def _link_to_previous_window(
        self, previous_key: tuple, key: tuple, window: message.Window
    ) -> None:
        """A shipped boundary merges now; a held one is not available yet."""
        if self.courier.has_committed(previous_key):
            boundary = self.courier.committed(previous_key)
            self.courier.merge_available(previous_key, window, boundary)
            return
        window.deps.append(previous_key)
        window.deps_remaining = 1
        previous = self.planner.windows_by_key[previous_key]
        previous.dependents.append(key)

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        """Close a dynamic stream once its full length has arrived."""
        if self.tracker.is_sealed(stream_id):
            return
        self.tracker.check_stream_length(stream_id, stream_round_count)
        self._grow_stream(stream_id, stream_round_count, stream_round_count)
        if not self.planner.is_finite_stream(stream_id):
            clipped = self.planner.trim_stream_tail(
                stream_id, stream_round_count
            )
            if clipped is not None:
                self._reset_stream_window_reads(clipped)
        self.tracker.seal(stream_id, stream_round_count)
        self.check_windows_for_operation(stream_id)
        self.finish_workload_if_ready()

    def _reset_stream_window_reads(self, window: message.Window) -> None:
        """After clipping a live tail, retain only the weak commit range."""
        stop_round = window.commit_hi + 1
        new_reads = []
        for round_index in range(window.start_round, stop_round):
            new_reads.append((window.op_id, round_index))
        new_reads.sort()
        self.syndrome_buffer.replace_hold(window.key, new_reads)

    def close_stream_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed boundary."""
        if not self.planner.has_stream(stream_id):
            return
        self.tracker.close_boundary(stream_id, stream_round_count)
        self._grow_stream(stream_id, stream_round_count, None)
        self._refresh_unqueued_stream_windows(stream_id)
        self.check_windows_for_operation(stream_id)

    def _refresh_unqueued_stream_windows(self, stream_id) -> None:
        """Refresh retained reads once a stream boundary closes the tail."""
        for window in self.planner.windows_of(stream_id):
            if window.queued or window.committed:
                continue
            self._replace_window_read_refs(window.key, window)

    def has_dynamic_stream(self, stream_id) -> bool:
        """True for a stream whose windows are planned at runtime."""
        return self.planner.has_stream(stream_id)

    # ---- arrivals: the WindowInput port and the room-side store's signal

    def accept_window_input(self, packet: message.SyndromeRoundPacket) -> None:
        """Publish one stored upstream round to window readiness.

        Assembly-to-retention is a state transition on the same
        allocation; no decoder input moves here. Window input transfer
        begins only when a decode request is admitted.
        """
        operation = self.tracker.operation_by_id[packet.operation_id]
        self._refuse_unplanned_round(packet, operation)
        if self.primary_tier is not message.DecoderTier.STRONG:
            # Buffer 0 publication is the readiness authority for the weak lane
            self._count_arrival(operation, packet.round_index)
            self._update_stream(operation.id)
            self.escalation.after_arrival(operation.id)
            self._wake_windows(operation)
        # a round whose every consumer already resolved (an absorbed window's
        # tail, or every round of a strong-primary plan) frees its Buffer 0
        # slot on arrival, the same drop-on-arrival rule syndrome buffer 1
        # applies
        round_key = (packet.operation_id, packet.round_index)
        self.syndrome_buffer.release_round_if_unheld(round_key)

    def _refuse_unplanned_round(
        self, packet: message.SyndromeRoundPacket, operation: message.Operation
    ) -> None:
        """A round past the plan is the device's mistake (it is a plug-in)."""
        if not self.syndrome_buffer.has_operation(operation.id):
            raise RuntimeError(
                f"round {packet.round_index} of {operation.name} arrived "
                f"after the op's last window committed and its syndrome RAM "
                f"was freed. The device emitted more rounds than the "
                f"execution plan expects."
            )
        round_limit = self.tracker.arrival_round_limit(operation.id)
        if round_limit is not None and packet.round_index > round_limit:
            raise ValueError(
                f"round {packet.round_index} of {operation.name} exceeds the "
                f"device round limit {round_limit}"
            )

    def accept_feedback_memory_round(self, source_operation_id) -> None:
        """Record one idle or memory round and re-check waiting windows."""
        memory_rounds = self.tracker.note_memory_round(source_operation_id)
        operation = self.tracker.operation_by_id[source_operation_id]
        self.engine.log(
            "DecoderCluster",
            f"memory round for {operation.name} "
            f"(idle buffer rounds: {memory_rounds})",
        )
        self.check_windows_for_operation(source_operation_id)

    def accept_room_round(self, operation_id, round_index: int) -> None:
        """Syndrome buffer 1 stored a round.

        Wake the strong tier's listeners, and drive readiness when the
        strong tier is primary.
        """
        self.tracker.note_room_round(operation_id, round_index)
        self.escalation.after_arrival(operation_id)
        if self.primary_tier is not message.DecoderTier.STRONG:
            return
        operation = self.tracker.operation_by_id[operation_id]
        stored_through = self.tracker.strong_rounds_arrived(operation_id)
        self._count_arrival(operation, stored_through)
        self._update_stream(operation_id)
        self._wake_windows(operation)

    def _count_arrival(
        self, operation: message.Operation, round_index: int
    ) -> None:
        """Advance the readiness arrival counter; the authority calls this."""
        arrived_now = self.tracker.note_arrival(operation.id, round_index)
        self.engine.log(
            "DecoderCluster",
            f"round {round_index} of {operation.name} arrived "
            f"(op now has rounds 1..{arrived_now})",
        )

    def _wake_windows(self, operation: message.Operation) -> None:
        self.check_windows_for_operation(operation.id)
        for predecessor_id in operation.decoder_boundary_predecessors:
            self.check_windows_for_operation(predecessor_id)

    def prepend_idle_rounds(self, operation_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation."""
        self.planner.prepend_idle_rounds(operation_id, round_count)

    # ---- readiness: a window with its data and no owed boundary is requested

    def check_windows_for_operation(self, operation_id: int) -> None:
        """Check every window of the operation, in index order."""
        window_count = self.planner.window_count_of(operation_id)
        for window_index in range(window_count):
            self.check_window((operation_id, window_index))

    def check_window(self, key: tuple) -> None:
        """If a window has its data, submit it through the escalation policy."""
        window = self.planner.windows_by_key[key]
        if window.queued or window.committed:
            return
        self._stamp_first_round_if_arrived(window)
        if not self.tracker.is_data_complete(window):
            return
        operation = self.tracker.operation_by_id[window.op_id]
        if window.t_data_complete is None:
            window.t_data_complete = self.engine.now
            self._note_memory_filled_buffer(window, operation)
        if window.deps_remaining > 0 and not window.blocked_logged:
            # raw rounds ship now; the boundary is XORed into the landed
            # input at the decoder when it arrives (qLDPC net_error /
            # cudaq-x syndrome_mods / LILLIPUT's state register)
            window.blocked_logged = True
        self._submit_window_decode(key, window, operation)

    def _stamp_first_round_if_arrived(self, window: message.Window) -> None:
        if window.t_first_round is not None:
            return
        if not self.tracker.has_first_round(window):
            return
        first_round_key = (window.op_id, window.start_round)
        window.t_first_round = self.syndrome_buffer.publication_tick(
            first_round_key
        )

    def _note_memory_filled_buffer(
        self, window: message.Window, operation: message.Operation
    ) -> None:
        """Flag a trailing buffer satisfied by memory rounds alone.

        Such a window releases on time with no syndrome content behind it;
        the flag keeps that approximation visible wherever the window's
        result is read.
        """
        readiness = self.tracker.readiness(window)
        if not windowing_schemes.buffer_filled_by_memory_only(
            window, readiness
        ):
            return
        self.engine.log(
            "DecoderCluster",
            f"{operation.name} W{window.k} buffer filled by memory rounds "
            f"(time-only, no syndrome content)",
        )

    # ---- the request: build the job, ask the policy, enqueue

    def _submit_window_decode(
        self, key: tuple, window: message.Window, operation: message.Operation
    ) -> None:
        """Build the weak job, ask the policy, enqueue its submissions."""
        self._stamp_first_round_tick(window, self.primary_store)
        window.t_queued = self.engine.now
        job = self._window_job(key, window, operation)
        window.queued = True
        submissions = self.escalation_policy.on_window_ready(
            window, job, self.escalation
        )
        for submission in submissions:
            self._submit(submission, key)

    def _window_job(
        self, key: tuple, window: message.Window, operation: message.Operation
    ) -> message.DecodeJob:
        """The primary tier's decode job for one complete window."""
        request_key = self._new_request_key(
            window.op_id, window.k, self.primary_tier
        )
        payloads = self._assemble_payloads(window, self.primary_store)
        payload_round_count = decoder_memory.count_decoder_input_round_demand(
            payloads
        )
        round_count = (
            payload_round_count + window.batched_preceding_idle_round_count
        )
        spatial_nodes = self.planner.spatial_node_count_of(operation.id)
        geometry = self.planner.code_geometry_of(operation.id)
        model = self.planner.model_by_window.get(key)
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
        )

    def _job_label(
        self, window: message.Window, operation: message.Operation
    ) -> str:
        """The decode job's log label."""
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

    def _submit(self, submission: message.Submission, key: tuple) -> None:
        """Enqueue one submission.

        A strong job goes through the escalation; a weak job carries its
        input send.
        """
        job = submission.job
        if job.strong_decode_for is not None:
            if submission.delay_ticks != 0:
                raise ValueError(
                    "strong transport delay is owned by the link fabric"
                )
            self.escalation.submit_strong(job)
            return
        if self._holds_input_already(job):
            resend = functools.partial(
                self._resend_held_input, submission.delay_ticks
            )
            self.submit_fn(job, resend)
            return
        self._bind_decoder_input_hold(job, key, self.primary_store)
        payload_bits = self._job_payload_bits(job)
        input_path = self._primary_input_path()
        send_input = functools.partial(
            self._send_window_input,
            job,
            payload_bits,
            input_path,
            submission.delay_ticks,
        )
        self.submit_fn(job, send_input)

    def _holds_input_already(self, job: message.DecodeJob) -> bool:
        """A job resubmitted after a withdrawal keeps the input it holds."""
        if job.submitted:
            return True
        if job.request_key is None:
            return False
        owner = message.DecoderInputHold(job.request_key)
        return self.syndrome_buffer.has_hold(owner)

    def _resend_held_input(self, delay_ticks: int, on_landed) -> int:
        return self._land_after(delay_ticks, on_landed)

    def _primary_input_path(self) -> message.LinkPath:
        """The link the primary tier's input rides into its unit."""
        if self.primary_tier is message.DecoderTier.WEAK:
            return message.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
        return message.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER

    def _send_window_input(
        self,
        job: message.DecodeJob,
        payload_bits: Optional[int],
        input_path: message.LinkPath,
        extra_delay_ticks: int,
        on_landed: Callable[[], None],
    ) -> int:
        """Move the window from the primary store into the unit's memory.

        Called once the unit is assigned; the input rides the primary
        tier's input link.
        """
        land = functools.partial(self._land_after, extra_delay_ticks, on_landed)
        expected_delay_ticks = self._send_job_transfer(
            input_path, job, payload_bits=payload_bits, on_delivered=land
        )
        return expected_delay_ticks + extra_delay_ticks

    def _stamp_first_round_tick(
        self, window: message.Window, store=None
    ) -> None:
        """Retain arrival provenance for latency accounting."""
        if store is None:
            store = self.syndrome_buffer
        if window.t_first_round is not None:
            return
        first_round_key = (window.op_id, window.start_round)
        window.t_first_round = store.publication_tick(first_round_key)

    def _new_request_key(
        self, operation_id, window_id: int, tier: message.DecoderTier
    ) -> message.DecoderRequestKey:
        request_key = message.DecoderRequestKey(
            operation_id, window_id, tier, self._next_decoder_request_sequence
        )
        self._next_decoder_request_sequence += 1
        return request_key

    def _bind_decoder_input_hold(
        self, job: message.DecodeJob, previous_owner, store=None
    ) -> None:
        """Transfer upstream retention to an admitted input request.

        The release runs only after decoder memory materialization, so
        overlapping rounds remain upstream until their last consumer
        transfer.
        """
        if store is None:
            store = self.syndrome_buffer
        owner = message.DecoderInputHold(job.request_key)
        if previous_owner != owner:
            _move_hold_to_input(store, previous_owner, owner, job)
        job.input_hold = functools.partial(store.release_hold, owner)

    def _assemble_payloads(self, window: message.Window, store=None) -> list:
        """Collect this window's raw payloads, with successor overflow rounds.

        The boundary is never folded here: the mask is XORed into the
        landed input at the decoder when the decode starts.
        """
        if store is None:
            store = self.syndrome_buffer
        operation_rounds = self.tracker.effective_round_count_for_window(
            window.op_id, window
        )
        end_round = min(window.buffer_hi, operation_rounds)
        window_info = message.WindowInfo.from_window(window)
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
            payload = self.window_interaction.apply_boundary(
                None, window_info, fragment, round_index
            )
            payloads.append(payload)

    @staticmethod
    def _job_payload_bits(job: message.DecodeJob) -> Optional[int]:
        payloads = job.payloads or ()
        sizes = []
        for payload in payloads:
            sizes.append(payload.size_bits)
        for size in sizes:
            if size is None:
                return None
        return sum(sizes)

    def _land_after(
        self, delay_ticks: int, on_landed: Callable[[], None]
    ) -> int:
        """Land an input that rides no link: now, or after a fixed delay."""
        if delay_ticks == 0:
            on_landed()
            return 0
        self.engine.schedule(delay_ticks, on_landed, label="held input lands")
        return delay_ticks

    # ---- sending in a window's or a job's name

    @staticmethod
    def _job_attribution(
        job: message.DecodeJob, request_key: message.DecoderRequestKey
    ) -> message.TransferAttribution:
        payloads = job.payloads or ()
        patches = {}
        for payload in payloads:
            patch_id = payload.patch_id
            order_key = message.stable_identity_order_key(patch_id)
            patches[order_key] = patch_id
        ordered_keys = sorted(patches)
        patch_ids = tuple(patches[key] for key in ordered_keys)
        window = job.window
        assert window is not None, (
            "window-scoped transport requires a DecodeJob window"
        )
        first_round, last_round = _read_range(window)
        relation = message.RequestTransferRelation(request_key)
        return message.TransferAttribution(
            operation_id=job.op_id,
            patch_ids=patch_ids,
            window_id=job.window_id,
            first_round=first_round,
            last_round=last_round,
            relation=relation,
        )

    @staticmethod
    def _window_attribution(
        window: message.Window,
        operation: message.Operation,
        request_key: message.DecoderRequestKey,
    ) -> message.TransferAttribution:
        ordered_patches = sorted(
            operation.patches, key=message.stable_identity_order_key
        )
        first_round, last_round = _read_range(window)
        relation = message.RequestTransferRelation(request_key)
        return message.TransferAttribution(
            operation_id=operation.id,
            patch_ids=tuple(ordered_patches),
            window_id=window.k,
            first_round=first_round,
            last_round=last_round,
            relation=relation,
        )

    def _send_window_transfer(
        self,
        path: message.LinkPath,
        window: message.Window,
        operation: message.Operation,
        request_key: message.DecoderRequestKey,
        payload_bits: Optional[int],
        on_delivered: Callable[[], None],
    ) -> None:
        """Send in a window's name; on_delivered runs at the delivery."""
        attribution = self._window_attribution(window, operation, request_key)
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.links.send(
            path, payload_bits, self.engine.now, attribution, delivered
        )

    @staticmethod
    def _result_payload_bits(
        result: message.DecodeResult, operation: message.Operation
    ) -> int:
        """A result reaches the frame as one bit per logical observable.

        A timing-only result stands for one observable per patch.
        """
        if result.logical_observables is not None:
            return len(result.logical_observables)
        patch_count = len(operation.patches)
        return max(1, patch_count)

    def _send_job_transfer(
        self,
        path: message.LinkPath,
        job: message.DecodeJob,
        *,
        payload_bits: Optional[int],
        request_key: Optional[message.DecoderRequestKey] = None,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send in a job's name; on_delivered runs at the delivery.

        Returns the delay the link expects, a scheduler's estimate.
        """
        relation_key = request_key
        if relation_key is None:
            relation_key = job.request_key
        attribution = self._job_attribution(job, relation_key)
        now_ticks = self.engine.now
        expected_delay_ticks = self.links.expected_delay_ticks(
            path, payload_bits, now_ticks
        )
        delivered = functools.partial(_run_at_delivery, on_delivered)
        self.links.send(path, payload_bits, now_ticks, attribution, delivered)
        return expected_delay_ticks

    @staticmethod
    def _strong_context_bounds(window: message.Window) -> tuple:
        """(context_lo, commit_lo, commit_hi, context_hi) of a strong redo."""
        buffer_span = window.buffer_hi - window.commit_hi
        buffer_rounds = max(0, buffer_span)
        context_start = window.commit_lo - buffer_rounds
        context_lo = max(1, context_start)
        context_hi = window.commit_hi + buffer_rounds
        return context_lo, window.commit_lo, window.commit_hi, context_hi

    # ---- the result: the boundary leaves at decode done, the frame commits

    def on_decode_done(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Hand the boundary on at decode done; publish after the output link.

        The boundary is decoder state (residual defects at the commit
        edge), so it leaves for the dependent windows at decode done, the
        way Skoric's blocks, LILLIPUT's state register and qLDPC's
        net_error do; the frame commit downstream never gates the next
        window.
        """
        window = self.planner.windows_by_key[(job.op_id, job.window_id)]
        window.t_done = self.engine.now
        if job.awaiting_strong_result:
            self._commit_decode_done(job, result)
            return
        operation = self.tracker.operation_by_id[job.op_id]
        self._hand_on_boundary(job, result, window, operation)
        # the result rides its tier's output link home: WDO for the weak
        # tier, DO for a strong-primary decode
        output_path = message.LinkPath.WEAK_DECODER_TO_FRAME
        if job.request_key.tier is not message.DecoderTier.WEAK:
            output_path = message.LinkPath.STRONG_DECODER_TO_FRAME
        payload_bits = self._result_payload_bits(result, operation)
        sink = functools.partial(self._sink_weak_correction, job, result)
        self._send_window_transfer(
            output_path, window, operation, job.request_key, payload_bits, sink
        )

    def _sink_weak_correction(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Charge and install one final correction before committing it."""
        if self.pauli_frame is None:
            self._commit_decode_done(job, result)
            return
        commit = functools.partial(self._commit_decode_done, job, result)
        self.pauli_frame.commit_correction(
            window_key=(job.op_id, job.window_id),
            logical_observables=result.logical_observables,
            request_key=job.request_key,
            on_committed=commit,
        )

    def _commit_decode_done(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Commit after the weak result's transport, or provisionally."""
        key = (job.op_id, job.window_id)
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[job.op_id]
        self._commit_window(result, key, window, operation)
        is_final = not job.awaiting_strong_result
        if is_final:
            window.published_request_key = job.request_key
        self._update_committed_round_count(operation.id)
        if job.awaiting_strong_result:
            # provisional: the boundary leaves with the commit
            self._hand_on_boundary(job, result, window, operation)
        if is_final and self.syndrome_buffer_1 is not None:
            potential = message.PotentialStrong(key)
            self._release_hold_if_live(potential, self.syndrome_buffer_1)
        self.escalation.after_weak_commit(key)
        self._finish_operation_if_ready(operation)
        self.finish_workload_if_ready()

    def _hand_on_boundary(
        self,
        job: message.DecodeJob,
        result: message.DecodeResult,
        window: message.Window,
        operation: message.Operation,
    ) -> None:
        """Ship the boundary to dependent windows, or hold it until final."""
        boundary = self.window_interaction.boundary_from_result(result, None)
        is_final = not job.awaiting_strong_result
        if self.boundary_policy.on_commit(window, final=is_final):
            self.courier.send(
                window, operation, boundary, source_request_key=job.request_key
            )
            return
        held = window_boundaries.HeldBoundary(
            job.request_key, operation.id, boundary
        )
        self.courier.hold((job.op_id, job.window_id), held)

    def rounds_backlog(self) -> tuple:
        """Rounds arrived but not decoded, per operation in stable order.

        A row is (operation id, patch, rounds arrived past the unbroken
        decoded prefix from round 1).
        """
        rows = []
        ordered_ids = sorted(
            self.tracker.operation_by_id, key=message.stable_identity_order_key
        )
        for operation_id in ordered_ids:
            operation = self.tracker.operation_by_id[operation_id]
            decoded = self._committed_prefix_round_count(operation_id)
            arrived = self.tracker.rounds_arrived(operation_id)
            undecoded = arrived - decoded
            waiting = max(0, undecoded)
            patch = _representative_patch(operation)
            rows.append((operation_id, patch, waiting))
        return tuple(rows)

    def _committed_prefix_round_count(self, operation_id) -> int:
        """Rounds decoded in an unbroken prefix from round 1."""
        committed_ranges = []
        for window in self._committed_windows_of(operation_id):
            committed_ranges.append((window.commit_lo, window.commit_hi))
        committed_ranges.sort()
        decoded = 0
        for start_round, end_round in committed_ranges:
            if start_round > decoded + 1:
                break
            decoded = max(decoded, end_round)
        return decoded

    def _commit_window(
        self,
        result: message.DecodeResult,
        key: tuple,
        window: message.Window,
        operation: message.Operation,
    ) -> None:
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
        existing = self.ledger.get(key)
        if existing is None or existing.ownership_kind != "strong_window":
            contribution = message.LogicalContribution(
                owner_key=key,
                commit_lo=window.commit_lo,
                commit_hi=window.commit_hi,
                ownership_kind="ordinary_window",
                logical_observables=result.logical_observables,
            )
            self.ledger.install(contribution)

    def on_strong_decode_done(
        self, completion: message.StrongDecodeCompletion
    ) -> None:
        """Publish an accepted strong result only after its DO transfer."""
        key = (
            completion.request_key.operation_id,
            completion.request_key.window_id,
        )
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[window.op_id]
        payload_bits = self._result_payload_bits(completion.result, operation)
        commit = functools.partial(self._commit_strong_decode_done, completion)
        self._send_window_transfer(
            message.LinkPath.STRONG_DECODER_TO_FRAME,
            window,
            operation,
            completion.request_key,
            payload_bits,
            commit,
        )

    def _commit_strong_decode_done(
        self, completion: message.StrongDecodeCompletion
    ) -> None:
        """Finalize a weak-committed window with the delivered strong result.

        A held boundary ships now that the result is final.
        """
        key = (
            completion.request_key.operation_id,
            completion.request_key.window_id,
        )
        result = completion.result
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[window.op_id]
        if self.pauli_frame is None:
            self._finish_strong_commit(
                completion, key, result, window, operation
            )
            return
        # the strong result is this window's FINAL correction: it folds into
        # the frame like any final result (the provisional weak bypassed it),
        # and the priced frame write gates the rest of the commit
        finish = functools.partial(
            self._finish_strong_commit,
            completion,
            key,
            result,
            window,
            operation,
        )
        self.pauli_frame.commit_correction(
            window_key=key,
            logical_observables=result.logical_observables,
            request_key=completion.request_key,
            on_committed=finish,
        )

    def _finish_strong_commit(
        self,
        completion: message.StrongDecodeCompletion,
        key: tuple,
        result: message.DecodeResult,
        window: message.Window,
        operation: message.Operation,
    ) -> None:
        if result.logical_observables is not None:
            self.ledger.replace_prediction(key, result.logical_observables)
        # the strong result is the window's final one: its status is now
        # published, and nothing of the operation waits on it any more
        window.published_request_key = completion.request_key
        held = self.courier.take_held(key)  # Held: ship now
        if held is not None:
            boundary = self.window_interaction.boundary_from_result(
                result, held.boundary
            )
            held_operation = self.tracker.operation_by_id[held.operation_id]
            self.courier.send(
                window,
                held_operation,
                boundary,
                source_request_key=completion.request_key,
            )
        committed_round_count = self._committed_round_count_by_stream.get(
            operation.id, 0
        )
        self.release_stream_segments_at_commit(
            operation.id, committed_round_count
        )
        self._finish_operation_if_ready(operation)
        self.finish_workload_if_ready()

    def _window_infos(self):
        infos = {}
        for key, window in self.planner.windows_by_key.items():
            infos[key] = message.WindowInfo.from_window(window)
        return types.MappingProxyType(infos)

    # ---- the operation's result: delivered once every window is final

    def _finish_operation_if_ready(self, operation: message.Operation) -> None:
        """Deliver the result once every window is final.

        Every window committed, no strong redo pending, the stream sealed.
        """
        if operation.id in self._finished_operation_ids:
            return
        if self._has_window_awaiting_strong(operation.id):
            return
        if self.syndrome_buffer.has_live_operation_reference(operation.id):
            return
        if self._strong_store_references(operation.id):
            return
        committed = self._committed_windows_of(operation.id)
        if len(committed) != self.planner.window_count_of(operation.id):
            return
        if not self.tracker.is_sealed(operation.id):
            return
        self._finished_operation_ids.add(operation.id)
        self._deliver_result(operation)
        self.syndrome_buffer.close_operation(operation.id)
        self._close_strong_store_operation(operation.id)

    def _strong_store_references(self, operation_id) -> bool:
        if self.syndrome_buffer_1 is None:
            return False
        return self.syndrome_buffer_1.has_live_operation_reference(operation_id)

    def _close_strong_store_operation(self, operation_id) -> None:
        if self.syndrome_buffer_1 is None:
            return
        if self.syndrome_buffer_1.has_operation(operation_id):
            self.syndrome_buffer_1.close_operation(operation_id)

    def finish_workload_if_ready(self) -> None:
        """Tell the root once every window of the workload is final."""
        if self._workload_complete_sent:
            return
        if self._committed_window_count() != self.planner.total_windows:
            return
        if self._has_window_awaiting_strong(None):
            return
        if self.tracker.has_unsealed_streams():
            return
        if self.on_workload_complete is None:
            return
        self._workload_complete_sent = True
        self.on_workload_complete()

    def _deliver_result(self, operation: message.Operation) -> None:
        windows = self.planner.windows_of(operation.id)
        commit_los = [window.commit_lo for window in windows]
        commit_his = [window.commit_hi for window in windows]
        commit_lo = min(commit_los)
        commit_hi = max(commit_his)
        logical_observables = self.ledger.observables_for_interval(
            operation.id, commit_lo, commit_hi, boundary_policy="strict"
        )
        self._record_result(operation.id, logical_observables)
        self.conditional_release.release_waiters(operation)

    def _record_result(self, operation_id, logical_observables) -> None:
        if logical_observables is None:
            self.result_by_operation.pop(operation_id, None)
            return
        self.result_by_operation[operation_id] = logical_observables

    def release_stream_segments_at_commit(
        self, stream_id, committed_round_count: int
    ) -> None:
        """Deliver the segment results whose full round range committed.

        Gated as operations are: no pending strong may still change it.
        """
        operations = self.tracker.operation_by_id.values()
        operations = list(operations)
        for operation in operations:
            self._release_segment_if_committed(
                operation, stream_id, committed_round_count
            )

    def _release_segment_if_committed(
        self,
        operation: message.Operation,
        stream_id,
        committed_round_count: int,
    ) -> None:
        binding = self._stream_binding_by_operation_id.get(operation.id)
        operation_stream_id = operation.stream_id
        stream_offset = operation.stream_offset
        if binding is not None:
            operation_stream_id, stream_offset = binding
        if operation_stream_id != stream_id:
            return
        if operation.id not in self.tracker.blocking_operation_ids:
            return
        if operation.id in self.segment_results_sent:
            return
        segment_end = self._stream_segment_end(operation)
        if segment_end is None or segment_end > committed_round_count:
            return
        if self._segment_waits_for_strong(stream_id, segment_end):
            return
        segment_start = stream_offset + 1
        logical_observables = self.ledger.observables_for_interval(
            stream_id,
            segment_start,
            segment_end,
            boundary_policy="stream_segment",
        )
        self._record_result(operation.id, logical_observables)
        self.segment_results_sent.add(operation.id)
        self.conditional_release.release_waiters(operation)

    def _update_committed_round_count(self, stream_id) -> None:
        """Advance the stream's committed prefix; release what it covers."""
        committed = self._committed_prefix_round_count(stream_id)
        cached = self._committed_round_count_by_stream.get(stream_id, 0)
        if committed <= cached:
            return
        self._committed_round_count_by_stream[stream_id] = committed
        self.release_stream_segments_at_commit(stream_id, committed)

    def _segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        for key, window in self.planner.windows_by_key.items():
            if key[0] != stream_id:
                continue
            if not _is_awaiting_strong(window):
                continue
            if window.commit_lo <= segment_end:
                return True
        return False

    def _committed_windows_of(self, operation_id) -> list:
        """The operation's committed windows, absorbed ones included."""
        committed = []
        for key, window in self.planner.windows_by_key.items():
            if key[0] != operation_id:
                continue
            if window.committed:
                committed.append(window)
        return committed

    def _committed_window_count(self) -> int:
        count = 0
        for window in self.planner.windows_by_key.values():
            if window.committed:
                count += 1
        return count

    def _has_window_awaiting_strong(self, operation_id) -> bool:
        """Whether a window of the operation (or of any) awaits its redo."""
        for key, window in self.planner.windows_by_key.items():
            if operation_id is not None and key[0] != operation_id:
                continue
            if _is_awaiting_strong(window):
                return True
        return False

    def _stream_segment_end(
        self, operation: message.Operation
    ) -> Optional[int]:
        required_end = self._required_stream_end_by_operation_id.get(
            operation.id
        )
        if required_end is not None:
            return required_end
        binding = self._stream_binding_by_operation_id.get(operation.id)
        stream_offset = operation.stream_offset
        if binding is not None:
            stream_offset = binding[1]
        if stream_offset is None:
            return None
        round_count = self.planner.round_count_of(operation.id)
        return stream_offset + round_count

    # ---- what the controller and the feedback streams ask

    def connect_idle_decode_demand_receiver(self, receiver) -> None:
        """Connect the optional synthetic idle-load model to decode service."""
        self._idle_decode_demand_receiver = receiver

    def accept_idle_decode_demand(
        self, *, rounds, code, spatial_nodes, label
    ) -> None:
        """Submit modeled idle-memory work; control never sees a decoder."""
        self._idle_decode_demand_receiver(
            rounds,
            on_done=_ignore_completion,
            code=code,
            spatial_nodes=spatial_nodes,
            label=label,
        )

    def bind_stream_operation(
        self, operation_id: int, stream_id, stream_offset: int
    ) -> None:
        """Note which stream and offset a segment's rounds fold into."""
        self._stream_binding_by_operation_id[operation_id] = (
            stream_id,
            stream_offset,
        )

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""
        self._required_stream_end_by_operation_id[operation_id] = (
            required_stream_end
        )


def _is_awaiting_strong(window: message.Window) -> bool:
    """Committed provisionally: the strong redo has not published yet."""
    if not window.committed:
        return False
    if window.is_absorbed:
        return False
    return window.published_request_key is None


def _is_released(store, arrived_for, round_key: tuple) -> bool:
    """True when a round that already arrived is no longer in the store."""
    operation_id, round_index = round_key
    if not store.has_operation(operation_id):
        return True
    arrived = arrived_for(operation_id)
    if round_index > arrived:
        return False
    fragments = store.retained_fragments(round_key)
    return fragments is None


def _move_hold_to_input(
    store, previous_owner, owner, job: message.DecodeJob
) -> None:
    """The window's hold becomes the request's, or a fresh one is made."""
    if store.has_hold(previous_owner):
        store.transfer_hold(previous_owner, owner)
        return
    identities = _round_identities_of(job.payloads)
    store.register_hold(owner, identities)


def _round_identities_of(payloads) -> tuple:
    """The distinct (operation, round) keys of the payloads, in order."""
    identities = {}
    for fragment in payloads:
        identities[(fragment.operation_id, fragment.round_index)] = None
    return tuple(identities)


def _fragment_patch_order(fragment):
    return message.stable_identity_order_key(fragment.patch_id)


def _read_range(window: message.Window) -> tuple:
    """The inclusive round range a window reads."""
    first_round = window.commit_lo
    if window.buffer_lo is not None:
        first_round = window.buffer_lo
    last_round = window.commit_hi
    if window.buffer_hi is not None:
        last_round = window.buffer_hi
    return first_round, last_round


def _run_at_delivery(on_delivered: Callable[[], None], _transfer) -> None:
    on_delivered()


def _representative_patch(operation: message.Operation):
    """The patch a backlog row names: the first patch, qubit, or the id."""
    if operation.patches:
        return operation.patches[0]
    if operation.qubits:
        return operation.qubits[0]
    return operation.id


def _ignore_completion() -> None:
    """An idle decode's completion has no listener."""
