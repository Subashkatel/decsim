"""The strong decoder tier on the window side.

StrongEscalation plans the strong re-decode of an escalated window: the
two-sided context window (Toshio et al. 2510.25222 Sec. III A, the
serial modification) or the forward strong window of the double window
scheme (Sec. III C, Fig. 12) that absorbs the windows it covers and
restarts the weak chain past it; it holds the forward window's job until
the restart window's weak commit or, at the operation's end, until every
strong window round is stored, and it carries the strong result's
selection to the decoder side. A run without the tier uses NoStrongTier.
The decoder side, which destination waits for which strong result, is
StrongRequests (strong_requests.py).

Wide state recorded: StrongEscalation sets twelve attributes, the nine
components it works on and its three tables (the pending escalations,
the delivered selections, the landings waiting for one). Slice 7's
structural commits dissolve it into StrongRedecode (design note
docs/rewrite/notes/slice_07_escalation.md, section 3), so the width is
recorded, not split.
"""

import copy
import dataclasses
import enum
import functools
import types
from typing import Callable, Optional

import decsim.message as message
import decsim.windows.round_retention as round_retention

LOG_SOURCE = "DecoderCluster"


class NoStrongTier:
    """The window side of a run that never escalates.

    Nothing is pending and every hook is a no-op; Baseline never asks
    it to build a strong job.
    """

    pending_escalations: dict = {}

    def after_arrival(self, operation_id) -> None:
        """A round arrived; nothing waits for it."""
        del operation_id

    def after_weak_commit(self, key) -> None:
        """A weak window committed; nothing waits for it."""
        del key

    def pending_strong_work_snapshot(self) -> tuple:
        """No strong window is ever assigned."""
        return ()


class StrongEscalation:
    """The window side of the strong tier.

    When a weak window escalates, how its strong job is built (the
    two-sided context, or a forward strong window that absorbs the
    windows it covers and restarts the weak chain past it), when the
    deferred job is submitted (the far weak boundary, or the terminal
    data), and how the strong result's selection reaches the decoder
    side. It works on the window components it is built with: the
    planner, the tracker, the retention, the request builder, the
    transfers, the requester, the ledger and the interaction.
    """

    def __init__(
        self,
        engine,
        planner,
        tracker,
        retention,
        builder,
        transfers,
        requester,
        ledger,
        interaction,
    ) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.builder = builder
        self.transfers = transfers
        self.requester = requester
        self.ledger = ledger
        self.interaction = interaction
        self._escalations = _EscalationRegistry()
        # strong request keys whose selection has arrived over
        # weak_decoder_to_strong_decoder, and the strong inputs landed in
        # their unit that wait for one that has not
        self._delivered_selections: set = set()
        self._landing_after_selection: dict = {}

    # ---- the context window: built now, submitted now or after the selection

    def make_strong_job(
        self, weak_job: message.DecodeJob, label: str
    ) -> message.Submission:
        """The strong job for a weak one, with its context held and its send.

        A parallel policy enqueues it at once; a serial one hands its job
        back through prepare_strong_selection once the selection is sent.
        """
        strong_job = self.make_strong_decode_job(weak_job, label)
        return self._strong_submission(strong_job, None)

    def make_strong_decode_job(
        self, weak_job: message.DecodeJob, label: str
    ) -> message.DecodeJob:
        """Build the two-sided strong re-decode job for an escalated window.

        The job is priced for the context rounds that exist: a window at
        the operation's edge has a shorter context than commit + 2 buffer.
        """
        key = (weak_job.op_id, weak_job.window_id)
        weak_window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[weak_job.op_id]
        strong_window = self._strong_context_window(weak_window)
        round_count = self.tracker.round_count_for_window(
            operation.id, strong_window
        )
        left_exclusions = _left_fault_exclusions(strong_window.commit_lo)
        model = self._build_strong_window_model(
            operation, strong_window, round_count, left_exclusions
        )
        request_key = self.builder.new_request_key(
            weak_job.op_id, weak_job.window_id, message.DecoderTier.STRONG
        )
        self._require_context_stored(key, weak_job.op_id, strong_window)
        strong_store = self.retention.strong_store
        self.builder.stamp_first_round(strong_window, strong_store)
        payloads = self.builder.assemble_payloads(strong_window, strong_store)
        payload_round_count = message.distinct_round_count(payloads)
        return message.DecodeJob(
            op_id=weak_job.op_id,
            window_id=weak_job.window_id,
            n_rounds=payload_round_count,
            ready_time=self.engine.now,
            label=label,
            hint="strong",
            spatial_nodes=weak_job.spatial_nodes,
            code=weak_job.code,
            dem=model,
            payloads=payloads,
            attempt=1,
            window=strong_window,
            strong_decode_for=key,
            request_key=request_key,
            request_created_ticks=self.engine.now,
            gate=self.builder,
        )

    # ---- the forward window: planned now, submitted when its far side exists

    def defer_strong_escalation(
        self, weak_job: message.DecodeJob
    ) -> message.DecoderRequestKey:
        """Lay out the forward strong window; hold its job until it may start.

        The strong window absorbs the windows it covers. Its job waits for
        the restart window's weak commit (waiting_far_boundary) or, at the
        operation's end, until every clamped strong window round is
        stored (waiting_terminal_data). One strong job per escalation; a
        duplicate raises.
        """
        key = (weak_job.op_id, weak_job.window_id)
        self._refuse_second_escalation(key)
        if weak_job.strong_label is None:
            raise RuntimeError(
                f"double-window escalation {key} needs a declared strong label"
            )
        strong_request_key = self.builder.new_request_key(
            weak_job.op_id, weak_job.window_id, message.DecoderTier.STRONG
        )
        strong_request_created_ticks = self.engine.now
        operation_id, escalated_index = key
        weak_window = self.planner.windows_by_key[key]
        round_count = self.tracker.round_count_for_window(
            operation_id, weak_window
        )
        later_windows = self._later_windows(operation_id, escalated_index)
        plan = self._plan_strong_region(weak_window, later_windows, round_count)
        resolved_region = self._resolve_strong_region_plan(
            key, weak_window, later_windows, round_count, plan
        )
        restart_key = resolved_region.restart_window_key
        self._refuse_readiness_collision(operation_id, restart_key)
        operation = self.tracker.operation_by_id[operation_id]
        proposed_restart = None
        restart_model = None
        if restart_key is not None:
            proposed_restart = self._proposed_restart_window(restart_key, plan)
            restart_model = self._build_strong_window_model(
                operation,
                proposed_restart,
                round_count,
                resolved_region.restart_fault_exclusion_ranges,
            )
        strong_window = _strong_window_of(key, plan)
        strong_model = self._build_strong_window_model(
            operation,
            strong_window,
            round_count,
            resolved_region.strong_fault_exclusion_ranges,
        )
        logical_candidate = self._strong_window_ownership_candidate(
            key, resolved_region
        )
        guard = self._guard_restart_reads(
            key,
            restart_key,
            proposed_restart,
            strong_request_key,
            resolved_region,
        )
        phase = _EscalationPhase.WAITING_FAR_BOUNDARY
        if restart_key is None:
            phase = _EscalationPhase.WAITING_TERMINAL_DATA
        pending = _PendingEscalation(
            key=key,
            weak_job=weak_job,
            label=weak_job.strong_label,
            resolved_region=resolved_region,
            strong_window=strong_window,
            strong_model=strong_model,
            selection_arrival_ticks=None,
            phase=phase,
            strong_request_key=strong_request_key,
            strong_request_created_ticks=strong_request_created_ticks,
        )
        try:
            self.ledger.contributions = logical_candidate
            self._register_pending(pending, operation_id, restart_key)
            self._hold_strong_context(key, strong_request_key, plan)
            pending_hold = message.PendingStrong(strong_request_key)
            for absorbed_key in resolved_region.absorbed_window_keys:
                self._absorb_window(absorbed_key, restart_key, pending_hold)
            self._log_assignment(pending, resolved_region)
            if restart_key is not None:
                self._reslice_restart_window(
                    restart_key,
                    plan.restart_buffer_lo,
                    restart_model,
                    plan.commit_hi,
                    plan.restart_seam_fault_owner,
                )
                restart = self.planner.windows_by_key[restart_key]
                # its absorbed dependency is gone
                self.requester.request_if_ready(restart, self)
            return strong_request_key
        finally:
            if guard is not None:
                self.retention.release_hold_if_live(guard)
                self.retention.release_hold_if_live(
                    guard, self.retention.strong_store
                )

    # ---- the selection: the decoder side asks for the strong result

    def prepare_strong_selection(
        self,
        weak_job: message.DecodeJob,
        strong_request_key: message.DecoderRequestKey,
        serial_strong_job: Optional[message.DecodeJob],
        *,
        deferred: bool,
        on_selection_delivered: Callable[[], None],
    ) -> None:
        """Send the selection; on_selection_delivered runs at its arrival."""
        key = (weak_job.op_id, weak_job.window_id)
        pending = self._escalations.peek_key(key)
        if deferred:
            self._send_deferred_selection(
                weak_job, strong_request_key, pending, on_selection_delivered
            )
            return
        if serial_strong_job is not None:
            selection_arrival_ticks = self._send_selection(
                weak_job, strong_request_key, on_selection_delivered
            )
            self._enqueue_reserved_strong(
                serial_strong_job, selection_arrival_ticks
            )
            return
        if pending is not None:
            raise RuntimeError("deferred pending request needs an explicit key")
        self._send_selection(
            weak_job, strong_request_key, on_selection_delivered
        )

    # ---- the hooks that submit a deferred job

    def after_weak_commit(self, key) -> None:
        """A weak commit at a strong window's far boundary releases its job."""
        pending = self._escalations.peek_far(key)
        if pending is not None:
            self._submit_far_strong(key, pending)

    def after_arrival(self, operation_id) -> None:
        """A round arrived: a terminal strong window waits for its tail."""
        pending = self._escalations.peek_terminal(operation_id)
        if pending is None:
            return
        if self._is_terminal_data_stored(operation_id, pending):
            self._submit_terminal_strong(operation_id, pending)

    # ---- observation

    @property
    def pending_escalations(self) -> dict:
        """The deferral phase of every escalated window, for the settlement."""
        phases = {}
        snapshot = self._escalations.snapshot_phases()
        for key, phase in snapshot.items():
            phases[key] = phase.name.lower()
        return phases

    def pending_strong_work_snapshot(self) -> tuple:
        """The strong windows assigned but not yet admitted for service."""
        return self._escalations.snapshot_work()

    # ---- private: the context window

    def _strong_context_window(
        self, weak_window: message.Window
    ) -> message.Window:
        context_lo, commit_lo, commit_hi, context_hi = (
            round_retention.strong_context_bounds(weak_window)
        )
        round_count = context_hi - context_lo + 1
        strong_window = message.Window(
            op_id=weak_window.op_id,
            k=weak_window.k,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=context_hi,
            buffer_lo=context_lo,
            n_rounds=round_count,
        )
        strong_window.boundary_in = weak_window.boundary_in
        return strong_window

    def _require_context_stored(
        self, key: tuple, operation_id, strong_window: message.Window
    ) -> None:
        """Every context round that already arrived must sit in buffer 1."""
        context_reads = self.retention.read_keys_for_bounds(
            operation_id,
            strong_window.buffer_lo,
            strong_window.buffer_hi,
            strong_window,
        )
        strong_store = self.retention.strong_store
        missing = []
        for round_key in context_reads:
            arrived = self.tracker.rounds_arrived(round_key[0])
            if round_key[1] > arrived:
                continue
            fragments = strong_store.retained_fragments(round_key)
            if fragments is None:
                missing.append(round_key)
        if missing:
            raise RuntimeError(
                f"strong context for {key} arrived at Buffer 0 but is not "
                f"stored in syndrome buffer 1: {missing} "
                f"(controller_to_strong_buffer lag beyond the escalation "
                f"margin, or an early release)"
            )

    # ---- private: submitting a strong job with its input send

    def _enqueue_strong_job(
        self, strong_job: message.DecodeJob, selection_arrival_ticks: int
    ) -> None:
        """Queue a deferred strong job now; its input is sent at dispatch."""
        submission = self._strong_submission(
            strong_job, selection_arrival_ticks
        )
        self.requester.enqueue(submission)

    def _enqueue_reserved_strong(
        self, strong_job: message.DecodeJob, selection_arrival_ticks: int
    ) -> None:
        """Queue a serial strong job whose context was held when it was built.

        Its send also waits for the selection.
        """
        send_input = self._input_send(strong_job, selection_arrival_ticks)
        submission = message.Submission(strong_job, send_input)
        self.requester.enqueue(submission)

    def _strong_submission(
        self,
        strong_job: message.DecodeJob,
        selection_arrival_ticks: Optional[int],
    ) -> message.Submission:
        """The strong job with its input send; its context rounds held.

        The unit is assigned first, then the input moves into that unit's
        memory over strong_buffer_to_strong_decoder; a serial job also
        waits for its selection to arrive. selection_arrival_ticks is the
        tick the selection is expected, for the pool's estimate; the
        landing itself waits for the delivery.
        """
        request_key = strong_job.request_key
        window_key = strong_job.strong_decode_for
        strong_store = self.retention.strong_store
        in_flight = message.StrongInputInFlight(request_key)
        potential = message.PotentialStrong(window_key)
        pending = message.PendingStrong(request_key)
        if strong_store.has_hold(potential):
            self.retention.transfer_hold(potential, in_flight, strong_store)
        elif strong_store.has_hold(pending):
            self.retention.transfer_hold(pending, in_flight, strong_store)
        else:
            round_identities = round_retention.round_identities_of(
                strong_job.payloads
            )
            strong_store.register_hold(in_flight, round_identities)
        self.retention.bind_input_hold(strong_job, in_flight, strong_store)
        send_input = self._input_send(strong_job, selection_arrival_ticks)
        return message.Submission(strong_job, send_input)

    def _input_send(
        self,
        strong_job: message.DecodeJob,
        selection_arrival_ticks: Optional[int],
    ) -> Callable[[Callable[[], None]], int]:
        """The job's input send, its payload bits and context fixed now.

        The staging clears the job's payloads when the input lands.
        """
        payload_bits = strong_job.payload_bits()
        context_identities = round_retention.round_identities_of(
            strong_job.payloads
        )
        return functools.partial(
            self._send_input,
            strong_job,
            payload_bits,
            context_identities,
            selection_arrival_ticks,
        )

    def _send_input(
        self,
        strong_job: message.DecodeJob,
        payload_bits: Optional[int],
        context_identities: tuple,
        selection_arrival_ticks: Optional[int],
        on_landed: Callable[[], None],
    ) -> int:
        """Send the input at dispatch; returns the delay the pool expects."""
        # the DMA reads only context rounds that landed in syndrome buffer
        # 1 (always, whenever the controller_to_strong_buffer margin holds)
        self.retention.strong_store.require_stored(context_identities)
        landed = on_landed
        if selection_arrival_ticks is not None:
            landed = functools.partial(
                self._land_after_selection, strong_job.request_key, on_landed
            )
        expected_delay_ticks = self.transfers.send_for_job(
            message.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
            strong_job,
            payload_bits=payload_bits,
            on_delivered=landed,
        )
        if selection_arrival_ticks is None:
            return expected_delay_ticks
        selection_delay_ticks = selection_arrival_ticks - self.engine.now
        return max(expected_delay_ticks, selection_delay_ticks)

    def _land_after_selection(
        self,
        request_key: message.DecoderRequestKey,
        on_landed: Callable[[], None],
    ) -> None:
        """A serial strong input lands only once its selection arrived."""
        if request_key in self._delivered_selections:
            on_landed()
            return
        self._landing_after_selection[request_key] = on_landed

    # ---- private: the selection

    def _send_selection(
        self,
        weak_job: message.DecodeJob,
        strong_request_key: message.DecoderRequestKey,
        on_selection_delivered: Callable[[], None],
    ) -> int:
        """Send the escalation; returns the tick its arrival is expected."""
        delivered = functools.partial(
            self._selection_delivered,
            strong_request_key,
            on_selection_delivered,
        )
        expected_delay_ticks = self.transfers.send_for_job(
            message.LinkPath.WEAK_DECODER_TO_STRONG_DECODER,
            weak_job,
            payload_bits=None,
            request_key=strong_request_key,
            on_delivered=delivered,
        )
        return self.engine.now + expected_delay_ticks

    def _selection_delivered(
        self,
        request_key: message.DecoderRequestKey,
        on_selection_delivered: Callable[[], None],
    ) -> None:
        """The selection arrived: a landed input waiting for it may start."""
        self._delivered_selections.add(request_key)
        waiting = self._landing_after_selection.pop(request_key, None)
        if waiting is not None:
            waiting()
        on_selection_delivered()

    def _send_deferred_selection(
        self,
        weak_job: message.DecodeJob,
        strong_request_key: message.DecoderRequestKey,
        pending: "_PendingEscalation",
        on_selection_delivered: Callable[[], None],
    ) -> None:
        """A forward window's selection; a terminal one may submit now."""
        selection_arrival_ticks = self._send_selection(
            weak_job, strong_request_key, on_selection_delivered
        )
        pending = self._escalations.update_selection_arrival(
            pending, selection_arrival_ticks
        )
        if pending.phase is not _EscalationPhase.WAITING_TERMINAL_DATA:
            return
        operation_id = pending.key[0]
        if self._is_terminal_data_stored(operation_id, pending):
            self._submit_terminal_strong(operation_id, pending)

    # ---- private: the forward window's plan

    def _refuse_second_escalation(self, key: tuple) -> None:
        if self._escalations.peek_key(key) is not None:
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one "
                f"switching event creates exactly one strong job"
            )
        contribution = self.ledger.contributions.get(key)
        if contribution is None:
            return
        if contribution.ownership_kind == "strong_window":
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one "
                f"switching event creates exactly one strong job"
            )

    def _later_windows(self, operation_id, escalated_index: int) -> list:
        """The operation's windows past the escalated one, in index order."""
        later_windows = []
        for window_index in self.planner.window_indices_of(operation_id):
            if window_index > escalated_index:
                window = self.planner.windows_by_key[
                    (operation_id, window_index)
                ]
                later_windows.append(window)
        return later_windows

    def _plan_strong_region(
        self, weak_window: message.Window, later_windows: list, round_count
    ) -> message.StrongRegionPlan:
        weak_info = message.WindowInfo.from_window(weak_window)
        later_infos = []
        for window in later_windows:
            window_info = message.WindowInfo.from_window(window)
            later_infos.append(window_info)
        return self.interaction.plan_strong_region(
            weak_info, later_infos, round_count
        )

    def _resolve_strong_region_plan(
        self,
        key: tuple,
        weak_window: message.Window,
        later_windows: list,
        round_count: int,
        plan: message.StrongRegionPlan,
    ) -> "_ResolvedStrongRegion":
        """Check the plan against the live window graph; resolve its reads."""
        _check_region_bounds(key, weak_window, round_count, plan)
        absorbed = _absorbed_window_keys(later_windows, plan)
        _refuse_crossing_window(later_windows, plan)
        self._withdraw_absorbed_windows(absorbed)
        restart_key = _restart_window_key(later_windows, plan)
        restart_reads = self._restart_reads(key, restart_key, plan)
        strong_exclusions, restart_exclusions = _fault_exclusions(
            plan, round_count, restart_key
        )
        context_reads = _context_round_keys(weak_window.op_id, plan)
        purpose = f"strong-region plan for {key}"
        # the strong window's context lives in syndrome buffer 1; the
        # restart window's weak reads stay retained in Buffer 0 by its
        # own window hold
        self.retention.require_retained(
            context_reads, purpose, self.retention.strong_store
        )
        self.retention.require_retained(restart_reads, purpose)
        return _ResolvedStrongRegion(
            plan=plan,
            absorbed_window_keys=absorbed,
            restart_window_key=restart_key,
            restart_read_keys=tuple(restart_reads),
            strong_fault_exclusion_ranges=strong_exclusions,
            restart_fault_exclusion_ranges=restart_exclusions,
        )

    def _withdraw_absorbed_windows(self, absorbed: tuple) -> None:
        for absorbed_key in absorbed:
            window = self.planner.windows_by_key[absorbed_key]
            _refuse_absorbing_decoded(absorbed_key, window)
            if window.queued:
                # early-shipped at data-complete but parked on the
                # escalated window's boundary, which will never arrive
                # (the strong window owns it); withdraw the unstarted attempt
                self.requester.withdraw(window)

    def _restart_reads(
        self, key: tuple, restart_key: Optional[tuple], plan
    ) -> list:
        """The restart window's reads from its re-sliced buffer start."""
        if restart_key is None:
            _refuse_terminal_restart_data(key, plan)
            return []
        restart = self.planner.windows_by_key[restart_key]
        _check_restart_tiling(key, restart_key, restart, plan)
        return self.retention.read_keys_for_bounds(
            restart.op_id, plan.restart_buffer_lo, restart.buffer_hi, restart
        )

    def _refuse_readiness_collision(
        self, operation_id, restart_key: Optional[tuple]
    ) -> None:
        if restart_key is None:
            collision = self._escalations.peek_terminal(operation_id)
            readiness_key = operation_id
        else:
            collision = self._escalations.peek_far(restart_key)
            readiness_key = restart_key
        if collision is not None:
            raise RuntimeError(f"readiness index collision for {readiness_key}")

    def _proposed_restart_window(
        self, restart_key: tuple, plan: message.StrongRegionPlan
    ) -> message.Window:
        """The restart window as it will read after the re-slice."""
        restart = self.planner.windows_by_key[restart_key]
        proposed = copy.deepcopy(restart)
        proposed.buffer_lo = plan.restart_buffer_lo
        return proposed

    def _build_strong_window_model(
        self,
        operation: message.Operation,
        window: message.Window,
        round_count: int,
        fault_exclusions: tuple,
    ):
        resolved = self.planner.resolved_operation_by_id[operation.id]
        return self.planner.models.strong_model_for_operation(
            operation, resolved, window, round_count, fault_exclusions
        )

    def _strong_window_ownership_candidate(
        self, key: tuple, resolved_region: "_ResolvedStrongRegion"
    ) -> dict:
        """The logical-owner map with the strong window; nothing live moves."""
        plan = resolved_region.plan
        replaced_owner_keys = {key, *resolved_region.absorbed_window_keys}
        candidate = {}
        for owner_key, contribution in self.ledger.contributions.items():
            if owner_key not in replaced_owner_keys:
                candidate[owner_key] = contribution
        for other_key, contribution in candidate.items():
            _refuse_overlapping_contribution(key, plan, other_key, contribution)
        candidate[key] = message.LogicalContribution(
            owner_key=key,
            commit_lo=plan.commit_lo,
            commit_hi=plan.commit_hi,
            ownership_kind="strong_window",
            logical_observables=None,
        )
        return candidate

    def _guard_restart_reads(
        self,
        key: tuple,
        restart_key: Optional[tuple],
        proposed_restart: Optional[message.Window],
        strong_request_key: message.DecoderRequestKey,
        resolved_region: "_ResolvedStrongRegion",
    ) -> Optional[message.RephaseGuard]:
        """Hold the restart window's reads in both stores while the plan lands.

        The restart window's weak reads must still sit under its own
        window hold; once its weak decode is built the hold moves to the
        job, and a rephase can no longer claim those rounds.
        """
        if restart_key is None:
            return None
        guard = message.RephaseGuard(strong_request_key)
        weak_store = self.retention.weak_store
        if not weak_store.has_hold(restart_key):
            raise RuntimeError(
                f"strong-region plan for {key} requires restart window "
                f"{restart_key}'s weak reads under its window hold, "
                f"which is no longer live (its weak decode already "
                f"consumed them)"
            )
        restart_identities = weak_store.hold_round_identities(restart_key)
        guarded_weak = list(restart_identities)
        guarded_weak += list(resolved_region.restart_read_keys)
        guarded_strong = self._guarded_strong_reads(
            key, restart_key, proposed_restart, resolved_region
        )
        weak_store.register_hold(guard, guarded_weak)
        self.retention.strong_store.register_hold(guard, guarded_strong)
        return guard

    def _guarded_strong_reads(
        self,
        key: tuple,
        restart_key: tuple,
        proposed_restart: message.Window,
        resolved_region: "_ResolvedStrongRegion",
    ) -> list:
        strong_store = self.retention.strong_store
        escalated_potential = message.PotentialStrong(key)
        restart_potential = message.PotentialStrong(restart_key)
        escalated_identities = strong_store.hold_round_identities(
            escalated_potential
        )
        restart_identities = strong_store.hold_round_identities(
            restart_potential
        )
        guarded = list(escalated_identities)
        guarded += list(restart_identities)
        guarded += _context_round_keys(key[0], resolved_region.plan)
        restart_reads = list(resolved_region.restart_read_keys)
        guarded += self.retention.strong_context_read_keys(
            proposed_restart, restart_reads
        )
        return guarded

    # ---- private: landing the forward window's plan

    def _register_pending(
        self,
        pending: "_PendingEscalation",
        operation_id,
        restart_key: Optional[tuple],
    ) -> None:
        if restart_key is None:
            self._escalations.register_terminal(pending, operation_id)
        else:
            self._escalations.register_far(pending, restart_key)

    def _hold_strong_context(
        self,
        key: tuple,
        strong_request_key: message.DecoderRequestKey,
        plan: message.StrongRegionPlan,
    ) -> None:
        """The window's potential strong read becomes the request's hold."""
        self.retention.transfer_potential_to_pending(key, strong_request_key)
        pending_hold = message.PendingStrong(strong_request_key)
        context_keys = _context_round_keys(key[0], plan)
        self.retention.strong_store.replace_hold(pending_hold, context_keys)

    def _absorb_window(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """A window the strong window covers is never weak-decoded.

        It counts committed with no logical contribution, the restart
        window no longer waits for it, and its rounds are the strong
        request's.
        """
        window = self.planner.windows_by_key[key]
        _refuse_absorbing_active(key, window)
        window.queued = True  # keeps the requester away
        window.committed = True
        window.is_absorbed = True
        if restart_key is not None:
            self._unhook_restart(key, restart_key, window)
        self.retention.release_hold_if_live(key)
        self._release_absorbed_strong_hold(key, restart_key, replacement)
        self.engine.log(
            LOG_SOURCE,
            f"window {key} absorbed into the strong window "
            f"(weak chain skips it)",
        )

    def _unhook_restart(
        self, key: tuple, restart_key: tuple, window: message.Window
    ) -> None:
        restart = self.planner.windows_by_key[restart_key]
        if key in restart.deps:
            restart.deps.remove(key)
            restart.deps_remaining -= 1
        if restart_key in window.dependents:
            window.dependents.remove(restart_key)

    def _release_absorbed_strong_hold(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """Drop the absorbed window's potential read; the request holds it."""
        strong_store = self.retention.strong_store
        absorbed = message.PotentialStrong(key)
        needed_identities = strong_store.hold_round_identities(absorbed)
        needed = set(needed_identities)
        replacement_identities = strong_store.hold_round_identities(replacement)
        replacements = set(replacement_identities)
        if restart_key is not None:
            restart_potential = message.PotentialStrong(restart_key)
            restart_identities = strong_store.hold_round_identities(
                restart_potential
            )
            replacements.update(restart_identities)
        if not needed <= replacements:
            raise RuntimeError("absorption replacement does not cover packets")
        strong_store.release_hold(absorbed)

    def _log_assignment(
        self,
        pending: "_PendingEscalation",
        resolved_region: "_ResolvedStrongRegion",
    ) -> None:
        plan = resolved_region.plan
        readiness_description = "the far-side weak boundary"
        if resolved_region.restart_window_key is None:
            readiness_description = "terminal data"
        absorbed_count = len(resolved_region.absorbed_window_keys)
        self.engine.log(
            LOG_SOURCE,
            f"{pending.label}: strong window rounds {plan.commit_lo}-"
            f"{plan.commit_hi} assigned; weak chain skips "
            f"{absorbed_count} window(s); "
            f"strong start deferred until {readiness_description}",
        )

    def _reslice_restart_window(
        self,
        restart_key: tuple,
        buffer_lo: int,
        model,
        strong_window_hi: int,
        seam_owner: message.SeamFaultOwner,
    ) -> None:
        """Install the restart window's re-sliced reads and their model."""
        restart = self.planner.windows_by_key[restart_key]
        restart.buffer_lo = buffer_lo
        restart.n_rounds = restart.buffer_hi - restart.buffer_lo + 1
        self.retention.replace_window_reads(restart_key, restart)
        if model is not None:
            self.planner.model_by_window[restart_key] = model
        seam_owner_name = seam_owner.name.lower()
        self.engine.log(
            LOG_SOURCE,
            f"restart window {restart_key} re-sliced across strong window "
            f"edge {strong_window_hi} (reads rounds {restart.buffer_lo}-"
            f"{restart.buffer_hi}; crossing faults owned by "
            f"{seam_owner_name})",
        )

    # ---- private: submitting the forward window's job

    def _is_terminal_data_stored(
        self, operation_id, pending: "_PendingEscalation"
    ) -> bool:
        stored_through = self.tracker.strong_rounds_arrived(operation_id)
        return stored_through >= pending.resolved_region.plan.context_hi

    def _build_pending_strong_job(
        self, pending: "_PendingEscalation"
    ) -> message.DecodeJob:
        """The strong window's job, once both of its boundaries exist.

        The strong window commits all r_strong rounds and reads one
        buffer of raw context per face, owning nothing that touches
        rounds before its extent.
        """
        key = pending.key
        weak_job = pending.weak_job
        strong_window = pending.strong_window
        strong_store = self.retention.strong_store
        payloads = self.builder.assemble_payloads(strong_window, strong_store)
        _check_every_round_retained(pending, payloads)
        self.builder.stamp_first_round(strong_window, strong_store)
        payload_round_count = message.distinct_round_count(payloads)
        return message.DecodeJob(
            op_id=key[0],
            window_id=key[1],
            n_rounds=payload_round_count,
            ready_time=self.engine.now,
            label=pending.label,
            hint="strong",
            spatial_nodes=weak_job.spatial_nodes,
            code=weak_job.code,
            dem=pending.strong_model,
            payloads=payloads,
            attempt=1,
            window=strong_window,
            strong_decode_for=key,
            request_key=pending.strong_request_key,
            request_created_ticks=pending.strong_request_created_ticks,
            gate=self.builder,
        )

    def _submit_far_strong(
        self, far_boundary_key: tuple, pending: "_PendingEscalation"
    ) -> None:
        if pending.selection_arrival_ticks is None:
            raise RuntimeError(
                "far strong submission requires the "
                "weak_decoder_to_strong_decoder send"
            )
        strong_job = self._build_pending_strong_job(pending)
        self._escalations.take_far(far_boundary_key, pending)
        self._enqueue_strong_job(strong_job, pending.selection_arrival_ticks)
        self.engine.log(
            LOG_SOURCE,
            f"{pending.label}: far-side weak boundary determined -> "
            "strong window submitted",
        )

    def _submit_terminal_strong(
        self, operation_id, pending: "_PendingEscalation"
    ) -> None:
        if pending.selection_arrival_ticks is None:
            raise RuntimeError(
                "terminal strong submission requires the "
                "weak_decoder_to_strong_decoder send"
            )
        strong_job = self._build_pending_strong_job(pending)
        self._escalations.take_terminal(operation_id, pending)
        self._enqueue_strong_job(strong_job, pending.selection_arrival_ticks)
        self.engine.log(
            LOG_SOURCE,
            f"{pending.label}: terminal data complete -> "
            "strong window submitted",
        )


@dataclasses.dataclass(frozen=True)
class _ResolvedStrongRegion:
    """One planned strong region resolved against the live window graph."""

    plan: message.StrongRegionPlan
    absorbed_window_keys: tuple
    restart_window_key: Optional[tuple]
    restart_read_keys: tuple
    strong_fault_exclusion_ranges: tuple
    restart_fault_exclusion_ranges: Optional[tuple]


class _EscalationPhase(enum.Enum):
    """The one readiness condition that releases a pending strong job."""

    WAITING_FAR_BOUNDARY = enum.auto()
    WAITING_TERMINAL_DATA = enum.auto()


@dataclasses.dataclass(frozen=True)
class _PendingEscalation:
    """Everything kept from the plan until the one strong-job submission."""

    key: tuple
    weak_job: message.DecodeJob
    label: str
    resolved_region: _ResolvedStrongRegion
    strong_window: message.Window
    strong_model: object
    selection_arrival_ticks: Optional[int]
    phase: _EscalationPhase
    strong_request_key: message.DecoderRequestKey
    strong_request_created_ticks: int


class _EscalationRegistry:
    """The pending escalations by window, each in one readiness index."""

    def __init__(self) -> None:
        self.pending_by_key: dict = {}
        self.key_by_far_boundary: dict = {}
        self.key_by_terminal_operation: dict = {}

    def register_far(
        self, pending: _PendingEscalation, far_boundary_key: tuple
    ) -> None:
        self._register(
            pending,
            _EscalationPhase.WAITING_FAR_BOUNDARY,
            self.key_by_far_boundary,
            far_boundary_key,
        )

    def register_terminal(
        self, pending: _PendingEscalation, operation_id
    ) -> None:
        self._register(
            pending,
            _EscalationPhase.WAITING_TERMINAL_DATA,
            self.key_by_terminal_operation,
            operation_id,
        )

    def update_selection_arrival(
        self, expected: _PendingEscalation, selection_arrival_ticks: int
    ) -> _PendingEscalation:
        """Record the selection's expected arrival without moving ownership."""
        if self.pending_by_key.get(expected.key) is not expected:
            raise RuntimeError(
                f"stale escalation timing update for {expected.key}"
            )
        if expected.selection_arrival_ticks is not None:
            raise RuntimeError(
                f"duplicate weak_decoder_to_strong_decoder send for "
                f"{expected.key}"
            )
        updated = dataclasses.replace(
            expected, selection_arrival_ticks=selection_arrival_ticks
        )
        self.pending_by_key[expected.key] = updated
        return updated

    def peek_key(self, key: tuple) -> Optional[_PendingEscalation]:
        return self.pending_by_key.get(key)

    def peek_far(self, far_boundary_key: tuple) -> Optional[_PendingEscalation]:
        key = self.key_by_far_boundary.get(far_boundary_key)
        if key is None:
            return None
        return self.pending_by_key[key]

    def peek_terminal(self, operation_id) -> Optional[_PendingEscalation]:
        key = self.key_by_terminal_operation.get(operation_id)
        if key is None:
            return None
        return self.pending_by_key[key]

    def take_far(
        self, far_boundary_key: tuple, expected: _PendingEscalation
    ) -> _PendingEscalation:
        return self._take(
            expected,
            _EscalationPhase.WAITING_FAR_BOUNDARY,
            self.key_by_far_boundary,
            far_boundary_key,
        )

    def take_terminal(
        self, operation_id, expected: _PendingEscalation
    ) -> _PendingEscalation:
        return self._take(
            expected,
            _EscalationPhase.WAITING_TERMINAL_DATA,
            self.key_by_terminal_operation,
            operation_id,
        )

    def snapshot_phases(self) -> types.MappingProxyType:
        phases = {}
        for key, pending in self.pending_by_key.items():
            phases[key] = pending.phase
        return types.MappingProxyType(phases)

    def snapshot_work(self) -> tuple:
        """The pending strong assignments without the live windows."""
        phase_names = {
            _EscalationPhase.WAITING_FAR_BOUNDARY: "waiting_far_boundary",
            _EscalationPhase.WAITING_TERMINAL_DATA: "waiting_terminal_data",
        }
        records = []
        for key, pending in self.pending_by_key.items():
            phase_name = phase_names[pending.phase]
            record = (key, phase_name, pending.strong_window.n_rounds)
            records.append(record)
        ordered = sorted(records, key=_work_record_order)
        return tuple(ordered)

    def _register(
        self,
        pending: _PendingEscalation,
        expected_phase: _EscalationPhase,
        readiness_index: dict,
        readiness_key,
    ) -> None:
        if pending.phase is not expected_phase:
            raise RuntimeError(
                f"pending escalation {pending.key} has phase "
                f"{pending.phase.name}, expected {expected_phase.name}"
            )
        if pending.key in self.pending_by_key:
            raise RuntimeError(
                f"duplicate strong escalation for window {pending.key}: one "
                "switching event creates exactly one strong job"
            )
        if readiness_key in readiness_index:
            raise RuntimeError(f"readiness index collision for {readiness_key}")
        self.pending_by_key[pending.key] = pending
        readiness_index[readiness_key] = pending.key

    def _take(
        self,
        expected: _PendingEscalation,
        expected_phase: _EscalationPhase,
        readiness_index: dict,
        readiness_key,
    ) -> _PendingEscalation:
        if expected.phase is not expected_phase:
            raise RuntimeError(
                f"wrong-phase take for escalation {expected.key}"
            )
        primary = self.pending_by_key.get(expected.key)
        indexed_key = readiness_index.get(readiness_key)
        if primary is not expected or indexed_key != expected.key:
            raise RuntimeError(
                f"stale escalation take for readiness key {readiness_key}"
            )
        del readiness_index[readiness_key]
        del self.pending_by_key[expected.key]
        return expected


def _left_fault_exclusions(commit_lo: int) -> tuple:
    """The rounds before the strong window, whose faults it never owns."""
    if commit_lo > 1:
        return ((1, commit_lo - 1),)
    return ()


def _context_round_keys(operation_id, plan: message.StrongRegionPlan) -> list:
    """The (operation, round) keys of the strong window's whole context."""
    stop_round = plan.context_hi + 1
    round_keys = []
    for round_index in range(plan.context_lo, stop_round):
        round_keys.append((operation_id, round_index))
    return round_keys


def _strong_window_of(
    key: tuple, plan: message.StrongRegionPlan
) -> message.Window:
    round_count = plan.context_hi - plan.context_lo + 1
    return message.Window(
        op_id=key[0],
        k=key[1],
        commit_lo=plan.commit_lo,
        commit_hi=plan.commit_hi,
        buffer_hi=plan.context_hi,
        buffer_lo=plan.context_lo,
        n_rounds=round_count,
    )


def _check_region_bounds(
    key: tuple,
    weak_window: message.Window,
    round_count: int,
    plan: message.StrongRegionPlan,
) -> None:
    """Context holds commit holds the weak window, inside the operation."""
    nested_bounds = (
        1,
        plan.context_lo,
        plan.commit_lo,
        weak_window.commit_lo,
        weak_window.commit_hi,
        plan.commit_hi,
        plan.context_hi,
        round_count,
    )
    if not _is_nondecreasing(nested_bounds):
        raise RuntimeError(
            f"invalid strong-region bounds for {key}: context "
            f"{plan.context_lo}-{plan.context_hi}, commit "
            f"{plan.commit_lo}-{plan.commit_hi}, operation 1-{round_count}"
        )
    if plan.commit_lo != weak_window.commit_lo:
        raise RuntimeError(
            f"strong-region commit for {key} must start at the "
            f"escalated window's commit start {weak_window.commit_lo}"
        )


def _is_nondecreasing(values: tuple) -> bool:
    for left, right in zip(values, values[1:]):
        if right < left:
            return False
    return True


def _absorbed_window_keys(
    later_windows: list, plan: message.StrongRegionPlan
) -> tuple:
    """The later windows whose whole commit region the strong window covers."""
    absorbed = []
    for window in later_windows:
        if window.commit_hi <= plan.commit_hi:
            absorbed.append(window.key)
    return tuple(absorbed)


def _refuse_crossing_window(
    later_windows: list, plan: message.StrongRegionPlan
) -> None:
    """A window committing across the strong window's edge has no owner."""
    for window in later_windows:
        is_begun = window.commit_lo <= plan.commit_hi
        is_unfinished = plan.commit_hi < window.commit_hi
        if is_begun and is_unfinished:
            raise RuntimeError(
                f"window {window.key} commits {window.commit_lo}-"
                f"{window.commit_hi} across the strong-region edge "
                f"{plan.commit_hi}"
            )


def _refuse_absorbing_decoded(key: tuple, window: message.Window) -> None:
    if window.committed:
        raise RuntimeError(f"cannot absorb window {key}: already committed")
    if window.t_done is not None:
        raise RuntimeError(f"cannot absorb window {key}: already decoded")


def _refuse_absorbing_active(key: tuple, window: message.Window) -> None:
    if window.queued:
        raise RuntimeError(f"cannot absorb window {key}: already queued")
    if window.committed:
        raise RuntimeError(f"cannot absorb window {key}: already committed")


def _restart_window_key(
    later_windows: list, plan: message.StrongRegionPlan
) -> Optional[tuple]:
    """The first later window past the strong window, or None at the end."""
    for window in later_windows:
        if window.commit_lo > plan.commit_hi:
            return window.key
    return None


def _refuse_terminal_restart_data(
    key: tuple, plan: message.StrongRegionPlan
) -> None:
    has_restart_data = plan.restart_buffer_lo is not None
    has_seam_owner = plan.restart_seam_fault_owner is not None
    if has_restart_data or has_seam_owner:
        raise RuntimeError(
            f"terminal strong-region plan for {key} cannot define "
            f"restart seam data"
        )


def _check_restart_tiling(
    key: tuple,
    restart_key: tuple,
    restart: message.Window,
    plan: message.StrongRegionPlan,
) -> None:
    """The restart window commits right after the strong window."""
    expected_start = plan.commit_hi + 1
    if restart.commit_lo != expected_start:
        raise RuntimeError(
            f"strong-region plan for {key} ends at "
            f"{plan.commit_hi}, but restart {restart_key} "
            f"starts at {restart.commit_lo}; committed regions must "
            "tile without a gap"
        )
    if plan.restart_buffer_lo is None:
        raise RuntimeError(
            f"strong-region restart {restart_key} needs a "
            f"buffer start in 1-{restart.commit_lo}"
        )
    is_inside = 1 <= plan.restart_buffer_lo <= restart.commit_lo
    if not is_inside:
        raise RuntimeError(
            f"strong-region restart {restart_key} needs a "
            f"buffer start in 1-{restart.commit_lo}"
        )


def _fault_exclusions(
    plan: message.StrongRegionPlan,
    round_count: int,
    restart_key: Optional[tuple],
) -> tuple:
    """(strong window's, restart window's) fault exclusion ranges.

    The seam's owner keeps the crossing faults: when the strong region
    owns them the restart window excludes everything up to the strong
    window's edge; otherwise the strong window excludes the rounds past
    its edge and the restart window excludes only the rounds before it.
    """
    left_exclusions = _left_fault_exclusions(plan.commit_lo)
    if restart_key is None:
        return left_exclusions, None
    if plan.restart_seam_fault_owner is message.SeamFaultOwner.STRONG_REGION:
        restart_exclusions = ((1, plan.commit_hi),)
        return left_exclusions, restart_exclusions
    right_exclusion = (plan.commit_hi + 1, round_count)
    strong_exclusions = left_exclusions + (right_exclusion,)
    return strong_exclusions, left_exclusions


def _refuse_overlapping_contribution(
    key: tuple,
    plan: message.StrongRegionPlan,
    other_key: tuple,
    contribution: message.LogicalContribution,
) -> None:
    if other_key[0] != key[0]:
        return
    is_begun = contribution.commit_lo <= plan.commit_hi
    is_unfinished = plan.commit_lo <= contribution.commit_hi
    if is_begun and is_unfinished:
        raise RuntimeError(
            f"strong window {key} extent {plan.commit_lo}-"
            f"{plan.commit_hi} overlaps unabsorbed logical "
            f"contribution {other_key} extent "
            f"{contribution.commit_lo}-{contribution.commit_hi}"
        )


def _check_every_round_retained(
    pending: _PendingEscalation, payloads: list
) -> None:
    """A strong window starts only once every context round is retained."""
    covered = set()
    for payload in payloads:
        covered.add(payload.round_index)
    plan = pending.resolved_region.plan
    stop_round = plan.context_hi + 1
    needed = set(range(plan.context_lo, stop_round))
    if covered != needed:
        listed = sorted(covered)
        raise RuntimeError(
            f"{pending.label}: strong window submitted with rounds "
            f"{listed} but it needs "
            f"{plan.context_lo}-{plan.context_hi}; a strong window may "
            "only start once every required round is retained"
        )


def _work_record_order(record: tuple) -> bytes:
    return message.stable_identity_order_key(record[0])
