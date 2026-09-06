"""The strong window's shape: which rounds the strong tier re-decodes, and when.

Two rows, selected by escalation.double_window (Toshio et al.
2510.25222). ContextWindow is Sec. III A: the escalated window's commit
region with one buffer of raw context on each side (r_strong = r_com +
2 r_buf, text lines 1250-1252 of tmp/papers/txt), built the moment it
is asked for. ForwardWindow is Sec. III C and Fig. 12: a strong window
that starts at the escalated commit and extends forward, absorbs the
weak windows it covers, re-slices the window past it (the restart
window) and is held until that window's weak commit or, at the
operation's end, until its last round is stored. A shape builds the
strong job on the window components (planner, tracker, retention,
builder) and hands it to the StrongRedecode, which submits it
(strong_redecode.py).

Seam modelling of the forward window: both faces are decoded as
two-sided windows, one buffer of raw context per face, exact
fault-ownership partition, no folded decoded defects (folding at a
raw-read face double-counts). Unlike the paper's exactly-r_strong read
with weak-pinned faces, the context reads are extra: seam-edge accuracy
is slightly optimistic, and the strong window is priced for the whole
context it reads rather than the r_strong rounds it commits, so its
decode cost is conservative against Theorem 1 rather than optimistic.

The restart window's weak decode reads its buffer into the strong
region from Buffer 0, so those rounds, the last absorbed window's
commit rounds, must still be stored when the plan lands, whether the
absorbed windows' inputs are in flight or already landed in a unit.
Every window an earlier window bounds claims its reads and one buffer
before them at planning (PotentialRestart, frontends/planner.py), past
its own request and landing; the plan withdraws the stale weak requests
of the windows it rewrites, re-slices the restart window, requests its
weak decode afresh and only then ends the claim (Sec. III C: the weak
decoder resumes past the strong region once its rounds are stored).

Wide state recorded: ForwardWindow sets nine attributes, the eight
window components its layout touches and its registry of pending
windows; it is one job, the forward window's layout, and the width is
the number of components a re-slice reaches.
"""

import copy
import dataclasses
import enum
from typing import Any, Optional, Protocol, runtime_checkable

import decsim.message as message
import decsim.observe.trace_source as trace_source
import decsim.windows.round_retention as round_retention

LOG_SOURCE = "DecoderCluster"


@dataclasses.dataclass(frozen=True)
class StrongAssignment:
    """A strong window assigned to an escalated weak window.

    job is the strong job when the shape builds it now (the context
    window); None when the shape holds it for its far boundary or its
    terminal data (the forward window).
    """

    request_key: message.DecoderRequestKey
    job: Optional[message.DecodeJob]


@dataclasses.dataclass(frozen=True)
class DeferredStrongJob:
    """A held strong job its condition released, with its selection's tick."""

    job: message.DecodeJob
    selection_arrival_ticks: int


@runtime_checkable
class StrongWindowShape(Protocol):
    """How the strong tier's window is laid out, as the redecode sees it.

    window_absorbed(key, owner_key) is the shape's one trace source: the
    forward window fires it for every weak window a strong one covers,
    and a shape that absorbs nothing exposes the silent source, so the
    machine connects the ledger and the trace without asking which shape
    it built.
    """

    window_absorbed: Any

    def plan(self, weak_job: message.DecodeJob) -> StrongAssignment:
        """Assign the strong window; build its job now or hold it."""

    def note_selection_sent(
        self, window_key: tuple, selection_arrival_ticks: int
    ) -> None:
        """The held window's selection left; its arrival tick is known."""

    def take_if_far_boundary_committed(
        self, window_key: tuple
    ) -> Optional[DeferredStrongJob]:
        """The held job whose far boundary this weak commit is, if any."""

    def take_if_terminal_data_stored(
        self, operation_id
    ) -> Optional[DeferredStrongJob]:
        """The held terminal job whose last round is stored, if any."""

    def has_pending(self) -> bool:
        """Whether a strong window is still held."""

    def pending_work(self) -> tuple:
        """The held strong windows as (key, phase, rounds), for the views."""


class ContextWindow:
    """Sec. III A: the commit region and one buffer of context each side.

    The job is built the moment it is asked for and priced for the
    context rounds that exist: a window at the operation's edge has a
    shorter context than commit + 2 buffer. This shape absorbs no weak
    window, so its window_absorbed source is the silent one.
    """

    window_absorbed = trace_source.SILENT

    def __init__(self, engine, planner, tracker, retention, builder) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.builder = builder

    def plan(self, weak_job: message.DecodeJob) -> StrongAssignment:
        """The two-sided context job, built now, its context held."""
        key = (weak_job.op_id, weak_job.window_id)
        weak_window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[weak_job.op_id]
        strong_window = _context_window_of(weak_window)
        round_count = self.tracker.round_count_for_window(
            operation.id, strong_window
        )
        left_exclusions = _left_fault_exclusions(strong_window.commit_lo)
        resolved = self.planner.resolved_operation_by_id[operation.id]
        model = self.planner.models.strong_model_for_operation(
            operation, resolved, strong_window, round_count, left_exclusions
        )
        request_key = self.builder.new_request_key(
            weak_job.op_id, weak_job.window_id, message.DecoderTier.STRONG
        )
        self._require_context_stored(key, weak_job.op_id, strong_window)
        strong_store = self.retention.strong_store
        self.builder.stamp_first_round(strong_window, strong_store)
        payloads = self.builder.assemble_payloads(strong_window, strong_store)
        payload_round_count = message.distinct_round_count(payloads)
        job = message.DecodeJob(
            op_id=weak_job.op_id,
            window_id=weak_job.window_id,
            n_rounds=payload_round_count,
            ready_time=self.engine.now,
            label=weak_job.strong_label,
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
        self.retention.hold_strong_input(job)
        return StrongAssignment(request_key, job)

    def note_selection_sent(
        self, window_key: tuple, selection_arrival_ticks: int
    ) -> None:
        """Nothing is held; the job left with its selection."""
        del window_key
        del selection_arrival_ticks

    def take_if_far_boundary_committed(
        self, window_key: tuple
    ) -> Optional[DeferredStrongJob]:
        """Nothing waits for a far boundary."""
        del window_key
        return None

    def take_if_terminal_data_stored(
        self, operation_id
    ) -> Optional[DeferredStrongJob]:
        """Nothing waits for terminal data."""
        del operation_id
        return None

    def has_pending(self) -> bool:
        """Nothing is ever held."""
        return False

    def pending_work(self) -> tuple:
        """Nothing is ever held."""
        return ()

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


class ForwardWindow:
    """Sec. III C, Fig. 12: a strong window that absorbs what it covers.

    The window starts at the escalated commit and extends forward by
    the interaction's plan; the weak chain skips the windows it absorbs
    and restarts past it on a re-sliced window; the strong result owns
    the whole extent. The job is held until both of its boundaries are
    weak-determined: the commits before it, and the restart window's
    commit or the terminal boundary. The weak pipeline never waits on
    strong work. One strong job per escalation; a second is refused.
    Trace source: window_absorbed(key, owner_key) for every window the
    strong window at owner_key covers.
    """

    def __init__(
        self,
        engine,
        planner,
        tracker,
        retention,
        builder,
        requester,
        ledger,
        interaction,
    ) -> None:
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.builder = builder
        self.requester = requester
        self.ledger = ledger
        self.interaction = interaction
        self.pending = _PendingWindows()
        self.window_absorbed = trace_source.TraceSource()

    # ---- the shape

    def plan(self, weak_job: message.DecodeJob) -> StrongAssignment:
        """Lay out the forward strong window; hold its job until it may start.

        The strong window absorbs the windows it covers. Its job waits for
        the restart window's weak commit (waiting_far_boundary) or, at the
        operation's end, until every clamped strong window round is
        stored (waiting_terminal_data).
        """
        key = (weak_job.op_id, weak_job.window_id)
        self._refuse_second_escalation(key)
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
        phase = _Phase.WAITING_FAR_BOUNDARY
        if restart_key is None:
            phase = _Phase.WAITING_TERMINAL_DATA
        held = _PendingWindow(
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
            self.pending.register(held, operation_id, restart_key)
            self._hold_strong_context(key, strong_request_key, plan)
            self._withdraw_stale_requests(resolved_region)
            pending_hold = message.PendingStrong(strong_request_key)
            for absorbed_key in resolved_region.absorbed_window_keys:
                self._absorb_window(absorbed_key, restart_key, pending_hold)
                self.window_absorbed.fire(absorbed_key, key)
            self._log_assignment(held, resolved_region)
            if restart_key is not None:
                self._restart_weak_chain(restart_key, plan, restart_model)
            return StrongAssignment(strong_request_key, None)
        finally:
            if guard is not None:
                self.retention.release_hold_if_live(
                    guard, self.retention.strong_store
                )

    def note_selection_sent(
        self, window_key: tuple, selection_arrival_ticks: int
    ) -> None:
        """The held window's selection left; its arrival tick is recorded."""
        held = self.pending.peek_key(window_key)
        self.pending.update_selection_arrival(held, selection_arrival_ticks)

    def take_if_far_boundary_committed(
        self, window_key: tuple
    ) -> Optional[DeferredStrongJob]:
        """The job held for this far boundary, built now, if there is one."""
        held = self.pending.peek_far(window_key)
        if held is None:
            return None
        job = self._build_pending_strong_job(held)
        self.pending.take_far(window_key, held)
        return _released(held, job)

    def take_if_terminal_data_stored(
        self, operation_id
    ) -> Optional[DeferredStrongJob]:
        """The operation's terminal job, built now once its tail is stored."""
        held = self.pending.peek_terminal(operation_id)
        if held is None:
            return None
        stored_through = self.tracker.strong_rounds_arrived(operation_id)
        if stored_through < held.resolved_region.plan.context_hi:
            return None
        job = self._build_pending_strong_job(held)
        self.pending.take_terminal(operation_id, held)
        return _released(held, job)

    def has_pending(self) -> bool:
        """Whether a strong window is still held for its condition."""
        return bool(self.pending.by_key)

    def pending_work(self) -> tuple:
        """The held strong windows without the live windows."""
        return self.pending.work()

    # ---- private: the plan

    def _refuse_second_escalation(self, key: tuple) -> None:
        if self.pending.peek_key(key) is not None:
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
        """Check the plan against the live window graph; resolve its reads.

        The interaction that planned the region is a plug-in, so its
        plan is checked like an input.
        """
        _check_region_bounds(key, weak_window, round_count, plan)
        absorbed = _absorbed_window_keys(later_windows, plan)
        _refuse_crossing_window(later_windows, plan)
        self._check_absorbable(absorbed)
        restart_key = _restart_window_key(later_windows, plan)
        restart_reads = self._restart_reads(key, restart_key, plan)
        strong_exclusions, restart_exclusions = _fault_exclusions(
            plan, round_count, restart_key
        )
        context_reads = _context_round_keys(weak_window.op_id, plan)
        purpose = f"strong-region plan for {key}"
        # the strong window's context lives in syndrome buffer 1; the
        # restart window's weak reads, the re-read range among them, sit
        # in Buffer 0 under its potential restart hold (planned with the
        # window, live until _restart_weak_chain ends it)
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

    def _check_absorbable(self, absorbed: tuple) -> None:
        for absorbed_key in absorbed:
            window = self.planner.windows_by_key[absorbed_key]
            assert not window.committed, f"absorbing committed {absorbed_key}"
            assert window.t_done is None, f"absorbing decoded {absorbed_key}"

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
        """Hold the restart window's strong context while the plan lands.

        Syndrome buffer 1 loses the absorbed windows' potential strong
        holds as the plan lands; the guard keeps the restart window's
        context until its re-sliced potential strong hold names it. Its
        Buffer 0 reads need no guard: its potential restart hold is
        live until the plan ends (_restart_weak_chain).
        """
        if restart_key is None:
            return None
        self._require_restart_claim(key, restart_key)
        guard = message.RephaseGuard(strong_request_key)
        guarded_strong = self._guarded_strong_reads(
            key, restart_key, proposed_restart, resolved_region
        )
        self.retention.strong_store.register_hold(guard, guarded_strong)
        return guard

    def _require_restart_claim(self, key: tuple, restart_key: tuple) -> None:
        """The restart window still claims its reads and the re-read range.

        The claim ends only when the window before the restart window
        commits, at absorption, or at the end of this plan, and the
        absorbed windows are checked uncommitted first, so no runtime
        path reaches a released claim here: an invariant, not a check.
        """
        claim = message.PotentialRestart(restart_key)
        assert self.retention.weak_store.has_hold(claim), (
            f"strong-region plan for {key}: restart window {restart_key}'s "
            f"potential restart hold is no longer live"
        )

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

    # ---- private: landing the plan

    def _withdraw_stale_requests(
        self, resolved_region: "_ResolvedStrongRegion"
    ) -> None:
        """Take back the weak decodes the strong window supersedes.

        An absorbed window's request, and the restart window's request
        built on its old shape: early-shipped at data-complete, parked
        on the escalated window's boundary, which never arrives. Their
        Buffer 0 holds end with them, in flight or landed; the restart
        window's potential restart hold keeps every round its re-sliced
        decode reads, the re-read range among them, until the plan ends.
        """
        stale_keys = list(resolved_region.absorbed_window_keys)
        if resolved_region.restart_window_key is not None:
            stale_keys.append(resolved_region.restart_window_key)
        for window_key in stale_keys:
            window = self.planner.windows_by_key[window_key]
            if window.queued:
                self.requester.withdraw(window)

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
        assert not window.queued, f"absorbing queued {key}"
        assert not window.committed, f"absorbing committed {key}"
        window.queued = True  # keeps the requester away
        window.committed = True
        window.is_absorbed = True
        if restart_key is not None:
            self._unhook_restart(key, restart_key, window)
        self.retention.release_hold_if_live(key)
        self.retention.release_restart_reads(key)
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
        held: "_PendingWindow",
        resolved_region: "_ResolvedStrongRegion",
    ) -> None:
        plan = resolved_region.plan
        readiness_description = "the far-side weak boundary"
        if resolved_region.restart_window_key is None:
            readiness_description = "terminal data"
        absorbed_count = len(resolved_region.absorbed_window_keys)
        self.engine.log(
            LOG_SOURCE,
            f"{held.label}: strong window rounds {plan.commit_lo}-"
            f"{plan.commit_hi} assigned; weak chain skips "
            f"{absorbed_count} window(s); "
            f"strong start deferred until {readiness_description}",
        )

    def _restart_weak_chain(
        self, restart_key: tuple, plan: message.StrongRegionPlan, model
    ) -> None:
        """Re-slice the restart window and request its weak decode afresh.

        Its rounds pass from its potential restart hold to its own hold
        or to the fresh request's, with no gap; the claim ends here,
        since no earlier escalation remains to re-slice it.
        """
        self._reslice_restart_window(
            restart_key,
            plan.restart_buffer_lo,
            model,
            plan.commit_hi,
            plan.restart_seam_fault_owner,
        )
        restart = self.planner.windows_by_key[restart_key]
        # its absorbed dependency is gone; no strong sibling in the
        # forward scheme
        self.requester.request_if_ready(restart, None)
        self.retention.release_restart_reads(restart_key)

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

    # ---- private: the held job

    def _build_pending_strong_job(
        self, held: "_PendingWindow"
    ) -> message.DecodeJob:
        """The strong window's job, once both of its boundaries exist.

        The strong window commits all r_strong rounds and reads one
        buffer of raw context per face, owning nothing that touches
        rounds before its extent.
        """
        key = held.key
        weak_job = held.weak_job
        strong_window = held.strong_window
        strong_store = self.retention.strong_store
        payloads = self.builder.assemble_payloads(strong_window, strong_store)
        _check_every_round_retained(held, payloads)
        self.builder.stamp_first_round(strong_window, strong_store)
        payload_round_count = message.distinct_round_count(payloads)
        job = message.DecodeJob(
            op_id=key[0],
            window_id=key[1],
            n_rounds=payload_round_count,
            ready_time=self.engine.now,
            label=held.label,
            hint="strong",
            spatial_nodes=weak_job.spatial_nodes,
            code=weak_job.code,
            dem=held.strong_model,
            payloads=payloads,
            attempt=1,
            window=strong_window,
            strong_decode_for=key,
            request_key=held.strong_request_key,
            request_created_ticks=held.strong_request_created_ticks,
            gate=self.builder,
        )
        self.retention.hold_strong_input(job)
        return job


@dataclasses.dataclass(frozen=True)
class _ResolvedStrongRegion:
    """One planned strong region resolved against the live window graph."""

    plan: message.StrongRegionPlan
    absorbed_window_keys: tuple
    restart_window_key: Optional[tuple]
    restart_read_keys: tuple
    strong_fault_exclusion_ranges: tuple
    restart_fault_exclusion_ranges: Optional[tuple]


class _Phase(enum.Enum):
    """The one readiness condition that releases a held strong job."""

    WAITING_FAR_BOUNDARY = enum.auto()
    WAITING_TERMINAL_DATA = enum.auto()


@dataclasses.dataclass(frozen=True)
class _PendingWindow:
    """Everything kept from the plan until the one strong-job submission."""

    key: tuple
    weak_job: message.DecodeJob
    label: str
    resolved_region: _ResolvedStrongRegion
    strong_window: message.Window
    strong_model: object
    selection_arrival_ticks: Optional[int]
    phase: _Phase
    strong_request_key: message.DecoderRequestKey
    strong_request_created_ticks: int


class _PendingWindows:
    """The held strong windows by window, each in one readiness index.

    Its invariants (one phase per index, one entry per window, a take
    that matches its register) are the shape's own and are asserted.
    """

    def __init__(self) -> None:
        self.by_key: dict = {}
        self.key_by_far_boundary: dict = {}
        self.key_by_terminal_operation: dict = {}

    def register(
        self, held: _PendingWindow, operation_id, restart_key: Optional[tuple]
    ) -> None:
        """Index the held window by its far boundary, or by its operation."""
        if restart_key is None:
            self._register(
                held,
                _Phase.WAITING_TERMINAL_DATA,
                self.key_by_terminal_operation,
                operation_id,
            )
            return
        self._register(
            held,
            _Phase.WAITING_FAR_BOUNDARY,
            self.key_by_far_boundary,
            restart_key,
        )

    def update_selection_arrival(
        self, expected: _PendingWindow, selection_arrival_ticks: int
    ) -> _PendingWindow:
        """Record the selection's expected arrival without moving ownership."""
        assert self.by_key.get(expected.key) is expected, expected.key
        assert expected.selection_arrival_ticks is None, expected.key
        updated = dataclasses.replace(
            expected, selection_arrival_ticks=selection_arrival_ticks
        )
        self.by_key[expected.key] = updated
        return updated

    def peek_key(self, key: tuple) -> Optional[_PendingWindow]:
        return self.by_key.get(key)

    def peek_far(self, far_boundary_key: tuple) -> Optional[_PendingWindow]:
        key = self.key_by_far_boundary.get(far_boundary_key)
        if key is None:
            return None
        return self.by_key[key]

    def peek_terminal(self, operation_id) -> Optional[_PendingWindow]:
        key = self.key_by_terminal_operation.get(operation_id)
        if key is None:
            return None
        return self.by_key[key]

    def take_far(
        self, far_boundary_key: tuple, expected: _PendingWindow
    ) -> None:
        self._take(
            expected,
            _Phase.WAITING_FAR_BOUNDARY,
            self.key_by_far_boundary,
            far_boundary_key,
        )

    def take_terminal(self, operation_id, expected: _PendingWindow) -> None:
        self._take(
            expected,
            _Phase.WAITING_TERMINAL_DATA,
            self.key_by_terminal_operation,
            operation_id,
        )

    def work(self) -> tuple:
        """The held windows as (key, phase name, rounds), in stable order."""
        phase_names = {
            _Phase.WAITING_FAR_BOUNDARY: "waiting_far_boundary",
            _Phase.WAITING_TERMINAL_DATA: "waiting_terminal_data",
        }
        records = []
        for key, held in self.by_key.items():
            phase_name = phase_names[held.phase]
            record = (key, phase_name, held.strong_window.n_rounds)
            records.append(record)
        ordered = sorted(records, key=_work_record_order)
        return tuple(ordered)

    def _register(
        self,
        held: _PendingWindow,
        expected_phase: _Phase,
        readiness_index: dict,
        readiness_key,
    ) -> None:
        assert held.phase is expected_phase, held.key
        assert held.key not in self.by_key, held.key
        assert readiness_key not in readiness_index, readiness_key
        self.by_key[held.key] = held
        readiness_index[readiness_key] = held.key

    def _take(
        self,
        expected: _PendingWindow,
        expected_phase: _Phase,
        readiness_index: dict,
        readiness_key,
    ) -> None:
        assert expected.phase is expected_phase, expected.key
        assert self.by_key.get(expected.key) is expected, expected.key
        assert readiness_index.get(readiness_key) == expected.key, readiness_key
        del readiness_index[readiness_key]
        del self.by_key[expected.key]


def _released(
    held: _PendingWindow, job: message.DecodeJob
) -> DeferredStrongJob:
    """The held window's job with its selection's tick."""
    assert held.selection_arrival_ticks is not None, held.key
    return DeferredStrongJob(job, held.selection_arrival_ticks)


def _context_window_of(weak_window: message.Window) -> message.Window:
    """The two-sided context window of a weak window (Sec. III A)."""
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
    """A window committing across the strong window's edge has no owner.

    The settings refuse the shape at build for the shipped interaction
    (policies.py, _refuse_crossing_strong_region); this is the contract
    behind it, since the interaction that planned the region is a
    plug-in and its plan is checked like an input.
    """
    for window in later_windows:
        is_begun = window.commit_lo <= plan.commit_hi
        is_unfinished = plan.commit_hi < window.commit_hi
        if is_begun and is_unfinished:
            raise RuntimeError(
                f"window {window.key} commits {window.commit_lo}-"
                f"{window.commit_hi} across the strong-region edge "
                f"{plan.commit_hi}"
            )


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


def _check_every_round_retained(held: _PendingWindow, payloads: list) -> None:
    """A strong window starts only once every context round is retained."""
    covered = set()
    for payload in payloads:
        covered.add(payload.round_index)
    plan = held.resolved_region.plan
    stop_round = plan.context_hi + 1
    needed = set(range(plan.context_lo, stop_round))
    if covered != needed:
        listed = sorted(covered)
        raise RuntimeError(
            f"{held.label}: strong window submitted with rounds "
            f"{listed} but it needs "
            f"{plan.context_lo}-{plan.context_hi}; a strong window may "
            "only start once every required round is retained"
        )


def _work_record_order(record: tuple) -> bytes:
    return message.stable_identity_order_key(record[0])
