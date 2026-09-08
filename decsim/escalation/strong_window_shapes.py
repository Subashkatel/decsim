"""The strong window's shape: which rounds the strong tier re-decodes, and when.

Two rows of STRONG_WINDOW_SHAPES, named by escalation.strong_window
(Toshio et al. 2510.25222). ContextWindow reads the escalated window's
commit region with one buffer of raw context on each side, built the
moment it is asked for; that geometry is decsim's own, not the paper's
(see ContextWindow). ForwardWindow is Sec. III C and Fig. 12: a strong window
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

The restart window's weak decode may read back into the strong region
from Buffer 0 (escalation.restart_reread_buffer_regions buffer
regions), so those rounds, the last absorbed window's commit rounds,
must still be stored when the plan lands, whether the absorbed
windows' inputs are in flight or already landed in a unit. Every
window an earlier window bounds claims exactly the rounds its restart
decode would read at planning (PotentialRestart,
frontends/planner.py), past its own request and landing; the plan
withdraws the stale weak requests of the windows it rewrites, re-slices
the restart window, requests its weak decode afresh and only then ends
the claim (Sec. III C: the weak decoder resumes past the strong region
once its rounds are stored).

Wide state recorded: ForwardWindow sets nine attributes, the eight
window components its layout touches and its registry of pending
windows; it is one job, the forward window's layout, and the width is
the number of components a re-slice reaches.
"""

import dataclasses
import enum
from typing import Any, Optional, Protocol, runtime_checkable

import decsim.decoders.decode_queue as decode_queue_module
import decsim.escalation.strong_regions as strong_regions
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class StrongAssignment:
    """A strong window assigned to an escalated weak window.

    job is the strong job when the shape builds it now (the context
    window); None when the shape holds it for its far boundary or its
    terminal data (the forward window).
    """

    request_key: window_records.DecoderRequestKey
    job: Optional[decoding_records.DecodeJob]


@dataclasses.dataclass(frozen=True)
class DeferredStrongJob:
    """A held strong job its condition released, with its selection's tick."""

    job: decoding_records.DecodeJob
    selection_arrival_ticks: int


@runtime_checkable
class StrongWindowShape(Protocol):
    """How the strong tier's window is laid out, as the redecode sees it.

    Table rows: two_sided_context and forward (STRONG_WINDOW_SHAPES,
    below). absorbs_weak_windows is the row's own declaration that its
    strong region replaces the weak windows it covers, so the planner
    claims the rounds a restart would read and the weak chain keeps
    committing; a reader of the run's shape asks the row rather than a
    yaml flag. window_absorbed(key, owner_key) is the shape's one trace
    source: the forward window fires it for every weak window a strong
    one covers, and a shape that absorbs nothing exposes the silent
    source, so the machine connects the ledger and the trace without
    asking which shape it built.
    """

    absorbs_weak_windows: bool
    window_absorbed: Any

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
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
    """The commit region and one buffer of raw context on each side.

    The geometry is decsim's own. Toshio et al. 2510.25222 Sec. III A
    feeds the strong decoder the same window as the weak one, "a
    sequence of syndrome data sigma is simultaneously fed to both the
    weak and strong decoders" (lines 599-601), and the formula
    r_strong = r_com + 2 r_buf is Sec. III C, stated with Fig. 12 (lines
    1250-1251) for the forward window. decsim reads context on both
    sides because escalation discards the weak result, which unpins the
    past face, and a buffer of raw context is the standard answer to an
    open face (Skoric 2209.08552 line 388; Tan 2209.09219 line 1021).

    The job is built the moment it is asked for and priced for the
    context rounds that exist: a window at the operation's edge has a
    shorter context than commit + 2 buffer. This shape absorbs no weak
    window, so its window_absorbed source is the silent one.
    """

    absorbs_weak_windows = False
    window_absorbed = trace_source.SILENT

    def __init__(self, engine, regions, tracker, retention, builder) -> None:
        self.engine = engine
        self.regions = regions
        self.tracker = tracker
        self.retention = retention
        self.builder = builder

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """The two-sided context job, built now, its context held."""
        key = (weak_job.operation_id, weak_job.window_id)
        region = self.regions.context_region(key)
        strong_window = region.window
        model = self.regions.context_model(key, strong_window)
        request_key = self.builder.new_request_key(
            weak_job.operation_id,
            weak_job.window_id,
            window_records.DecoderTier.STRONG,
        )
        self._require_context_stored(key, region)
        strong_store = self.retention.strong_store
        self.builder.stamp_first_round(strong_window, strong_store)
        payloads = self.builder.assemble_payloads(strong_window, strong_store)
        payload_round_count = decoding_records.distinct_round_count(payloads)
        job = decoding_records.DecodeJob(
            operation_id=weak_job.operation_id,
            window_id=weak_job.window_id,
            round_count=payload_round_count,
            ready_time=self.engine.now,
            label=weak_job.strong_label,
            kind=decoding_records.DecodeJobKind.STRONG_REDECODE,
            spatial_nodes=weak_job.spatial_nodes,
            code=weak_job.code,
            detector_error_model=model,
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
        self, key: tuple, region: strong_regions.ContextRegion
    ) -> None:
        """Every context round that already arrived must sit in buffer 1."""
        strong_store = self.retention.strong_store
        missing = []
        for round_key in region.context_read_keys:
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

    absorbs_weak_windows = True

    def __init__(
        self,
        engine,
        regions,
        planner,
        retention,
        builder,
        requester,
        ledger,
    ) -> None:
        self.engine = engine
        self.regions = regions
        self.planner = planner
        self.retention = retention
        self.builder = builder
        self.requester = requester
        self.ledger = ledger
        self.pending = _PendingWindows()
        self.window_absorbed = trace_source.TraceSource()

    # ---- the shape

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """Lay out the forward strong window; hold its job until it may start.

        The strong window absorbs the windows it covers. Its job waits for
        the restart window's weak commit (waiting_far_boundary) or, at the
        operation's end, until every clamped strong window round is
        stored (waiting_terminal_data).
        """
        key = (weak_job.operation_id, weak_job.window_id)
        self._refuse_second_escalation(key)
        strong_request_key = self.builder.new_request_key(
            weak_job.operation_id,
            weak_job.window_id,
            window_records.DecoderTier.STRONG,
        )
        strong_request_created_ticks = self.engine.now
        operation_id = key[0]
        resolved_region = self.regions.forward_region(key)
        plan = resolved_region.plan
        restart_key = resolved_region.restart_window_key
        strong_window = resolved_region.strong_window
        strong_model = resolved_region.strong_model
        restart_model = resolved_region.restart_model
        logical_candidate = self._strong_window_ownership_candidate(
            key, resolved_region
        )
        guard = self._guard_restart_reads(
            key,
            restart_key,
            resolved_region.proposed_restart_window,
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
            self._hold_strong_context(
                key,
                strong_request_key,
                resolved_region.context_round_keys,
            )
            self._withdraw_stale_requests(resolved_region)
            pending_hold = decoding_records.PendingStrong(strong_request_key)
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
        stored_through = self.regions.strong_rounds_stored(
            operation_id
        )
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

    def _strong_window_ownership_candidate(
        self, key: tuple, resolved_region: strong_regions.ForwardRegion
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
        candidate[key] = decoding_records.LogicalContribution(
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
        proposed_restart: Optional[window_records.Window],
        strong_request_key: window_records.DecoderRequestKey,
        resolved_region: strong_regions.ForwardRegion,
    ) -> Optional[decoding_records.RephaseGuard]:
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
        guard = decoding_records.RephaseGuard(strong_request_key)
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
        claim = decoding_records.PotentialRestart(restart_key)
        assert self.retention.weak_store.has_hold(claim), (
            f"strong-region plan for {key}: restart window {restart_key}'s "
            f"potential restart hold is no longer live"
        )

    def _guarded_strong_reads(
        self,
        key: tuple,
        restart_key: tuple,
        proposed_restart: window_records.Window,
        resolved_region: strong_regions.ForwardRegion,
    ) -> list:
        strong_store = self.retention.strong_store
        escalated_potential = decoding_records.PotentialStrong(key)
        restart_potential = decoding_records.PotentialStrong(restart_key)
        escalated_identities = strong_store.hold_round_identities(
            escalated_potential
        )
        restart_identities = strong_store.hold_round_identities(
            restart_potential
        )
        guarded = list(escalated_identities)
        guarded += list(restart_identities)
        guarded += list(resolved_region.context_round_keys)
        restart_reads = list(resolved_region.restart_read_keys)
        guarded += self.retention.strong_context_read_keys(
            proposed_restart, restart_reads
        )
        return guarded

    # ---- private: landing the plan

    def _withdraw_stale_requests(
        self, resolved_region: strong_regions.ForwardRegion
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
        strong_request_key: window_records.DecoderRequestKey,
        context_keys: tuple,
    ) -> None:
        """The window's potential strong read becomes the request's hold."""
        self.retention.transfer_potential_to_pending(key, strong_request_key)
        pending_hold = decoding_records.PendingStrong(strong_request_key)
        self.retention.strong_store.replace_hold(
            pending_hold, list(context_keys)
        )

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
            decode_queue_module.LOG_SOURCE,
            f"window {key} absorbed into the strong window "
            f"(weak chain skips it)",
        )

    def _unhook_restart(
        self, key: tuple, restart_key: tuple, window: window_records.Window
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
        absorbed = decoding_records.PotentialStrong(key)
        needed_identities = strong_store.hold_round_identities(absorbed)
        needed = set(needed_identities)
        replacement_identities = strong_store.hold_round_identities(replacement)
        replacements = set(replacement_identities)
        if restart_key is not None:
            restart_potential = decoding_records.PotentialStrong(restart_key)
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
        resolved_region: strong_regions.ForwardRegion,
    ) -> None:
        plan = resolved_region.plan
        readiness_description = "the far-side weak boundary"
        if resolved_region.restart_window_key is None:
            readiness_description = "terminal data"
        absorbed_count = len(resolved_region.absorbed_window_keys)
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
            f"{held.label}: strong window rounds {plan.commit_lo}-"
            f"{plan.commit_hi} assigned; weak chain skips "
            f"{absorbed_count} window(s); "
            f"strong start deferred until {readiness_description}",
        )

    def _restart_weak_chain(
        self, restart_key: tuple, plan: window_records.StrongRegionPlan, model
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
        seam_owner: window_records.SeamFaultOwner,
    ) -> None:
        """Install the restart window's re-sliced reads and their model."""
        restart = self.planner.windows_by_key[restart_key]
        restart.buffer_lo = buffer_lo
        restart.round_count = restart.buffer_hi - restart.buffer_lo + 1
        self.retention.replace_window_reads(restart_key, restart)
        if model is not None:
            self.planner.model_by_window[restart_key] = model
        seam_owner_name = seam_owner.name.lower()
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
            f"restart window {restart_key} re-sliced across strong window "
            f"edge {strong_window_hi} (reads rounds {restart.buffer_lo}-"
            f"{restart.buffer_hi}; crossing faults owned by "
            f"{seam_owner_name})",
        )

    # ---- private: the held job

    def _build_pending_strong_job(
        self, held: "_PendingWindow"
    ) -> decoding_records.DecodeJob:
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
        payload_round_count = decoding_records.distinct_round_count(payloads)
        job = decoding_records.DecodeJob(
            operation_id=key[0],
            window_id=key[1],
            round_count=payload_round_count,
            ready_time=self.engine.now,
            label=held.label,
            kind=decoding_records.DecodeJobKind.STRONG_REDECODE,
            spatial_nodes=weak_job.spatial_nodes,
            code=weak_job.code,
            detector_error_model=held.strong_model,
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


# escalation.strong_window names one of these rows: the shape of the
# window the strong tier re-decodes. The root resolves the name once and
# builds the row with the window components it needs.
STRONG_WINDOW_SHAPES = {
    "two_sided_context": ContextWindow,
    "forward": ForwardWindow,
}


class _Phase(enum.Enum):
    """The one readiness condition that releases a held strong job."""

    WAITING_FAR_BOUNDARY = enum.auto()
    WAITING_TERMINAL_DATA = enum.auto()


@dataclasses.dataclass(frozen=True)
class _PendingWindow:
    """Everything kept from the plan until the one strong-job submission."""

    key: tuple
    weak_job: decoding_records.DecodeJob
    label: str
    resolved_region: strong_regions.ForwardRegion
    strong_window: window_records.Window
    strong_model: object
    selection_arrival_ticks: Optional[int]
    phase: _Phase
    strong_request_key: window_records.DecoderRequestKey
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
            record = (key, phase_name, held.strong_window.round_count)
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
    held: _PendingWindow, job: decoding_records.DecodeJob
) -> DeferredStrongJob:
    """The held window's job with its selection's tick."""
    assert held.selection_arrival_ticks is not None, held.key
    return DeferredStrongJob(job, held.selection_arrival_ticks)


def _refuse_overlapping_contribution(
    key: tuple,
    plan: window_records.StrongRegionPlan,
    other_key: tuple,
    contribution: decoding_records.LogicalContribution,
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
    return identity_records.stable_identity_order_key(record[0])
