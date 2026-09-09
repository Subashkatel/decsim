"""The strong window's shape: which rounds the strong tier re-decodes, and when.

Four rows of STRONG_WINDOW_SHAPES (escalation/settings.py), named by
escalation.strong_window
(Toshio et al. 2510.25222). Two read every face raw and two pin a face
on a neighbour's committed correction, which is Bombin et al.
2303.04846's input adaptation (lines 775-788): NearSeamWindow is the
context row's commit region with its past face pinned, and
ForwardSeamWindow is the forward row's extent with both faces pinned. A
fifth shape, both faces pinned and absorbing nothing, is not a row: it
waits for the window after it, which waits for its own strong result,
and the serial sliding chain deadlocks (design audit note 21). What
would make it a row is a windowing scheme whose windows do not commit in
one serial chain, the shape Skoric et al. 2209.08552 decode block by
block (lines 398-401, 1038-1040); the row would read that off a fact the
scheme declares, the way it reads absorption off itself, and refuse a
scheme that does not declare it.
ContextWindow reads the escalated window's
commit region with one buffer of raw context on each side, built the
moment it is asked for; that geometry is decsim's own, not the paper's
(see ContextWindow). ForwardWindow is Sec. III C and Fig. 12: a strong window
that starts at the escalated commit and extends forward, absorbs the
weak windows it covers, re-slices the window past it (the restart
window) and is held until that window's weak commit or, at the
operation's end, until its last round is stored. A shape builds the
strong job on the window components (planner, tracker, retention,
builder) and hands it to the StrongRedecode, which submits it
(strong_redecode.py); a row that cannot build its job at the escalation
declares what releases it instead (pending_strong_windows.py), and the
redecode holds the assignment until those conditions fire.

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

Every row is built with one StrongWindowCollaborators record, so the
root wires a shape without knowing which geometry it is and a row reads
only the components its own layout needs.
"""

import dataclasses
from typing import Any, Optional, Protocol, runtime_checkable

import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.strong_regions as strong_regions
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source


@dataclasses.dataclass(frozen=True)
class StrongAssignment:
    """A strong window assigned to an escalated weak window.

    job is the strong job when the row builds it now (the context
    window); None when the row holds it until the conditions it declares
    fire (the forward window). held_plan is then the row's own record of
    what it planned, handed back to the row when the redecode asks for
    the job, and round_count is the strong window's rounds, which the
    views name while the job is held. folded_boundaries names the
    neighbour windows whose committed boundary conditions the row folds
    into the job's input, Bombin et al. 2303.04846's input adaptation
    (lines 775-788); both shipped rows fold none and read raw rounds.
    """

    request_key: window_records.DecoderRequestKey
    job: Optional[decoding_records.DecodeJob]
    held_plan: Any = None
    round_count: int = 0
    folded_boundaries: tuple = ()


@dataclasses.dataclass(frozen=True)
class StrongWindowCollaborators:
    """The window components every strong window shape is built on.

    One record so every row of STRONG_WINDOW_SHAPES has one constructor
    signature and the root builds a row without asking which geometry it
    is; a row reads the components its own layout needs and ignores the
    rest. This is gem5's params object, where a SimObject's collaborators
    arrive as one structure rather than as a signature per subclass
    (tmp/resources/gem5/src/python/m5/SimObject.py:204-205). The courier
    is what a row with a pinned face reads: the committed boundary of
    the neighbour it pins on, and the hop that carries it.
    """

    engine: engine_module.Engine
    regions: strong_regions.StrongRegions
    planner: ports.WindowPlan
    retention: ports.WindowRetention
    builder: ports.WindowJobBuilder
    requester: ports.WindowRequests
    ledger: ports.LogicalLedger
    courier: ports.BoundaryCourier


@runtime_checkable
class StrongWindowShape(Protocol):
    """How the strong tier's window is laid out, as the redecode sees it.

    Every row of STRONG_WINDOW_SHAPES (escalation/settings.py)
    implements it, and escalation.strong_window names one. A row that
    cannot build its job at the escalation returns an
    assignment with no job and declares what releases it
    (release_conditions), and the redecode asks held_job for the job when
    those conditions fire; the row never learns which hook rang.
    absorbs_weak_windows is the row's own declaration that its
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

    def release_conditions(
        self, assignment: StrongAssignment
    ) -> pending_strong_windows.ReleaseConditions:
        """What must happen before a held job may be built."""

    def held_job(
        self, assignment: StrongAssignment
    ) -> Optional[decoding_records.DecodeJob]:
        """The held job, once its rounds are there; None while they are not."""


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

    The job is built as soon as its context is stored in syndrome
    buffer 1, and priced for the context rounds that exist: a window at
    the operation's edge has a shorter context than commit + 2 buffer.
    A context round still crossing controller_to_strong_buffer holds the
    job instead, because Step 1 feeds both decoders the same data and a
    model that prices transport starts the strong decoder when its copy
    lands (the paper's Monte-Carlo simulations set T_comm^strong to ten
    times T_comm^weak, lines 1109-1114; its Table I is a notation table
    and prices nothing). This shape absorbs no weak window, so its
    window_absorbed source is the silent one.

    Both faces are read raw, so the row folds no neighbour boundary
    (FOLDS_NO_BOUNDARY): a mask on the commit_lo layer would flip a seam
    whose rounds the input already carries as raw defects, which is the
    double count Bombin et al. 2303.04846 lines 775-788 rule out.
    """

    absorbs_weak_windows = False
    pins_the_far_face = False
    window_absorbed = trace_source.SILENT

    def __init__(self, collaborators: StrongWindowCollaborators) -> None:
        self.collaborators = collaborators

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """The two-sided context job, built now or held for its context."""
        key = (weak_job.operation_id, weak_job.window_id)
        region = self.collaborators.regions.context_region(key)
        held = _held_redo(
            self.collaborators,
            weak_job,
            region.window,
            region.context_read_keys,
            FOLDS_NO_BOUNDARY,
        )
        return _assignment_of(self.collaborators, held)

    def release_conditions(
        self, assignment: StrongAssignment
    ) -> pending_strong_windows.ReleaseConditions:
        """The rounds of its own context, stored in syndrome buffer 1."""
        return _stored_rounds_conditions(assignment.held_plan)

    def held_job(
        self, assignment: StrongAssignment
    ) -> Optional[decoding_records.DecodeJob]:
        """The job, once every context round that arrived is stored."""
        return _job_once_stored(self.collaborators, assignment.held_plan)


class NearSeamWindow:
    """The commit region with its past face pinned and one open buffer.

    The escalated window's commit region is re-decoded on an input
    whose past face carries the correction the earlier neighbour
    committed: Bombin et al. 2303.04846 lines 775-788, "the input
    instance to decoder j will consist of the syndrome d(e + kappa_Pj)",
    which sets a boundary condition for the task. A pinned face needs no
    buffer behind it, "using no extra buffers as their boundary
    conditions are now fixed" (lines 1456-1458), so the row reads the
    commit region and one trailing buffer for its open future face,
    which is the b >= d of lines 850-852; a run whose
    windows.buffer_rounds is below the code distance is outside that
    condition, which is a statement about the configuration and not
    about the row. Tan et al. 2209.09219 lines 947-949 name the same
    shape on the weak tier: "it has a closed past boundary and an open
    future boundary".

    The pin is on whatever the neighbour committed (Toshio et al.
    2510.25222 line 1250, boundary conditions "determined by the weak
    decoder"). This row absorbs no weak window, so serial switching
    gives it Held boundaries (escalation/policies.py, check_plan), under
    which a committed boundary is a final one; and the neighbour has
    committed by the time its dependent escalates, since a window's weak
    decode starts only once every boundary it owes has arrived and the
    escalation follows that decode. The escalated window that has no
    earlier neighbour pins nothing: its past face is the operation's
    first round layer, closed by the initialisation.

    The job waits for what the two-sided context row waits for, its own
    rounds stored in syndrome buffer 1, and absorbs nothing, so the weak
    chain runs on untouched and there is no restart window.
    """

    absorbs_weak_windows = False
    pins_the_far_face = False
    window_absorbed = trace_source.SILENT

    def __init__(self, collaborators: StrongWindowCollaborators) -> None:
        self.collaborators = collaborators

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """The near-pinned job, built now or held for its own rounds."""
        key = (weak_job.operation_id, weak_job.window_id)
        region = self.collaborators.regions.near_seam_region(key)
        pinned_source_key = self.collaborators.regions.near_seam_source(key)
        folded_boundaries = _declared_faces(pinned_source_key)
        held = _held_redo(
            self.collaborators,
            weak_job,
            region.window,
            region.context_read_keys,
            folded_boundaries,
        )
        return _assignment_of(self.collaborators, held)

    def release_conditions(
        self, assignment: StrongAssignment
    ) -> pending_strong_windows.ReleaseConditions:
        """The rounds it reads, stored in syndrome buffer 1."""
        return _stored_rounds_conditions(assignment.held_plan)

    def held_job(
        self, assignment: StrongAssignment
    ) -> Optional[decoding_records.DecodeJob]:
        """The job, once every round it reads is stored."""
        return _job_once_stored(self.collaborators, assignment.held_plan)


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
    pins_the_far_face = False

    def __init__(self, collaborators: StrongWindowCollaborators) -> None:
        self.collaborators = collaborators
        self.window_absorbed = trace_source.TraceSource()

    # ---- the shape

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """Lay out the forward strong window; hold its job until it may start.

        The strong window absorbs the windows it covers. Its job waits for
        the restart window's weak commit (waiting_far_boundary) or, at the
        operation's end, until every clamped strong window round is
        stored (waiting_terminal_data); release_conditions declares which
        of the two, and the redecode holds the assignment until it fires.
        """
        key = (weak_job.operation_id, weak_job.window_id)
        self._refuse_second_escalation(key)
        strong_request_key = self.collaborators.builder.new_request_key(
            weak_job.operation_id,
            weak_job.window_id,
            window_records.DecoderTier.STRONG,
        )
        strong_request_created_ticks = self.collaborators.engine.now
        resolved_region = self._resolved_region(key)
        folded_boundaries = self._declared_faces(key, resolved_region)
        plan = resolved_region.plan
        restart_key = resolved_region.restart_window_key
        strong_window = resolved_region.strong_window
        strong_model = resolved_region.strong_model
        restart_model = resolved_region.restart_model
        guard = self.collaborators.retention.guard_restart_reads(
            key,
            restart_key,
            resolved_region.proposed_restart_window,
            strong_request_key,
            resolved_region.context_round_keys,
            resolved_region.restart_read_keys,
        )
        operation = self.collaborators.regions.operation(weak_job.operation_id)
        held = _HeldForwardWindow(
            key=key,
            weak_job=weak_job,
            label=weak_job.strong_label,
            resolved_region=resolved_region,
            strong_window=strong_window,
            strong_model=strong_model,
            operation=operation,
            strong_request_key=strong_request_key,
            strong_request_created_ticks=strong_request_created_ticks,
            folded_boundaries=folded_boundaries,
        )
        try:
            self.collaborators.ledger.replace_contributions(
                key,
                plan.commit_lo,
                plan.commit_hi,
                resolved_region.absorbed_window_keys,
            )
            self.collaborators.retention.hold_strong_context(
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
            return StrongAssignment(
                strong_request_key,
                None,
                held_plan=held,
                round_count=strong_window.round_count,
                folded_boundaries=folded_boundaries,
            )
        finally:
            if guard is not None:
                self.collaborators.retention.release_strong_hold_if_live(guard)

    def release_conditions(
        self, assignment: StrongAssignment
    ) -> pending_strong_windows.ReleaseConditions:
        """The restart window's commit, or the operation's stored tail.

        A strong window bounded by a later weak window waits for that
        window to commit; one at the end of the stream has no later
        window, so it waits for the rounds it reads to be stored
        (Toshio et al. 2510.25222 lines 1248-1250).
        """
        held = assignment.held_plan
        restart_key = held.resolved_region.restart_window_key
        if restart_key is None:
            return pending_strong_windows.ReleaseConditions(
                stored_data_of_operation=held.key[0],
                name="terminal_data",
                released_description="terminal data complete",
            )
        return pending_strong_windows.ReleaseConditions(
            committed_windows=(restart_key,),
            name="far_boundary",
            released_description="far-side weak boundary determined",
        )

    def held_job(
        self, assignment: StrongAssignment
    ) -> Optional[decoding_records.DecodeJob]:
        """The held job, once the rounds it reads are stored.

        A window waiting on a later weak commit reads rounds that were
        stored long before that commit; a terminal one is released by
        the stored round itself, so it is the one that can be asked
        before its tail is there.
        """
        held = assignment.held_plan
        if held.resolved_region.restart_window_key is None:
            stored_through = self.collaborators.regions.strong_rounds_stored(
                held.key[0]
            )
            if stored_through < held.resolved_region.plan.context_hi:
                return None
        return self._build_strong_job(held)

    # ---- the two facts a row of this geometry declares

    def _resolved_region(self, key: tuple) -> strong_regions.ForwardRegion:
        """The extent, read with one buffer of raw context per face."""
        return self.collaborators.regions.forward_region(key)

    def _declared_faces(
        self, key: tuple, resolved_region: strong_regions.ForwardRegion
    ) -> tuple:
        """Which neighbours' boundaries the row folds in: none here.

        Both faces are read raw, and a mask on a raw-read face would
        double count the rounds behind it (Bombin et al. 2303.04846
        lines 775-788).
        """
        del key
        del resolved_region
        return FOLDS_NO_BOUNDARY

    # ---- private: the plan

    def _refuse_second_escalation(self, key: tuple) -> None:
        # the plan claims the extent in the ledger before it holds
        # anything, so an escalation of a window already claimed is the
        # second one, held or committed
        if self.collaborators.ledger.owns_strong_window(key):
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one "
                f"switching event creates exactly one strong job"
            )

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
            window = self.collaborators.planner.window_at(window_key)
            if window.queued:
                self.collaborators.requester.withdraw(window)

    def _absorb_window(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """A window the strong window covers is never weak-decoded.

        It counts committed with no logical contribution, the restart
        window no longer waits for it, and its rounds are the strong
        request's.
        """
        self.collaborators.planner.absorb_window(key, restart_key)
        reads = decoding_records.WindowReads(key)
        self.collaborators.retention.release_hold_if_live(reads)
        self.collaborators.retention.release_restart_reads(key)
        self.collaborators.retention.release_absorbed_strong_hold(
            key, restart_key, replacement
        )
        self.collaborators.engine.log(
            log_sources.DECODER_MANAGER,
            f"window {key} absorbed into the strong window "
            f"(weak chain skips it)",
        )

    def _log_assignment(
        self,
        held: "_HeldForwardWindow",
        resolved_region: strong_regions.ForwardRegion,
    ) -> None:
        plan = resolved_region.plan
        readiness_description = "the far-side weak boundary"
        if resolved_region.restart_window_key is None:
            readiness_description = "terminal data"
        absorbed_count = len(resolved_region.absorbed_window_keys)
        self.collaborators.engine.log(
            log_sources.DECODER_MANAGER,
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
        restart = self.collaborators.planner.window_at(restart_key)
        # its absorbed dependency is gone; no strong sibling in the
        # forward scheme
        self.collaborators.requester.request_if_ready(restart, None)
        self.collaborators.retention.release_restart_reads(restart_key)

    def _reslice_restart_window(
        self,
        restart_key: tuple,
        buffer_lo: int,
        model,
        strong_window_hi: int,
        seam_owner: window_records.SeamFaultOwner,
    ) -> None:
        """Install the restart window's re-sliced reads and their model."""
        restart = self.collaborators.planner.reslice_window(
            restart_key, buffer_lo, model
        )
        self.collaborators.retention.replace_window_reads(restart_key, restart)
        seam_owner_name = seam_owner.name.lower()
        self.collaborators.engine.log(
            log_sources.DECODER_MANAGER,
            f"restart window {restart_key} re-sliced across strong window "
            f"edge {strong_window_hi} (reads rounds {restart.buffer_lo}-"
            f"{restart.buffer_hi}; crossing faults owned by "
            f"{seam_owner_name})",
        )

    # ---- private: the held job

    def _build_strong_job(
        self, held: "_HeldForwardWindow"
    ) -> decoding_records.DecodeJob:
        """The strong window's job, once both of its boundaries exist.

        The strong window commits all r_strong rounds and reads one
        buffer of raw context per face, owning nothing that touches
        rounds before its extent.
        """
        key = held.key
        weak_job = held.weak_job
        strong_window = held.strong_window
        payloads = strong_job_payloads(
            self.collaborators,
            strong_window,
            held.strong_model,
            held.operation,
            held.strong_request_key,
            held.folded_boundaries,
        )
        plan = held.resolved_region.plan
        self.collaborators.retention.require_rounds_retained(
            held.label, payloads, plan.context_lo, plan.context_hi
        )
        payload_round_count = decoding_records.distinct_round_count(payloads)
        job = decoding_records.DecodeJob(
            operation_id=key[0],
            window_id=key[1],
            round_count=payload_round_count,
            ready_time=self.collaborators.engine.now,
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
            gate=self.collaborators.builder.gate,
        )
        self.collaborators.retention.hold_strong_input(job)
        return job


class ForwardSeamWindow(ForwardWindow):
    """Sec. III C, Fig. 12 read as the paper states it: both faces pinned.

    The extent is the forward row's, r_strong = r_com + 2 r_buf rounds
    from the escalated window's commit start (Toshio et al. 2510.25222
    line 1250), and it is read with no context at all: "the strong
    decoder processes all the assigned data at once, after the boundary
    conditions at both ends have been determined by the weak decoder"
    (lines 1248-1250), and a determined boundary condition needs no
    buffer behind it (Bombin et al. 2303.04846 lines 1456-1458). At
    r_com = r_buf = d the row reads 3d rounds where the shipped forward
    row reads 5d. Tan et al. 2209.09219 lines 1026-1029 call a window
    closed at both ends a type-2 window, "the entire window is the core
    region".

    The near face is pinned on the window before the escalated one, the
    far face on the restart window's weak commit, which is exactly the
    boundary the shipped forward row already waits for, so the wait does
    not change and the circular wait that a non-absorbing both-faces row
    runs into (design audit note 21) does not arise: this row absorbs the
    windows it covers, the weak chain keeps committing, and the restart
    window commits the rounds past the strong region.

    A strong window at the end of the operation has no later window, so
    it has no far pin: its future face is closed by the readout and it
    waits for the terminal data, which is what Tan says of the last
    window (2209.09219 lines 953-955, "both time boundaries of the last
    windows are closed").

    A near face that no committed correction closes is read instead of
    pinned, with one buffer region of raw context (Bombin lines
    850-852). That is the window that restarts the weak chain after an
    earlier strong region: the absorption takes its dependency out of
    the chain, so nothing hands it the rounds before its own, and its
    own weak decode reads that face open too.

    The row needs escalation.restart_reread_buffer_regions 0, the
    paper's value: with a re-read the restart window shares rounds with
    the strong region, and pinning the far face on a correction that
    explains rounds inside the region is the double count Bombin's
    input adaptation rules out. The escalation section refuses the
    pairing at load.
    """

    pins_the_far_face = True

    def _resolved_region(self, key: tuple) -> strong_regions.ForwardRegion:
        """The extent, read with no context on the faces it pins."""
        near_source_key = self.collaborators.regions.near_seam_source(key)
        pins_near_face = near_source_key is not None
        return self.collaborators.regions.forward_seam_region(
            key, pins_near_face=pins_near_face
        )

    def _declared_faces(
        self, key: tuple, resolved_region: strong_regions.ForwardRegion
    ) -> tuple:
        """The window before the extent, and the window that restarts after it.

        The restart window's weak commit is the far boundary (Toshio et
        al. 2510.25222 lines 1253-1259); a terminal region has no
        restart window and no far pin.
        """
        near_source_key = self.collaborators.regions.near_seam_source(key)
        faces = _declared_faces(near_source_key)
        restart_key = resolved_region.restart_window_key
        if restart_key is None:
            return faces
        return faces + (restart_key,)


# both shipped rows read raw rounds on both faces and fold no committed
# neighbour boundary into the strong job's input
FOLDS_NO_BOUNDARY: tuple = ()


def _declared_faces(pinned_source_key: Optional[tuple]) -> tuple:
    """The faces a row pins, as StrongAssignment names them."""
    if pinned_source_key is None:
        return FOLDS_NO_BOUNDARY
    return (pinned_source_key,)


def _held_redo(
    collaborators: StrongWindowCollaborators,
    weak_job: decoding_records.DecodeJob,
    strong_window: window_records.Window,
    read_keys: tuple,
    folded_boundaries: tuple,
) -> "_HeldStrongRedo":
    """What a row keeps from its plan: its window, its reads, its faces."""
    key = (weak_job.operation_id, weak_job.window_id)
    model = collaborators.regions.redecode_model(key, strong_window)
    request_key = collaborators.builder.new_request_key(
        weak_job.operation_id,
        weak_job.window_id,
        window_records.DecoderTier.STRONG,
    )
    operation = collaborators.regions.operation(weak_job.operation_id)
    return _HeldStrongRedo(
        key=key,
        weak_job=weak_job,
        label=weak_job.strong_label,
        read_keys=read_keys,
        strong_window=strong_window,
        model=model,
        operation=operation,
        request_key=request_key,
        request_created_ticks=collaborators.engine.now,
        folded_boundaries=folded_boundaries,
    )


def _assignment_of(
    collaborators: StrongWindowCollaborators, held: "_HeldStrongRedo"
) -> StrongAssignment:
    """The job now, or the assignment held until its rounds are stored."""
    crossing = collaborators.retention.context_rounds_in_flight(
        held.key, held.read_keys
    )
    if crossing:
        _log_hold(collaborators, held, crossing)
        return StrongAssignment(
            held.request_key,
            None,
            held_plan=held,
            round_count=held.strong_window.round_count,
            folded_boundaries=held.folded_boundaries,
        )
    job = _strong_job_of(collaborators, held)
    return StrongAssignment(
        held.request_key,
        job,
        round_count=job.round_count,
        folded_boundaries=held.folded_boundaries,
    )


def _stored_rounds_conditions(
    held: "_HeldStrongRedo",
) -> pending_strong_windows.ReleaseConditions:
    """The rounds the row reads, stored in syndrome buffer 1."""
    return pending_strong_windows.ReleaseConditions(
        stored_data_of_operation=held.key[0],
        name="context_stored",
        released_description="strong context stored in syndrome buffer 1",
    )


def _job_once_stored(
    collaborators: StrongWindowCollaborators, held: "_HeldStrongRedo"
) -> Optional[decoding_records.DecodeJob]:
    """The job, once every round the window reads is stored."""
    crossing = collaborators.retention.context_rounds_in_flight(
        held.key, held.read_keys
    )
    if crossing:
        return None
    return _strong_job_of(collaborators, held)


def _log_hold(
    collaborators: StrongWindowCollaborators,
    held: "_HeldStrongRedo",
    crossing: tuple,
) -> None:
    """The rounds the window reads are still on controller_to_strong_buffer."""
    collaborators.engine.log(
        log_sources.DECODER_MANAGER,
        f"{held.label}: strong start deferred until the context "
        f"rounds {list(crossing)} are stored in syndrome buffer 1",
    )


def _strong_job_of(
    collaborators: StrongWindowCollaborators, held: "_HeldStrongRedo"
) -> decoding_records.DecodeJob:
    """The row's job: the rounds its store holds, the faces it pins."""
    weak_job = held.weak_job
    payloads = strong_job_payloads(
        collaborators,
        held.strong_window,
        held.model,
        held.operation,
        held.request_key,
        held.folded_boundaries,
    )
    payload_round_count = decoding_records.distinct_round_count(payloads)
    job = decoding_records.DecodeJob(
        operation_id=held.key[0],
        window_id=held.key[1],
        round_count=payload_round_count,
        ready_time=collaborators.engine.now,
        label=held.label,
        kind=decoding_records.DecodeJobKind.STRONG_REDECODE,
        spatial_nodes=weak_job.spatial_nodes,
        code=weak_job.code,
        detector_error_model=held.model,
        payloads=payloads,
        attempt=1,
        window=held.strong_window,
        strong_decode_for=held.key,
        request_key=held.request_key,
        request_created_ticks=held.request_created_ticks,
        gate=collaborators.builder.gate,
    )
    collaborators.retention.hold_strong_input(job)
    return job


@dataclasses.dataclass(frozen=True)
class _HeldStrongRedo:
    """What a row that waits only for its own rounds keeps from its plan.

    The two rows built on it lay a strong window over the escalated
    window's commit region and build the job as soon as the rounds it
    reads are stored; they differ in the rounds they read and in the
    faces they pin, which is what read_keys and folded_boundaries carry.
    """

    key: tuple
    weak_job: decoding_records.DecodeJob
    label: str
    read_keys: tuple
    strong_window: window_records.Window
    model: object
    operation: program_records.Operation
    request_key: window_records.DecoderRequestKey
    request_created_ticks: int
    folded_boundaries: tuple


@dataclasses.dataclass(frozen=True)
class _HeldForwardWindow:
    """All the forward row keeps from its plan until it builds the job."""

    key: tuple
    weak_job: decoding_records.DecodeJob
    label: str
    resolved_region: strong_regions.ForwardRegion
    strong_window: window_records.Window
    strong_model: object
    operation: program_records.Operation
    strong_request_key: window_records.DecoderRequestKey
    strong_request_created_ticks: int
    folded_boundaries: tuple


def strong_job_payloads(
    collaborators: StrongWindowCollaborators,
    strong_window: window_records.Window,
    model,
    operation: program_records.Operation,
    request_key: window_records.DecoderRequestKey,
    folded_boundaries: tuple,
) -> list:
    """The rounds a strong job reads, and the boundaries it folds into them.

    Both shipped rows fold no boundary: their faces are raw context, so
    the input is the stored rounds of the window, and a mask on a face
    read raw would double count the rounds behind it (Bombin et al.
    2303.04846 lines 775-788). A row that pins a face carries its
    neighbour's committed correction into the input instead: the
    courier ships the committed boundary to this window over
    decoder_to_decoder and writes it into the window's boundary state,
    and the job's gate XORs it into the landed input when the decode
    starts (windows/decode_requests.py, WindowInputGate.mask_input),
    which is the path the weak side's boundaries already take.
    """
    for source_key in folded_boundaries:
        collaborators.courier.pin_strong_face(
            source_key, strong_window, model, operation, request_key
        )
    return collaborators.retention.strong_window_input(
        collaborators.builder, strong_window
    )
