"""The strong window's shape: which rounds the strong tier re-decodes, and when.

Two rows of STRONG_WINDOW_SHAPES (escalation/settings.py), named by
escalation.strong_window
(Toshio et al. 2510.25222). Both pin a face on a neighbour's committed
correction, which is Bombin et al. 2303.04846's input adaptation (lines
775-788): RedoWindow is the escalated window's commit region with
its past face pinned, and DoubleWindow is Sec. III C's forward
extent with both faces pinned. A third shape, both faces pinned and
absorbing nothing, is not a row: it
waits for the window after it, which waits for its own strong result,
and the serial sliding chain deadlocks. What would make it a row is a
windowing scheme whose windows do not commit in one serial chain, the
shape Skoric et al. 2209.08552 decode block by block (lines 398-401,
1038-1040); the row would read that off a fact the scheme declares, the
way it reads absorption off itself, and refuse a scheme that does not
declare it.
RedoWindow is built the moment its rounds are stored in the strong
syndrome buffer. DoubleWindow is Sec. III C and Fig. 12: a strong
window that starts at the escalated commit and extends forward, absorbs
the weak windows it covers, re-slices the window past it (the restart
window) and is held until that window's weak commit or, at the
operation's end, until its last round is stored. A shape builds the
strong job on the window components (planner, tracker, retention,
builder) and hands it to the StrongRedecode, which submits it
(strong_redecode.py); a row that cannot build its job at the escalation
declares what releases it instead (pending_strong_windows.py), and the
redecode holds the assignment until those conditions fire.

The restart window's weak decode may read back into the strong region
from the weak syndrome buffer (escalation.restart_reread_buffer_regions buffer
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

Every row declares the same ports (StrongWindowPorts), so the root
wires a shape without knowing which geometry it is and a row reads only
the components its own layout needs.
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

    job is the strong job when the row builds it now (the redo
    window); None when the row holds it until the conditions it declares
    fire (the double window). held_plan is then the row's own record of
    what it planned, handed back to the row when the redecode asks for
    the job, and round_count is the strong window's rounds, which the
    views name while the job is held. folded_boundaries names the
    neighbour windows whose committed boundary conditions the row folds
    into the job's input, Bombin et al. 2303.04846's input adaptation
    (lines 775-788); a row that reads raw rounds folds none. first_round
    is the strong window's first round, so the carried rounds before it
    are the raw rounds a strong side that forms the events reads
    (windows/round_retention.py, strong_rounds_before).
    """

    request_key: window_records.DecoderRequestKey
    job: Optional[decoding_records.DecodeJob]
    held_plan: Any = None
    round_count: int = 0
    folded_boundaries: tuple = ()
    first_round: int = 1


class StrongWindowPorts:
    """The window components every strong window shape is built on.

    One base so every row of STRONG_WINDOW_SHAPES has one constructor
    signature and the same wires, and the root builds and binds a row
    without asking which geometry it is; a row reads the components its
    own layout needs and ignores the rest. This is gem5's params object,
    where a SimObject's collaborators arrive as one structure rather
    than as a signature per subclass
    (gem5 src/python/m5/SimObject.py:204-205). The courier
    is what a row with a pinned face reads: the committed boundary of
    the neighbour it pins on, and the hop that carries it.
    """

    regions = ports.Port(strong_regions.StrongRegions)
    planner = ports.Port(ports.WindowPlan)
    retention = ports.Port(ports.WindowRetention)
    builder = ports.Port(ports.WindowJobBuilder)
    requester = ports.Port(ports.WindowRequests)
    ledger = ports.Port(ports.LogicalLedger)
    courier = ports.Port(ports.BoundaryCourier)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine


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
    yaml flag. default_boundary_policy is the row of
    BOUNDARY_POLICIES a
    run gets when windows.boundaries is null and the escalation may
    escalate: an absorbing region needs the weak chain to keep
    committing, so it names eager, and a region that absorbs nothing
    names held, since its escalation would revise a boundary already
    shipped. window_absorbed(key, owner_key) is the shape's one trace
    source: the double window fires it for every weak window a strong
    one covers, and a shape that absorbs nothing exposes the silent
    source, so the machine connects the ledger and the trace without
    asking which shape it built.
    """

    absorbs_weak_windows: bool
    default_boundary_policy: str
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

    def rounds_to_carry(self, assignment: StrongAssignment) -> tuple:
        """The rounds the row reads that the strong side does not have yet."""


class RedoWindow(StrongWindowPorts):
    """The redo window: the escalated window decoded again, past face pinned.

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

    The pin is on whatever the neighbour committed (Toshio et al. 2510.25222
    line 1250, boundary conditions "determined by the weak decoder"). This row
    absorbs no weak window, so serial switching gives it Held boundaries
    (escalation/policies.py, check_plan), under which a committed boundary is a
    final one; and the neighbour has committed by the time its dependent
    escalates, since a window's weak decode starts only once every boundary it
    owes has arrived and the escalation follows that decode; a Step 1
    speculative strong decode is planned at that same instant, when the weak job
    leaves its park (strong_redecode.py, parallel_strong_submission). The
    escalated window that has no earlier neighbour pins nothing: its past face
    is the operation's first round layer, closed by the initialisation.

    The job waits for its own rounds stored in the strong syndrome
    buffer, and the rounds the chip still holds are carried up with the
    escalation (the paper's Monte-Carlo simulations set T_comm^strong to
    ten times T_comm^weak, lines 1109-1114); a window at the operation's
    end reads the rounds that exist. It absorbs nothing, so the weak
    chain runs on untouched and there is no restart window.

    Its past face is pinned and its future face is read raw, one buffer
    region. That is not Fig. 12's geometry, where the strong decoder
    starts "after the boundary conditions at both ends have been
    determined by the weak decoder" (Toshio et al. 2510.25222 lines
    1249-1250): pinning the future face too without absorbing the
    window after it is the both-faces shape this module refuses, since
    that window waits for this one's strong result. The double window is
    the Fig. 12 geometry.
    """

    absorbs_weak_windows = False
    default_boundary_policy = "held"
    window_absorbed = trace_source.SILENT

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """The near-pinned job, built now or held for its own rounds."""
        key = (weak_job.operation_id, weak_job.window_id)
        region = self.regions.redo_region(key)
        pinned_source_key = self.regions.redo_pin_source(key)
        folded_boundaries = _declared_faces(pinned_source_key)
        held = _held_redo(
            self,
            weak_job,
            region.window,
            region.context_read_keys,
            folded_boundaries,
        )
        return _assignment_of(self, held)

    def release_conditions(
        self, assignment: StrongAssignment
    ) -> pending_strong_windows.ReleaseConditions:
        """The rounds it reads, stored in the strong syndrome buffer."""
        return _stored_rounds_conditions(assignment.held_plan)

    def held_job(
        self, assignment: StrongAssignment
    ) -> Optional[decoding_records.DecodeJob]:
        """The job, once every round it reads is stored."""
        return _job_once_stored(self, assignment.held_plan)

    def rounds_to_carry(self, assignment: StrongAssignment) -> tuple:
        """The rounds it reads that the strong syndrome buffer lacks."""
        return _rounds_not_stored(self, assignment.held_plan)


class DoubleWindow(StrongWindowPorts):
    """Toshio's double window (Sec. III C, Fig. 12): both faces pinned.

    The window starts at the escalated commit and extends forward by
    the interaction's plan, r_strong = r_com + 2 r_buf rounds (Toshio et
    al. 2510.25222 line 1250); the weak chain skips the windows it
    absorbs and restarts past it on a re-sliced window; the strong
    result owns the whole extent. It is read with no context at all:
    "the strong decoder processes all the assigned data at once, after
    the boundary conditions at both ends have been determined by the
    weak decoder" (lines 1248-1250), and a determined boundary condition
    needs no buffer behind it (Bombin et al. 2303.04846 lines
    1456-1458). Tan et al. 2209.09219 lines 1026-1029 call a window
    closed at both ends a type-2 window, "the entire window is the core
    region".

    The job is held until both of its boundaries are weak-determined:
    the near face is pinned on the window before the escalated one, the
    far face on the restart window's weak commit. This row absorbs the
    windows it covers, so the circular wait that a non-absorbing
    both-faces row runs into does not arise: the weak chain keeps
    committing and the restart window commits the rounds past the
    strong region. The weak pipeline never waits on strong work. One
    strong job per escalation; a second is refused.

    A strong window at the end of the operation has no later window, so
    it has no far pin: its future face is closed by the readout and it
    waits for the terminal data, which is what Tan says of the last
    window (2209.09219 lines 953-955, "both time boundaries of the last
    windows are closed").

    At a back-to-back seam, where the escalated window is the one that
    restarted the weak chain after an earlier strong region, the near
    face pins on that window's own weak commit. The absorption takes its
    dependency out of the chain, so no neighbour commits the round
    before it and it owns the faults crossing that seam itself; the
    region ending at the seam is pinned on its choice of them, so this
    region keeps it: those faults are prior faults of its model, their
    committed effect is folded into its input, and their observable
    flips stay with the window when the strong result replaces its
    prediction.

    Only the region at the operation's first round reads a face open,
    with one buffer region of raw context (Bombin lines 850-852), since
    there is no earlier commit to pin on.

    The restart window owns the faults crossing the far face at every
    escalation.restart_reread_buffer_regions width, since this region
    reads no round past its commit (strong_regions.py,
    double_window_region). At width 1, the width Fig. 12 step 5 draws,
    the restart window reads the region's last buffer region raw as its
    own past context and commits nothing inside the region, so the far
    pin carries only those crossing faults and no round of the input is
    explained twice (Bombin lines 775-788).

    Trace source: window_absorbed(key, owner_key) for every window the
    strong window at owner_key covers.
    """

    absorbs_weak_windows = True
    default_boundary_policy = "eager"

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.window_absorbed = trace_source.TraceSource()

    # ---- the shape

    def plan(self, weak_job: decoding_records.DecodeJob) -> StrongAssignment:
        """Lay out the double window; hold its job until it may start.

        The strong window absorbs the windows it covers. Its job waits for
        the restart window's weak commit (waiting_far_boundary) or, at the
        operation's end, until every clamped strong window round is
        stored (waiting_terminal_data); release_conditions declares which
        of the two, and the redecode holds the assignment until it fires.
        """
        key = (weak_job.operation_id, weak_job.window_id)
        self._refuse_second_escalation(key)
        strong_request_key = self.builder.new_request_key(
            weak_job.operation_id,
            weak_job.window_id,
            window_records.DecoderTier.STRONG,
        )
        strong_request_created_ticks = self.engine.now
        near_source_key = self.regions.double_window_near_face(key)
        resolved_region = self.regions.double_window_region(
            key, near_source_key=near_source_key
        )
        folded_boundaries = _pinned_faces(near_source_key, resolved_region)
        plan = resolved_region.plan
        restart_key = resolved_region.restart_window_key
        strong_window = resolved_region.strong_window
        strong_model = resolved_region.strong_model
        restart_model = resolved_region.restart_model
        guard = self.retention.guard_restart_reads(
            key,
            restart_key,
            resolved_region.proposed_restart_window,
            strong_request_key,
            resolved_region.context_round_keys,
            resolved_region.restart_read_keys,
        )
        operation = self.regions.operation(weak_job.operation_id)
        held = _HeldDoubleWindow(
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
            self.ledger.replace_contributions(
                key,
                plan.commit_lo,
                plan.commit_hi,
                resolved_region.absorbed_window_keys,
            )
            self.retention.hold_strong_context(
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
                first_round=strong_window.start_round,
            )
        finally:
            if guard is not None:
                self.retention.release_strong_hold_if_live(guard)

    def release_conditions(
        self, assignment: StrongAssignment
    ) -> pending_strong_windows.ReleaseConditions:
        """The far boundary, and its own weak commit when it pins on it.

        A region at a back-to-back seam reads its near boundary
        condition off the escalated window's own weak commit, so it
        waits for that commit as it waits for the far one (Toshio et al.
        2510.25222 lines 1248-1250).
        """
        conditions = self._far_face_conditions(assignment.held_plan)
        key = assignment.held_plan.key
        if key not in assignment.folded_boundaries:
            return conditions
        committed_windows = conditions.committed_windows + (key,)
        return dataclasses.replace(
            conditions, committed_windows=committed_windows
        )

    def held_job(
        self, assignment: StrongAssignment
    ) -> Optional[decoding_records.DecodeJob]:
        """The held job, once every round it reads is stored.

        The rounds the chip has are carried up first; a terminal window's
        tail is still being measured, so the store must also have been
        filled through the extent's last round.
        """
        held = assignment.held_plan
        crossing = self.rounds_to_carry(assignment)
        if crossing:
            return None
        stored_through = self.regions.strong_rounds_stored(held.key[0])
        if stored_through < held.resolved_region.plan.context_hi:
            return None
        return self._build_strong_job(held)

    def rounds_to_carry(self, assignment: StrongAssignment) -> tuple:
        """The rounds of its extent that the strong syndrome buffer lacks."""
        held = assignment.held_plan
        return self.retention.context_rounds_in_flight(
            held.key, held.resolved_region.context_round_keys
        )

    # ---- private: what releases the job

    def _far_face_conditions(
        self, held: "_HeldDoubleWindow"
    ) -> pending_strong_windows.ReleaseConditions:
        """The restart window's commit and then its rounds, or the stored tail.

        A strong window bounded by a later weak window waits for that
        window to commit, and then for the rounds it reads to be carried
        up and stored; one at the end of the stream has no later window,
        so it waits for its stored rounds alone (Toshio et al.
        2510.25222 lines 1248-1250).
        """
        restart_key = held.resolved_region.restart_window_key
        if restart_key is None:
            return pending_strong_windows.ReleaseConditions(
                stored_data_of_operation=held.key[0],
                name="terminal_data",
                released_description="terminal data complete",
            )
        return pending_strong_windows.ReleaseConditions(
            committed_windows=(restart_key,),
            stored_data_of_operation=held.key[0],
            name="far_boundary",
            released_description="far-side weak boundary determined",
        )

    # ---- private: the plan

    def _refuse_second_escalation(self, key: tuple) -> None:
        # the plan claims the extent in the ledger before it holds
        # anything, so an escalation of a window already claimed is the
        # second one, held or committed
        if self.ledger.owns_strong_window(key):
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one "
                f"switching event creates exactly one strong job"
            )

    # ---- private: landing the plan

    def _withdraw_stale_requests(
        self, resolved_region: strong_regions.DoubleWindowRegion
    ) -> None:
        """Take back the weak decodes the strong window supersedes.

        An absorbed window's request, and the restart window's request
        built on its old shape: early-shipped at data-complete, parked
        on the escalated window's boundary, which never arrives. Their
        the weak syndrome buffer holds end with them, in flight or landed; the
        restart window's potential restart hold keeps every round its re-sliced
        decode reads, the re-read range among them, until the plan ends.
        """
        stale_keys = list(resolved_region.absorbed_window_keys)
        if resolved_region.restart_window_key is not None:
            stale_keys.append(resolved_region.restart_window_key)
        for window_key in stale_keys:
            window = self.planner.window_at(window_key)
            if window.queued:
                self.requester.withdraw(window)

    def _absorb_window(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """A window the strong window covers is never weak-decoded.

        It counts committed with no logical contribution, the restart
        window no longer waits for it, and its rounds are the strong
        request's.
        """
        self.planner.absorb_window(key, restart_key)
        reads = decoding_records.WindowReads(key)
        self.retention.release_hold_if_live(reads)
        self.retention.release_restart_reads(key)
        self.retention.release_absorbed_strong_hold(
            key, restart_key, replacement
        )
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"window {key} absorbed into the strong window "
            f"(weak chain skips it)",
        )

    def _log_assignment(
        self,
        held: "_HeldDoubleWindow",
        resolved_region: strong_regions.DoubleWindowRegion,
    ) -> None:
        plan = resolved_region.plan
        readiness_description = "the far-side weak boundary"
        if resolved_region.restart_window_key is None:
            readiness_description = "terminal data"
        absorbed_count = len(resolved_region.absorbed_window_keys)
        self.engine.log(
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
        restart = self.planner.window_at(restart_key)
        # its absorbed dependency is gone; no speculative strong decode in the
        # double window
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
        restart = self.planner.reslice_window(restart_key, buffer_lo, model)
        self.retention.replace_window_reads(restart_key, restart)
        seam_owner_name = seam_owner.name.lower()
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"restart window {restart_key} re-sliced across strong window "
            f"edge {strong_window_hi} (reads rounds {restart.buffer_lo}-"
            f"{restart.buffer_hi}; crossing faults owned by "
            f"{seam_owner_name})",
        )

    # ---- private: the held job

    def _build_strong_job(
        self, held: "_HeldDoubleWindow"
    ) -> decoding_records.DecodeJob:
        """The strong window's job, once both of its boundaries exist.

        The strong window commits all r_strong rounds and reads only
        those, its faces pinned, owning nothing that touches rounds
        before its extent.
        """
        payloads = strong_job_payloads(
            self,
            held.strong_window,
            held.strong_model,
            held.operation,
            held.strong_request_key,
            held.folded_boundaries,
        )
        plan = held.resolved_region.plan
        self.retention.require_rounds_retained(
            held.label, payloads, plan.context_lo, plan.context_hi
        )
        return _strong_redecode_job(
            self,
            held.weak_job,
            held.strong_window,
            held.strong_model,
            payloads,
            held.strong_request_key,
            held.strong_request_created_ticks,
        )


# a row that pins no face, the redo window at the operation's
# first round, folds no committed neighbour boundary into its input
FOLDS_NO_BOUNDARY: tuple = ()


def strong_job_payloads(
    shape: StrongWindowPorts,
    strong_window: window_records.Window,
    model,
    operation: program_records.Operation,
    request_key: window_records.DecoderRequestKey,
    folded_boundaries: tuple,
) -> list:
    """The rounds a strong job reads, and the boundaries it folds into them.

    A row that reads a face raw folds no boundary there: the input is
    the stored rounds of the window, and a mask on a face read raw would
    double count the rounds behind it (Bombin et al. 2303.04846 lines
    775-788). A row that pins a face carries its
    neighbour's committed correction into the input instead: the
    courier ships the committed boundary to this window over
    decoder_to_decoder and writes it into the window's boundary state,
    and the job's gate XORs it into the landed input when the decode
    starts (windows/decode_requests.py, WindowInputGate.mask_input),
    which is the path the weak side's boundaries already take.
    """
    for source_key in folded_boundaries:
        shape.courier.pin_strong_face(
            source_key, strong_window, model, operation, request_key
        )
    return shape.retention.strong_window_input(shape.builder, strong_window)


def _declared_faces(pinned_source_key: Optional[tuple]) -> tuple:
    """The faces a row pins, as StrongAssignment names them."""
    if pinned_source_key is None:
        return FOLDS_NO_BOUNDARY
    return (pinned_source_key,)


def _pinned_faces(
    near_source_key: Optional[tuple],
    resolved_region: strong_regions.DoubleWindowRegion,
) -> tuple:
    """The commit closing the near face, and the window restarting after.

    The restart window's weak commit is the far boundary (Toshio et al.
    2510.25222 lines 1253-1259); a terminal region has no restart window
    and no far pin.
    """
    faces = _declared_faces(near_source_key)
    restart_key = resolved_region.restart_window_key
    if restart_key is None:
        return faces
    return faces + (restart_key,)


def _held_redo(
    shape: StrongWindowPorts,
    weak_job: decoding_records.DecodeJob,
    strong_window: window_records.Window,
    read_keys: tuple,
    folded_boundaries: tuple,
) -> "_HeldStrongRedo":
    """What a row keeps from its plan: its window, its reads, its faces."""
    key = (weak_job.operation_id, weak_job.window_id)
    model = shape.regions.redecode_model(key, strong_window, folded_boundaries)
    request_key = shape.builder.new_request_key(
        weak_job.operation_id,
        weak_job.window_id,
        window_records.DecoderTier.STRONG,
    )
    operation = shape.regions.operation(weak_job.operation_id)
    return _HeldStrongRedo(
        key=key,
        weak_job=weak_job,
        label=weak_job.strong_label,
        read_keys=read_keys,
        strong_window=strong_window,
        model=model,
        operation=operation,
        request_key=request_key,
        request_created_ticks=shape.engine.now,
        folded_boundaries=folded_boundaries,
    )


def _assignment_of(
    shape: StrongWindowPorts, held: "_HeldStrongRedo"
) -> StrongAssignment:
    """The job now, or the assignment held until its rounds are stored."""
    crossing = shape.retention.context_rounds_in_flight(
        held.key, held.read_keys
    )
    if crossing:
        _log_hold(shape, held, crossing)
        return StrongAssignment(
            held.request_key,
            None,
            held_plan=held,
            round_count=held.strong_window.round_count,
            folded_boundaries=held.folded_boundaries,
            first_round=held.strong_window.start_round,
        )
    job = _strong_job_of(shape, held)
    return StrongAssignment(
        held.request_key,
        job,
        round_count=job.round_count,
        folded_boundaries=held.folded_boundaries,
        first_round=held.strong_window.start_round,
    )


def _stored_rounds_conditions(
    held: "_HeldStrongRedo",
) -> pending_strong_windows.ReleaseConditions:
    """The rounds the row reads, stored in the strong syndrome buffer."""
    return pending_strong_windows.ReleaseConditions(
        stored_data_of_operation=held.key[0],
        name="context_stored",
        released_description="strong context stored in strong syndrome buffer",
    )


def _job_once_stored(
    shape: StrongWindowPorts, held: "_HeldStrongRedo"
) -> Optional[decoding_records.DecodeJob]:
    """The job, once every round the window reads is stored."""
    crossing = _rounds_not_stored(shape, held)
    if crossing:
        return None
    return _strong_job_of(shape, held)


def _rounds_not_stored(
    shape: StrongWindowPorts, held: "_HeldStrongRedo"
) -> tuple:
    """The rounds the row reads that the strong syndrome buffer lacks."""
    return shape.retention.context_rounds_in_flight(held.key, held.read_keys)


def _log_hold(
    shape: StrongWindowPorts,
    held: "_HeldStrongRedo",
    crossing: tuple,
) -> None:
    """The rounds the window reads are not on the strong side yet."""
    shape.engine.log(
        log_sources.DECODER_MANAGER,
        f"{held.label}: strong start deferred until the context "
        f"rounds {list(crossing)} are stored in strong syndrome buffer",
    )


def _strong_job_of(
    shape: StrongWindowPorts, held: "_HeldStrongRedo"
) -> decoding_records.DecodeJob:
    """The row's job: the rounds its store holds, the faces it pins."""
    payloads = strong_job_payloads(
        shape,
        held.strong_window,
        held.model,
        held.operation,
        held.request_key,
        held.folded_boundaries,
    )
    return _strong_redecode_job(
        shape,
        held.weak_job,
        held.strong_window,
        held.model,
        payloads,
        held.request_key,
        held.request_created_ticks,
    )


def _strong_redecode_job(
    shape: StrongWindowPorts,
    weak_job: decoding_records.DecodeJob,
    strong_window: window_records.Window,
    model,
    payloads: list,
    request_key: window_records.DecoderRequestKey,
    request_created_ticks: int,
) -> decoding_records.DecodeJob:
    """The strong job every row submits, its input held until it lands.

    It re-decodes the escalated weak window, priced for the rounds its
    payloads carry.
    """
    key = (weak_job.operation_id, weak_job.window_id)
    payload_round_count = decoding_records.distinct_round_count(payloads)
    job = decoding_records.DecodeJob(
        operation_id=key[0],
        window_id=key[1],
        round_count=payload_round_count,
        ready_time=shape.engine.now,
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
        request_created_ticks=request_created_ticks,
        gate=shape.builder.gate,
    )
    shape.retention.hold_strong_input(job)
    return job


@dataclasses.dataclass(frozen=True)
class _HeldStrongRedo:
    """What a row that waits only for its own rounds keeps from its plan.

    The redo window lays a strong window over the escalated window's
    commit region and builds the job as soon as the rounds it reads are
    stored; read_keys and folded_boundaries are the rounds it reads and
    the face it pins.
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
class _HeldDoubleWindow:
    """All the double window keeps from its plan until it builds the job."""

    key: tuple
    weak_job: decoding_records.DecodeJob
    label: str
    resolved_region: strong_regions.DoubleWindowRegion
    strong_window: window_records.Window
    strong_model: object
    operation: program_records.Operation
    strong_request_key: window_records.DecoderRequestKey
    strong_request_created_ticks: int
    folded_boundaries: tuple
